import argparse
import torch
import torch.backends.cudnn as cudnn
import time
from archs.network_arch import Network
import copy
import math
import os
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from pretrain_util import get_logger, AverageMeter, get_datasets, \
    accuracy, evaluate, ProgressMeter, init_distributed_mode, get_rank, get_world_size, NativeScalerWithGradNormCount


def adjust_learning_rate(optimizer, epoch, args, total_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (total_epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def train(train_loader, model, criterion, optimizer, epoch, args, lg):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def main(args):
    init_distributed_mode(args)

    lg = get_logger('Pretraining')
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_id}'
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = args.seed + get_rank()
    torch.manual_seed(seed)

    train_datasets, val_datasets, total_epochs = get_datasets(args, lg)
    if True:  # args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_datasets, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.rank == 0:
            lg.info("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(val_datasets) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                val_datasets, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(val_datasets)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_datasets)
        sampler_val = torch.utils.data.SequentialSampler(val_datasets)

    data_loader_train = torch.utils.data.DataLoader(
        train_datasets, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_datasets, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        lg.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = Network().to(device)
    model_params = sum(map(lambda x: x.numel(), model.parameters()))
    if args.rank == 0:
        lg.info(f'Total params: {model_params:,d}')

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, lg)
        lg.info(f"Accuracy of the network on the {len(val_datasets)}")
        exit(0)

    eff_batch_size = args.batch_size * get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.rank == 0:
        lg.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        lg.info("actual lr: %.2e" % args.lr)
        lg.info("effective batch size: %d" % eff_batch_size)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    current_iter = 0
    total_iters = args.total_iters
    best_top1 = 0
    for epoch in range(total_epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(len(data_loader_train), [batch_time, data_time, losses, top1, top5],
                                 prefix="Epoch: [{}]".format(epoch), )
        model.train()
        end = time.time()
        for i, (images, labels) in enumerate(data_loader_train):
            current_iter += 1
            if current_iter > total_iters:
                break
            adjust_learning_rate(optimizer, i / len(data_loader_train) + epoch, args, total_epochs)
            data_time.update(time.time() - end)
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)

            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if current_iter % args.print_freq == 0:
                if args.rank == 0:
                    progress.display(current_iter, lg)

            if current_iter % args.val_freq == 0:
                if args.rank == 0:
                    test_stats = evaluate(data_loader_val, model, device, lg)
                    if best_top1 < test_stats[0]:
                        best_top1 = test_stats[0]
                        best_weights = copy.deepcopy(model.state_dict())
                        torch.save(best_weights, os.path.join(args.outputs_dir, f'best_{current_iter}.pth'))
                        lg.info('This is a new best record. Saving model.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='/root/autodl-tmp/imagenet/train')
    parser.add_argument('--val_dir', type=str, default='/root/autodl-tmp/imagenet/val')
    parser.add_argument('--outputs_dir', type=str, default='./experiments/pretrain')
    parser.add_argument('--gpu_id', type=int, default=4)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--val_freq', type=int, default=20000)
    parser.add_argument('--print_freq', type=int, default=2000)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--total_iters', type=int, default=10000000)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=10)

    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--schedule', type=int, default=[120, 160], nargs="*")
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.set_defaults(pin_mem=True)

    args = parser.parse_args()
    main(args)


