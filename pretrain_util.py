import logging
import colorlog
import sys
import time
from torchvision import datasets, transforms
import torch
import math
import os
import builtins
import torch.distributed as dist
import datetime


def get_logger(description):
    time_tuple = time.localtime(time.time())
    cur_time = "time_{}-{}-{}".format(time_tuple[3], time_tuple[4], time_tuple[5])
    path = './experiments/' + cur_time + '.log'

    lg = logging.getLogger(description)
    lg.setLevel(logging.INFO)

    stream_hd = logging.StreamHandler(sys.stdout)
    file_hd = logging.FileHandler(path)

    formatter_console = colorlog.ColoredFormatter('%(asctime)s | %(message)s',
                                                  datefmt='%Y-%m-%d %H:%M:%S')
    formatter_file = logging.Formatter('%(asctime)s |[%(lineno)03d]%(filename)-11s | %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    stream_hd.setFormatter(formatter_console)
    file_hd.setFormatter(formatter_file)

    # add a Handler for StreamHandler and FileHandler respectively
    lg.addHandler(stream_hd)
    lg.addHandler(file_hd)
    return lg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, lg):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        lg.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_datasets(args, lg):
    train_transforms = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomResizedCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dir = args.train_dir
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)

    val_dir = args.val_dir
    val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)

    rank = 8
    num_iter_per_epoch = math.ceil(len(train_datasets) / args.batch_size / rank)
    total_iters = int(args.total_iters)
    total_epochs = math.ceil(total_iters / num_iter_per_epoch)
    lg.info('Training statistics:'
            f'\n\tNumber of train images: {len(train_datasets)}'
            f'\n\tBatch size per gpu: {args.batch_size}'
            f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
            f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
    return train_datasets, val_datasets, total_epochs


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True) # 求tensor中某个dim的前k大或者前k小的值以及对应的index
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


@torch.no_grad()
def evaluate(data_loader, model, device, lg):
    criterion = torch.nn.CrossEntropyLoss()
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    model.eval()
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, labels)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        batch_size = images.shape[0]
        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

    lg.info('Acc@1 {:.3f} Acc@5 {:.3f} loss {:.3f}'.format(top1.avg, top5.avg, losses.avg))
    #lg.info(f"Acc@1 {top1.avg} Acc@5 {top5.avg} loss {losses.avg}")
    return top1.avg, top5.avg, losses.avg


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

