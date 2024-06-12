import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch import nn
import random
import numpy as np
import math
from enum import Enum
from torch.nn import Softmax
from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img, DiffJPEG, USMSharp, filter2D
from utils.registry import MODEL_REGISTRY
from data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from .base_model import BaseModel
import torch.nn.functional as F


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))  # 准确率
        return res


class ConStyleModel(BaseModel):
    def __init__(self, opt):
        super(ConStyleModel, self).__init__(opt)
        self.class_iter = int(self.opt['train'].get('class_iter', None))
        self.kl_iter = int(self.opt['train'].get('kl_iter', None))
        self.class_use = True
        self.kl_use = True
        self.itreation = 0

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 160)
        self.train_top1 = AverageMeter('Acc@1', ':6.2f')
        self.train_top5 = AverageMeter('Acc@5', ':6.2f')

        if self.mixing_flag:
            print("mixing_flag")
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path_g = self.opt['path'].get('pretrain_network_g', None)
            if load_path_g is not None:
                self.load_network(self.net_g_ema, load_path_g, self.opt['path'].get('strict_load_g', True),
                                  'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('CrossEntropy_opt'):
            self.cri_CrossEntropy = torch.nn.CrossEntropyLoss().to(self.device)
        else:
            self.cri_CrossEntropy = None

        if train_opt.get('content_opt'):
            self.cri_content = build_loss(train_opt['content_opt']).to(self.device)
        else:
            self.cri_content = None

        if train_opt.get('style_opt'):
            self.cri_style = build_loss(train_opt['style_opt']).to(self.device)
        else:
            self.cri_style = None

        self.cri_cosine = nn.CosineSimilarity(dim=1).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    @torch.no_grad()
    def feed_data(self, data):
        self.gt, self.classes = data['original_tuple']
        self.lq = data['lq']
        self.flags = data['flags']
        self.lq = self.lq.to(self.device)
        self.gt = self.gt.to(self.device)
        self.classes = self.classes.to(self.device)

        bs, ch, hight, width = self.gt.size()
        clean_position = torch.randn(bs)
        degraded_position = torch.randn(bs)

        clean_num = 0
        degraded_num = 0
        for i in range(bs):
            # flags indicate that each img is the hq or degraded in a batch
            if self.flags[i]:
                degraded_position[degraded_num] = i
                degraded_num = degraded_num + 1
            else:
                clean_position[clean_num] = i
                clean_num = clean_num + 1

        positions = torch.concat((
            degraded_position[:degraded_num], clean_position[:clean_num]
        )).to(self.device, dtype=torch.int64)

        self.classes = torch.gather(self.classes, dim=0, index=positions)
        positions = positions.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, ch, hight, width)
        self.lq = torch.gather(self.lq, dim=0, index=positions)
        self.gt = torch.gather(self.gt, dim=0, index=positions)
        degraded = self.lq[:degraded_num]
        clean = self.lq[degraded_num:]

        self.lq = degraded
        if clean_num != 0:
            self.gt_usm = self.usm_sharpener(clean)
            self.kernel1 = data['kernel1'][degraded_num:].to(self.device)
            self.kernel2 = data['kernel2'][degraded_num:].to(self.device)
            self.sinc_kernel = data['sinc_kernel'][degraded_num:].to(self.device)
            real_esrgan = self.opt['datasets']['train']['real_esrgan']
            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], real_esrgan['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, real_esrgan['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(real_esrgan['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = real_esrgan['gray_noise_prob']
            if np.random.uniform() < real_esrgan['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=real_esrgan['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=real_esrgan['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*real_esrgan['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < real_esrgan['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], real_esrgan['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, real_esrgan['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(real_esrgan['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = real_esrgan['gray_noise_prob2']
            if np.random.uniform() < real_esrgan['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=real_esrgan['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=real_esrgan['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*real_esrgan['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*real_esrgan['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            self.lq = torch.cat((degraded, torch.clamp((out * 255.0).round(), 0, 255) / 255.), dim=0)


    def optimize_parameters(self):
        self.optimizer_g.zero_grad()
        query, pos, neg, logits, labels, feas, predict = self.net_g(self.lq, self.gt)

        l_total = 0
        loss_dict = OrderedDict()

        if self.itreation <= self.class_iter and self.class_use:
            l_CrossEntropy = self.cri_CrossEntropy(predict[0], self.classes)
            l_total += l_CrossEntropy
            loss_dict['l_CrossEntropy'] = l_CrossEntropy

        if self.itreation <= self.kl_iter and self.kl_use:
            l_KL_div = F.kl_div(predict[0].softmax(-1).log(), predict[1].softmax(-1), reduction='batchmean')
            l_total += l_KL_div
            loss_dict['l_KL_div'] = l_KL_div

        if self.cri_CrossEntropy:
            l_InfoNCE = self.cri_CrossEntropy(logits, labels)
            l_total += l_InfoNCE
            loss_dict['l_InfoNCE'] = l_InfoNCE

        if self.cri_content:
            l_content = self.cri_content(query[1], pos)
            l_total += l_content
            loss_dict['l_content'] = l_content

        if self.cri_style:
            neg[0] = neg[0].to(self.device)
            neg[1] = neg[1].to(self.device)

            n = query[0].shape[-1] * query[0].shape[-2]
            similarity_query1 = torch.matmul(query[0], query[0].T) / n
            similarity_neg1 = torch.matmul(neg[0], neg[0].T) / n

            l_style = self.cri_style(similarity_query1, similarity_neg1)
            similarity_query2 = torch.matmul(query[1], query[1].T) / n
            similarity_neg2 = torch.matmul(neg[1], neg[1].T) / n
            l_style = l_style + self.cri_style(similarity_query2, similarity_neg2)
            l_total += l_style
            loss_dict['l_style'] = l_style

        acc1, acc5 = accuracy(predict[0], self.classes, topk=(1, 5))

        self.train_top1.update(acc1[0], self.classes.size(0))
        self.train_top5.update(acc5[0], self.classes.size(0))

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def calc_mean_std(self, feat, eps=1e-5):
        latent_var = feat.contiguous().view(-1).var() + eps
        latent_std = latent_var.sqrt()
        latent_mean = feat.contiguous().view(-1).mean()
        return latent_mean, latent_std

    def calc_style_loss(self, input, target):
        input_mean, input_std = self.calc_mean_std(input)
        target_mean, target_std = self.calc_mean_std(target)
        loss_mean, loss_std = self.weighted_mse_loss_merge(input_mean, target_mean, input_std, target_std)
        return loss_mean + loss_std

    def weighted_mse_loss_merge(self, input_mean, target_mean, input_std, target_std):
        loss_mean = ((input_mean - target_mean) ** 2)
        loss_std = ((input_std - target_std) ** 2)
        return loss_mean.mean(), loss_std.mean()

    ############

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                _, _, self.output = self.net_g_ema.encoder_q(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                _, _, self.output = self.net_g.encoder_q(self.lq)

            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = 'ImageNet'
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        # metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader))

        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            # tentative for out of GPU memory
            del self.lq
            del self.gt
            #del self.gt_usm
            torch.cuda.empty_cache()

            if with_metrics:
                acc1, acc5 = accuracy(self.output, self.classes, topk=(1, 5))
                self.metric_results['top1'] += acc1
                self.metric_results['top5'] += acc5
            if use_pbar:
                pbar.update(1)
                pbar.set_description('Test img')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value.item():.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += f'\tBest Record: {self.best_metric_results[dataset_name][metric]["val"].item():.4f}'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value.item(), current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)
