import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch import nn

from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img
from utils.registry import MODEL_REGISTRY
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


@MODEL_REGISTRY.register()
class IRConStyleModel(BaseModel):

    def __init__(self, opt):
        super(IRConStyleModel, self).__init__(opt)
        self.Test = self.opt.get('test', False)
        if not self.Test:
            self.itreation = 0
            self.ConStyle_iter = int(self.opt['train'].get('ConStyle_iter', None))
            self.use_constyle = True
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        else:
            self.mixing_flag = False

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = 'params_ema'
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        load_constyle = self.opt['path'].get('pretrain_constyle', None)
        if load_constyle is not None:
            param_key = 'params_ema'
            self.load_network(self.net_g, load_constyle, self.opt['path'].get('strict_load_constyle', True), param_key, True)

        if self.is_train:
            self.init_training_settings()

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
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

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

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def optimize_parameters(self):
        if self.itreation <= self.ConStyle_iter and self.use_constyle:
            self.use_constyle = True
        else:
            self.use_constyle = False
            for param_q, param_k in zip(
                self.net_g.ConStyle.encoder_q.parameters(), self.net_g.ConStyle.encoder_k.parameters()
            ):
                param_k.requires_grad = False
                param_q.requires_grad = False
                
        self.optimizer_g.zero_grad()
        query, pos, neg, self.logits, self.labels, feas = self.net_g.ConStyle(self.lq, self.gt)
        self.output = self.net_g(self.lq, query[1], feas)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        if self.use_constyle:
            if self.cri_CrossEntropy:
                l_CrossEntropy = self.cri_CrossEntropy(self.logits, self.labels)
                l_total += l_CrossEntropy
                loss_dict['l_CrossEntropy'] = l_CrossEntropy

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

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    ############
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
                query, feas = self.net_g_ema.ConStyle.encoder_q(self.lq)
                self.output = self.net_g_ema(self.lq, query[1], feas)
                # self.output = self.net_g_ema(self.lq, feas)
        else:
            self.net_g.eval()
            with torch.no_grad():
                query, feas = self.net_g.ConStyle.encoder_q(self.lq)
                self.output = self.net_g(self.lq, query[1], feas)
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
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                # pbar.set_description(f'Test {img_name}')
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
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += f'\tBest Record: {self.best_metric_results[dataset_name][metric]["val"]:.4f}'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

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
