import torch
from natsort import natsorted
from glob import glob
import cv2
import numpy as np
from skimage import img_as_ubyte
import os
from tqdm import tqdm
import torch.nn.functional as F


def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    # Origin MAXIM
    # from archs.Maxim_arch import Maxim
    # weights = '/home/fandongqi/workspace/constylev2/experiments/ConStyle_v2_ok/Origin_MAXIM/models/net_g_696000.pth'
    # inp_dir = '/data/fandongqi/rain_real_rain100H/'
    # result_dir = '/home/fandongqi/workspace/constylev2/results/rain_maxim/'

    # ConStyle v2 MAXIM
    # from archs.ConStyle_Maxim_arch import ConStyleMaxim
    # weights = '/home/fandongqi/workspace/constylev2/experiments/ConStyle_v2_ok/Finally_MAXIM/models/net_g_696000.pth'
    # inp_dir = '/data/fandongqi/rain_real_rain100H/'
    # result_dir = '/home/fandongqi/workspace/constylev2/results/rain_constyle_maxim/'

    # Origin NAFNet
    # from archs.NAFNet_arch import NAFNet
    # weights = '/home/fandongqi/workspace/constylev2/experiments/ConStyle_v2_ok/Origin_NAFNet/models/net_g_696000.pth'
    # inp_dir = '/data/fandongqi/rain_real_rain100H/'
    # result_dir = '/home/fandongqi/workspace/constylev2/results/rain_nafnet/'

    # ConStyle v2 NAFNet
    # from archs.ConStyle_NAFNet_arch import ConStyleNAFNet
    # weights = '/home/fandongqi/workspace/constylev2/experiments/ConStyle_v2_ok/Finally_NAFNet/models/net_g_648000.pth'
    # inp_dir = '/data/fandongqi/rain_real_rain100H/'
    # result_dir = '/home/fandongqi/workspace/constylev2/results/rain_constyle_nafnet/'

    # # ConStyle v2 Restormer
    # from archs.ConStyle_Restormer_arch import ConStyleRestormer
    # weights = '/home/fandongqi/workspace/constylev2/experiments/ConStyle_v2_ok/Finally_Restormer/models/net_g_672000.pth'
    # inp_dir = '/data/fandongqi/rain_real_rain100H/'
    # result_dir = '/home/fandongqi/workspace/constylev2/results/rain_constyle_restormer/'

    # Origin Conv
    # from archs.Conv_arch import OriginConv
    # weights = '/home/fandongqi/workspace/constylev2/experiments/ConStyle_v2_ok/Origin_Conv/models/net_g_600000.pth'
    # inp_dir = '/data/fandongqi/rain_real_rain100H/'
    # result_dir = '/home/fandongqi/workspace/constylev2/results/rain_conv/'

    # # ConStyle v2 Conv
    from archs.ConStyle_Conv_arch import ConStyleConv
    weights = '/home/fandongqi/workspace/constylev2/experiments/ConStyle_v2_ok/Finally_Conv/models/net_g_384000.pth'
    inp_dir = '/data/fandongqi/rain_real_rain100H/'
    result_dir = '/home/fandongqi/workspace/constylev2/results/rain_constyle_conv/'


    is_origin = False
    if 'Origin' in weights: is_origin = True

    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    os.makedirs(result_dir, exist_ok=True)
    factor = 64

    if is_origin:
        # model = Maxim()
        # model = NAFNet()
        model = OriginConv()

        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['params_ema'])
        model.eval()
    else:
        # model = ConStyleMaxim(Train=False)
        # model = ConStyleNAFNet(Train=False)
        # model = ConStyleRestormer(Train=False)
        model = ConStyleConv(Train=False)

        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['params_ema'], strict=False)
        model.eval()

    with torch.no_grad():
        for file_ in tqdm(files):
            img = np.float32(load_img(file_)) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0)

            # Padding in case images are not multiples of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if is_origin:
                restored = model(input_)
            else:
                query, feas = model.ConStyle.encoder_q(input_)
                restored = model(input_, query[1], feas)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            name = os.path.splitext(os.path.split(file_)[-1])[0] + '.png'
            save_img(os.path.join(result_dir, name), img_as_ubyte(restored))
            