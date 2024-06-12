import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
import cv2
import random
from wand.image import Image as WandImage
from scipy.ndimage import zoom as scizoom
from io import BytesIO
from PIL import Image as PILImage
from wand.api import library as wandlibrary
import wand.color as WandColor
from PIL import Image


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    w = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))

    top = (h - ch) // 2
    left = (w - cw) // 2
    img = scizoom(img[top:top + ch, left:left + cw], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[trim_top:trim_top + h, trim_left:trim_left + w]


def contrast(x, severity=3):
    c = [.4, .3, .2, .1, 0.05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def snow(x, severity=3):
    c = [(0.1,0.2,1,0.6,8,3,0.8),
         (0.1,0.2,1,0.5,10,4,0.8),
         (0.15,0.3,1.75,0.55,10,4,0.7),
         (0.25,0.3,2.25,0.6,12,6,0.65),
         (0.3,0.3,1.25,0.65,14,12,0.6)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    h, w = x.shape[0], x.shape[1]
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255



def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def noise_generate(file_):
    img = np.float32(load_img(file_)) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    sigma_value = random.uniform(10, 50)
    noise_level = torch.FloatTensor([sigma_value]) / 255.0
    noise = torch.randn(img.size()).mul_(noise_level).float()
    img.add_(noise)
    img = torch.clamp(img, 0, 1).cpu().detach().permute(1, 2, 0).numpy()
    name = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img(os.path.join(args.result_dir, name + '.png'), img_as_ubyte(img))


def jpeg_generate(file_):
    img = np.float32(load_img(file_))
    quality_factor = random.randint(10, 50)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 3)
    name = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img(os.path.join(args.result_dir, name + '.jpg'), img)


def snow_generate(file_):
    img = Image.open(file_).convert('RGB')
    img = snow(img)
    name = os.path.splitext(os.path.split(file_)[-1])[0]
    Image.fromarray(np.uint8(img)).save(os.path.join(args.result_dir, name + '.png'))


def contrast_generate(file_):
    img = Image.open(file_).convert('RGB')
    img = contrast(img)
    name = os.path.splitext(os.path.split(file_)[-1])[0]
    Image.fromarray(np.uint8(img)).save(os.path.join(args.result_dir, name + '.png'))


def contrast_snow_generate(file_):
    img = Image.open(file_).convert('RGB')
    img = contrast(img)
    img = snow(img)
    name = os.path.splitext(os.path.split(file_)[-1])[0]
    Image.fromarray(np.uint8(img)).save(os.path.join(args.result_dir, name + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--input_dir', default='/hd-data/fandongqi/Mix/Noise/hq', type=str)
    parser.add_argument('--result_dir', default='/hd-data/fandongqi/Mix/Noise/lq', type=str)
    args = parser.parse_args()

    inp_dir = os.path.join(args.input_dir)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')) + glob(os.path.join(inp_dir, '*.bmp')))
    with torch.no_grad():
        for file_ in tqdm(files):
            # Change the [input_dir] and [result_dir] to generate degradation images (noise, jpeg, snow, contrast+snow)

            noise_generate(file_)
            # jpeg_generate(file_)
            # snow_generate(file_)
            # contrast_snow_generate(file_)





