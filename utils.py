import torch
import numpy as np
from pytorch_msssim import ssim
import PIL.Image as pil_image

def check_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])

# 전처리 과정 함수
def preprocess(img):
    # uInt8 -> float32로 변환
    x = np.array(img).astype(np.float32)
    x = x.transpose([2,0,1])
    # Normalize x 값
    x /= 255.
    # 넘파이 x를 텐서로 변환
    x = torch.from_numpy(x)
    # x의 차원의 수 증가
    x = x.unsqueeze(0)
    # x 값 반환
    return x

def calc_psnr(img1, img2):
    """ PSNR 계산 """
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def calc_ssim(img1, img2):
    """ SSIM 계산 """
    return ssim(img1, img2, data_range=1,size_average=False)

def calc_lpips(img1, img2, lpips_metric):
    """ LPIPS 계산 """
    return lpips_metric(img1, img2)

def get_concat_h(img1, img2):
    merge = pil_image.new('RGB', (img1.width + img2.width, img1.height))
    merge.paste(img1, (0, 0))
    merge.paste(img2, (img2.width, 0))
    return merge


# Copy from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# Copy from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'