import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import time

from models import LDSR_V4
from utils import preprocess, calc_psnr, calc_ssim, calc_lpips, check_image_file, get_concat_h, AverageMeter
from lpips import LPIPS


def print_profile(sr_psnr, sr_ssim, sr_lpips, bic_psnr, bic_ssim, bic_lpips):
    print('{}_{} PSNR: {:.2f}'.format("SR", img_path.split('/')[-1],sr_psnr))
    print('{}_{} SSIM: {:.2f}'.format("SR", img_path.split('/')[-1],sr_ssim))
    print('{}_{} LPIPS: {:.2f}'.format("SR", img_path.split('/')[-1],sr_lpips))
        
    print('{}_{} PSNR: {:.2f}'.format("Bicubic", img_path.split('/')[-1], bic_psnr))
    print('{}_{} SSIM: {:.2f}'.format("Bicubic", img_path.split('/')[-1], bic_ssim))
    print('{}_{} LPIPS: {:.2f}'.format("Bicubic", img_path.split('/')[-1], bic_lpips))

def calc_metrics(hr, preds):
    return calc_psnr(hr, preds), calc_ssim(hr,preds)[0], calc_lpips(hr,preds, lpips)[0][0][0][0]


# python test.py --weights-file weights/ --image examples/ --scale 4
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    
    output_dir = os.path.join("examples", f"LDSR_x{args.scale}_{args.images_dir.split('/')[-1]}")
    psnr_avg = AverageMeter(name="PSNR", fmt=":.6f")
    ssim_avg = AverageMeter(name="SSIM", fmt=":.6f")
    lpips_avg = AverageMeter(name="LPIPS", fmt=":.6f")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    lpips = LPIPS(net='vgg').to(device)

    model = LDSR_V4(scale_factor=args.scale).to(device)

    try:
        model.load_state_dict(torch.load(args.weights_file, map_location=device))
    except:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
    
    model.eval()

    images_path = [os.path.join(args.images_dir, x) for x in os.listdir(args.images_dir) if check_image_file(x)]

    for img_path in images_path:
        image = pil_image.open(img_path).convert("RGB")

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale

        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        bicubic.save(os.path.join(output_dir, f"bicubic_x{args.scale}_{img_path.split('/')[-1]}"))

        lr = preprocess(lr).to(device)
        hr = preprocess(hr).to(device)
        bic = preprocess(bicubic).to(device)

        with torch.no_grad():
            start = time.time()
            preds = model(lr)
            print(f"SR TIME : {time.time()-start}")

        """ bicubic quality measurement"""
        bic_psnr, bic_ssim, bic_lpips = calc_metrics(hr, bic)

        """ SR quality measurement"""
        sr_psnr, sr_ssim, sr_lpips = calc_metrics(hr, preds)
        psnr_avg.update(sr_psnr, len(images_path))
        ssim_avg.update(sr_ssim, len(images_path))
        lpips_avg.update(sr_lpips, len(images_path))

        """ print results """
        print_profile(sr_psnr, sr_ssim, sr_lpips, bic_psnr, bic_ssim, bic_lpips)

        """ Post Process """
        preds = preds.mul(255.0).cpu().numpy().squeeze(0)
        output = np.array(preds).transpose([1,2,0])
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(os.path.join(output_dir, f"LDSR_V4_x{args.scale}_{img_path.split('/')[-1]}"))

        if args.merge:
            merge = get_concat_h(bicubic, output).save(os.path.join(output_dir, f"hconcat_{img_path.split('/')[-1]}"))

    print(f"AVG PSNR : {psnr_avg.avg}")
    print(f"AVG SSIM : {ssim_avg.avg}")
    print(f"AVG LPIPS : {lpips_avg.avg}")
