import argparse
from os import terminal_size

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import Generator
from utils import preprocess, get_concat_h

import time
# python test.py --weights-file weights/LDSR_DENSE_light_DEGRADE_x4_2_14x256/best.pth --image-file examples/sweden_samples/192899_061337_updates.jpg --scale 4 --merge
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = Generator().to(device)
    state_dict = model.state_dict()
    try:
        for n, p in torch.load(args.weights_file,map_location=device).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
    except:
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage)['model_state_dict'].items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    #image = image.resize((1024, 1024), resample=pil_image.BICUBIC)
    lr = preprocess(image).to(device)

    with torch.no_grad():
        preds = model(lr)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0)

    output = np.array(preds).transpose([1,2,0])
    output = np.clip(output, 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', 'RRBNET_UNET.'))

    if args.merge:
        get_concat_h(image, output).save(args.image_file.replace('.', '_LDSR_hconcat_.'))