import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models.models import Generator
from utils import preprocess, get_concat_h


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = Generator(args.scale).to(device)
    state_dict = model.state_dict()

    for n, p in torch.load(args.weights_file, map_location=device)[
        "params_ema"
    ].items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise RuntimeError("Model error")

    model.eval()

    image = pil_image.open(args.image_file).convert("RGB")
    bicubic = image.resize(
        (image.width * args.scale, image.height * args.scale),
        resample=pil_image.BICUBIC,
    )
    bicubic.save(args.image_file.replace(".", "_bicubic_x{}.".format(args.scale)))

    lr = preprocess(image).to(device)

    with torch.no_grad():
        preds = model(lr)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0)

    output = np.array(preds).transpose([1, 2, 0])
    output = np.clip(output, 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace(".", "_Real_ESRGAN."))

    if args.merge:
        merge = get_concat_h(bicubic, output).save(
            args.image_file.replace(".", "_hconcat_.")
        )
