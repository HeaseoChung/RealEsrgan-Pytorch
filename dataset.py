import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from degradation.degradation import Degradation
from utils import check_image_file


# 데이터 셋 생성 클래스
class Dataset(object):
    def __init__(self, images_dir, image_size, upscale_factor):
        deg = Degradation()
        self.filenames = [
            os.path.join(images_dir, x)
            for x in os.listdir(images_dir)
            if check_image_file(x)
        ]
        self.lr_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(deg.degradation_pipeline),
                transforms.ToTensor(),
            ]
        )
        self.hr_transforms = transforms.Compose(
            [
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )

    # lr & hr 이미지를 읽고 크롭하여 lr & hr 이미지를 반환하는 함수
    def __getitem__(self, idx):
        hr = self.hr_transforms(Image.open(self.filenames[idx]).convert("RGB"))
        lr = self.lr_transforms(hr)
        return lr, hr

    def __len__(self):
        return len(self.filenames)
