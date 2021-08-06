# python train.py --train-file data/train --eval-file data/eval --outputs-dir models --scale 3
# python train.py --train-file data/train --eval-file data/eval --outputs-dir models --scale 3 --checkpoint-file 
import argparse
import os
import math
import logging

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from models import Generator
from utils import AverageMeter, calc_psnr
from dataset import Dataset

# 52655
# BSRNETv2: 8362
if __name__ == '__main__':
    """ 로그 설정 """
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

    """ Argparse 설정 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--psnr-lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=10000)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--checkpoint-file', type=str, default='checkpoint-file.pth')
    args = parser.parse_args()
    
    """ weight를 저장 할 경로 설정 """ 
    args.outputs_dir = os.path.join(args.outputs_dir,  f"BSRGANx{args.scale}")
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    """ 텐서보드 설정 """
    writer = SummaryWriter(args.outputs_dir)

    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    """ Torch Seed 설정 """
    torch.manual_seed(args.seed)

    """ FUNIE 모델 설정 """
    model = Generator().to(device)

    """ Loss 및 Optimizer 설정 """
    pixel_criterion = nn.L1Loss().to(device)
    psnr_optimizer = torch.optim.Adam(model.parameters(), args.psnr_lr, (0.9, 0.99))
    interval_epoch = math.ceil(args.num_epochs // 8)
    epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
    #psnr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(psnr_optimizer, psnr_epoch_indices, 1, 1e-7)
    psnr_scheduler = torch.optim.lr_scheduler.MultiStepLR(psnr_optimizer, milestones=[1000,2000,3000,4000], gamma=0.5)
    scaler = amp.GradScaler()

    total_epoch = args.num_epochs
    start_epoch = 0
    best_psnr = 0

    """ 체크포인트 weight 불러오기 """
    if os.path.exists(args.checkpoint_file):
        checkpoint = torch.load(args.checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        psnr_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        best_psnr = checkpoint['best_psnr']

    """ 로그 인포 프린트 하기 """
    logger.info(
                f"FUNIE MODEL INFO:\n"

                f"FUNIE TRAINING INFO:\n"
                f"\tTotal Epoch:                   {args.num_epochs}\n"
                f"\tStart Epoch:                   {start_epoch}\n"
                f"\tTrain directory path:          {args.train_file}\n"
                f"\tTest directory path:           {args.eval_file}\n"
                f"\tOutput weights directory path: {args.outputs_dir}\n"
                f"\tPSNR learning rate:            {args.psnr_lr}\n"
                f"\tPatch size:                    {args.patch_size}\n"
                f"\tBatch size:                    {args.batch_size}\n"
                )

    """ 데이터셋 & 데이터셋 설정 """
    train_dataset = Dataset(args.train_file, args.patch_size, args.scale)
    train_dataloader = DataLoader(
                            dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True
                        )
    eval_dataset = Dataset(args.eval_file, args.patch_size, args.scale)
    eval_dataloader = DataLoader(
                                dataset=eval_dataset, 
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True
                                )
    
    """ 트레이닝 시작 & 테스트 시작"""
    for epoch in range(start_epoch, total_epoch):
        model.train()
        losses = AverageMeter(name="PSNR Loss", fmt=":.6f")
        psnr = AverageMeter(name="PSNR", fmt=":.6f")
        
        """  트레이닝 Epoch 시작 """
        for i, (lr, hr) in enumerate(train_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)

            psnr_optimizer.zero_grad()

            with amp.autocast():
                preds = model(lr)
                loss = pixel_criterion(preds, hr)

            if i == 0:
                vutils.save_image(lr.detach(), os.path.join(args.outputs_dir, f"LR_{epoch}.jpg"))
                vutils.save_image(hr.detach(), os.path.join(args.outputs_dir, f"HR_{epoch}.jpg"))
                vutils.save_image(preds.detach(), os.path.join(args.outputs_dir, f"preds_{epoch}.jpg"))
            
            """ Scaler 업데이트 """
            scaler.scale(loss).backward()
            scaler.step(psnr_optimizer)
            scaler.update()

            """ Loss 업데이트 """
            losses.update(loss.item(), len(lr))
        
        """ 1 epoch 마다 텐서보드 업데이트 """
        writer.add_scalar('L1Loss/train', losses.avg, epoch)

        psnr_scheduler.step()

        """  테스트 Epoch 시작 """
        model.eval()

        for i, (lr, hr) in enumerate(eval_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            with torch.no_grad():
                preds = model(lr)
            psnr.update(calc_psnr(preds, hr), len(lr))
    
        """ 1 epoch 마다 텐서보드 업데이트 """
        writer.add_scalar('psnr/test', psnr.avg, epoch)

        if psnr.avg > best_psnr:
            best_psnr = psnr.avg
            torch.save(
                model.state_dict(), os.path.join(args.outputs_dir, 'best.pth')
            )

        if epoch % 10 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': psnr_optimizer.state_dict(),
                    'loss': loss,
                    'best_psnr': best_psnr,
                }, os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch))
            )
