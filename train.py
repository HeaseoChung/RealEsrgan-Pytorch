import argparse
import os
import math
import logging

from PIL.Image import RASTERIZE

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from models.models import Generator, Discriminator
from models.loss import VGGLoss, GANLoss
from utils import (
    AverageMeter,
    calc_psnr,
    calc_ssim,
)
from dataset import Dataset

# pid : 6323 python3 train.py --train-file /dataset/data/ --eval-file /dataset/test --outputs-dir weights --num-net-epochs 0 --num-gan-epochs 100000 --resume-g pretrained/RealESRGAN_x4plus.pth  --scale 2 --cuda 3 --patch-size 80

def net_trainer(train_dataloader, eval_dataloader, model, pixel_criterion, net_optimizer, epoch, best_psnr, scaler, writer, device, args):
        model.train()
        losses = AverageMeter(name="PSNR Loss", fmt=":.6f")
        psnr = AverageMeter(name="PSNR", fmt=":.6f")
        
        """  트레이닝 Epoch 시작 """
        for i, (lr, hr) in enumerate(train_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)

            net_optimizer.zero_grad()

            with amp.autocast():
                preds = model(lr)
                loss = pixel_criterion(preds, hr)

            if i == 0:
                vutils.save_image(lr.detach(), os.path.join(args.outputs_dir, f"LR_{epoch}.jpg"))
                vutils.save_image(hr.detach(), os.path.join(args.outputs_dir, f"HR_{epoch}.jpg"))
                vutils.save_image(preds.detach(), os.path.join(args.outputs_dir, f"preds_{epoch}.jpg"))
            
            """ Scaler 업데이트 """
            scaler.scale(loss).backward()
            scaler.step(net_optimizer)
            scaler.update()

            """ Loss 업데이트 """
            losses.update(loss.item(), len(lr))
        
        """ 1 epoch 마다 텐서보드 업데이트 """
        writer.add_scalar('L1Loss/train', losses.avg, epoch)

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
                    'optimizer_state_dict': net_optimizer.state_dict(),
                    'loss': loss,
                    'best_psnr': best_psnr,
                }, os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch))
            )

def gan_trainer(train_dataloader, eval_dataloader, generator, discriminator, pixel_criterion, content_criterion, adversarial_criterion, generator_optimizer, discriminator_optimizer, epoch, best_ssim, scaler, writer, device, args):
    generator.train()
    discriminator.train()

    """ Losses average meter 설정 """
    d_losses = AverageMeter(name="D Loss", fmt=":.6f")
    g_losses = AverageMeter(name="G Loss", fmt=":.6f")
    pixel_losses = AverageMeter(name="Pixel Loss", fmt=":6.4f")
    content_losses = AverageMeter(name="Content Loss", fmt=":6.4f")
    adversarial_losses = AverageMeter(name="adversarial losses", fmt=":6.4f")

    """ 모델 평가 measurements 설정 """
    psnr = AverageMeter(name="PSNR", fmt=":.6f")
    ssim = AverageMeter(name="SSIM", fmt=":.6f")

    """  트레이닝 Epoch 시작 """
    for i, (lr, hr) in enumerate(train_dataloader):
        """LR & HR 디바이스 설정"""
        lr = lr.to(device)
        hr = hr.to(device)

        """ 식별자 최적화 초기화 """
        discriminator_optimizer.zero_grad()

        with amp.autocast():
            """추론"""
            preds = generator(lr)
            """ 식별자 통과 후 loss 계산 """
            real_output = discriminator(hr)
            d_loss_real = adversarial_criterion(real_output, True)

            fake_output = discriminator(preds.detach())
            d_loss_fake = adversarial_criterion(fake_output, False)

            d_loss = (d_loss_real + d_loss_fake) / 2

        """ 가중치 업데이트 """
        scaler.scale(d_loss).backward()
        scaler.step(discriminator_optimizer)
        scaler.update()

        """ 생성자 최적화 초기화 """
        generator_optimizer.zero_grad()

        with amp.autocast():
            """추론"""
            preds = generator(lr)
            """ 식별자 통과 후 loss 계산 """
            real_output = discriminator(hr.detach())
            fake_output = discriminator(preds)
            pixel_loss = pixel_criterion(preds, hr.detach())
            content_loss = content_criterion(preds, hr.detach())
            adversarial_loss = adversarial_criterion(fake_output, True)
            g_loss = 1 * pixel_loss + 1 * content_loss + 0.1 * adversarial_loss

        """ 1 epoch 마다 테스트 이미지 확인 """
        if i == 0:
            vutils.save_image(
                lr.detach(), os.path.join(args.outputs_dir, f"LR_{epoch}.jpg")
            )
            vutils.save_image(
                hr.detach(), os.path.join(args.outputs_dir, f"HR_{epoch}.jpg")
            )
            vutils.save_image(
                preds.detach(), os.path.join(args.outputs_dir, f"preds_{epoch}.jpg")
            )

        """ 가중치 업데이트 """
        scaler.scale(g_loss).backward()
        scaler.step(generator_optimizer)
        scaler.update()

        """ 생성자 초기화 """
        generator.zero_grad()

        """ loss 업데이트 """
        d_losses.update(d_loss.item(), lr.size(0))
        g_losses.update(g_loss.item(), lr.size(0))
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))

    """ 스케줄러 업데이트 """
    discriminator_scheduler.step()
    generator_scheduler.step()

    """ 1 epoch 마다 텐서보드 업데이트 """
    writer.add_scalar("d_Loss/train", d_losses.avg, epoch)
    writer.add_scalar("g_Loss/train", g_losses.avg, epoch)
    writer.add_scalar("pixel_losses/train", pixel_losses.avg, epoch)
    writer.add_scalar("adversarial_losses/train", content_losses.avg, epoch)
    writer.add_scalar("adversarial_losses/train", adversarial_losses.avg, epoch)

    """  테스트 Epoch 시작 """
    generator.eval()
    with torch.no_grad():
        for i, (lr, hr) in enumerate(eval_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            preds = generator(lr)
            psnr.update(calc_psnr(preds, hr), len(lr))
            ssim.update(calc_ssim(preds, hr).mean(), len(lr))

    """ 1 epoch 마다 텐서보드 업데이트 """
    writer.add_scalar("psnr/test", psnr.avg, epoch)
    writer.add_scalar("ssim/test", ssim.avg, epoch)

    """  Best 모델 저장 """
    if ssim.avg > best_ssim:
        best_ssim = ssim.avg
        torch.save(
            generator.state_dict(), os.path.join(args.outputs_dir, "best_g.pth")
        )

    """ Epoch 1000번에 1번 저장 """
    if epoch % 100 == 0:
        """Discriminator 모델 저장"""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": discriminator.state_dict(),
                "optimizer_state_dict": discriminator_optimizer.state_dict(),
            },
            os.path.join(args.outputs_dir, "d_epoch_{}.pth".format(epoch)),
        )

        """ Generator 모델 저장 """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": generator.state_dict(),
                "optimizer_state_dict": generator_optimizer.state_dict(),
                "best_ssim": best_ssim,
            },
            os.path.join(args.outputs_dir, "g_epoch_{}.pth".format(epoch)),
        )


if __name__ == '__main__':
    """ 로그 설정 """
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

    """ Argparse 설정 """
    parser = argparse.ArgumentParser()
    """data args setup"""
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)

    """model args setup"""
    parser.add_argument('--scale', type=int, default=4, required=True)

    """ GAN model args setup"""
    parser.add_argument('--num-net-epochs', type=int, default=100000)
    parser.add_argument('--resume_net', type=str, default='resume_net.pth')
    parser.add_argument('--psnr-lr', type=float, default=0.0001)

    """ GAN model args setup"""
    parser.add_argument('--num-gan-epochs', type=int, default=100000)
    parser.add_argument('--resume-g', type=str, default='resume_genertaor.pth')
    parser.add_argument('--resume-d', type=str, default='resume_discriminator.pth')
    parser.add_argument('--gan-lr', type=float, default=0.0002)
    
    """etc args setup"""
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    
    """ weight를 저장 할 경로 설정 """ 
    args.outputs_dir = os.path.join(args.outputs_dir,  f"RealESRGANx{args.scale}")
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    """ 텐서보드 설정 """
    writer = SummaryWriter(args.outputs_dir)

    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    """ Torch Seed 설정 """
    torch.manual_seed(args.seed)

    """ RealESRGAN psnr 모델 설정 """
    generator = Generator(args.scale).to(device)

    pixel_criterion = nn.L1Loss().to(device)
    net_optimizer = torch.optim.Adam(generator.parameters(), args.psnr_lr, (0.9, 0.99))
    interval_epoch = math.ceil(args.num_net_epochs // 8)
    epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
    net_scheduler = torch.optim.lr_scheduler.MultiStepLR(net_optimizer, milestones=epoch_indices, gamma=0.5)
    scaler = amp.GradScaler()

    total_net_epoch = args.num_net_epochs
    start_net_epoch = 0
    best_psnr = 0

    """ RealESNet 체크포인트 weight 불러오기 """
    if os.path.exists(args.resume_net):
        checkpoint = torch.load(args.resume_net)
        generator.load_state_dict(checkpoint['model_state_dict'])
        net_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_net_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        best_psnr = checkpoint['best_psnr']

    """ RealESRNet 로그 인포 프린트 하기 """
    logger.info(
                f"RealESRNET MODEL INFO:\n"
                f"\tScale factor:                  {args.scale}\n"

                f"RealESRGAN TRAINING INFO:\n"
                f"\tTotal Epoch:                   {args.num_net_epochs}\n"
                f"\tStart Epoch:                   {start_net_epoch}\n"
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
    """NET Training"""
    for epoch in range(start_net_epoch, total_net_epoch):
        net_trainer(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, model=generator, pixel_criterion=pixel_criterion, net_optimizer=net_optimizer, epoch=epoch, best_psnr=best_psnr, scaler=scaler, writer=writer, device=device, args=args)
        net_scheduler.step()


    """ RealESNet 체크포인트 weight 불러오기 """
    discriminator = Discriminator().to(device)

    total_gan_epoch = args.num_gan_epochs
    start_gan_epoch = 0
    best_ssim = 0

    content_criterion = VGGLoss().to(device)
    adversarial_criterion = GANLoss().to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.9, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.9, 0.999))

    interval_epoch = math.ceil(args.num_gan_epochs // 8)
    epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer, milestones=epoch_indices, gamma=0.5)
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_optimizer, milestones=epoch_indices, gamma=0.5)

    """ 체크포인트 weight 불러오기 """
    if os.path.exists(args.resume_g) :
        """ resume generator """
        #checkpoint_g = torch.load(args.resume_g)['params_ema']
        
        # state_dict = generator.state_dict()
        # for n, p in torch.load(args.resume_g,map_location=device)['params_ema'].items():
        #     if n in state_dict.keys():
        #         state_dict[n].copy_(p)
        #     else:
        #         raise RuntimeError("Model error")

        checkpoint_g = torch.load(args.resume_g)
        generator.load_state_dict(checkpoint_g['model_state_dict'])
        start_gan_epoch = checkpoint_g['epoch'] + 1
        generator_optimizer.load_state_dict(checkpoint_g['optimizer_state_dict'])

    if os.path.exists(args.resume_d):
        """ resume discriminator """
        checkpoint_d = torch.load(args.resume_d)
        discriminator.load_state_dict(checkpoint_d['model_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint_d['optimizer_state_dict'])

    """ RealESGAN 로그 인포 프린트 하기 """
    logger.info(
                f"RealESRGAN MODEL INFO:\n"
                f"\tScale factor:                  {args.scale}\n"

                f"RealESRGAN TRAINING INFO:\n"
                f"\tTotal Epoch:                   {args.num_gan_epochs}\n"
                f"\tStart Epoch:                   {start_gan_epoch}\n"
                f"\tTrain directory path:          {args.train_file}\n"
                f"\tTest directory path:           {args.eval_file}\n"
                f"\tOutput weights directory path: {args.outputs_dir}\n"
                f"\tPSNR learning rate:            {args.psnr_lr}\n"
                f"\tPatch size:                    {args.patch_size}\n"
                f"\tBatch size:                    {args.batch_size}\n"
                )

    """GAN Training"""
    for epoch in range(start_gan_epoch, total_gan_epoch):
        gan_trainer(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, generator=generator, discriminator=discriminator, pixel_criterion=pixel_criterion, content_criterion=content_criterion, adversarial_criterion=adversarial_criterion, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, epoch=epoch, best_ssim=best_ssim, scaler=scaler, writer=writer, device=device, args=args)
        discriminator_scheduler.step()
        generator_scheduler.step()