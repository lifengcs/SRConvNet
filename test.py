import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
from torchvision import utils as vutils
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from datas.utils import create_datasets
from multiprocessing import Process
from multiprocessing import Queue
import time


class save_img():
    def __init__(self):
        self.n_processes = 32

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, filename, img):
        tensor_cpu = img[0].byte().permute(1, 2, 0).cpu()
        self.queue.put((filename, tensor_cpu))


parser = argparse.ArgumentParser(description='config')

parser.add_argument('--config', type=str, default=None, help='pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help='resume training or not')
parser.add_argument('--custom', type=str, default=None, help='use custom block')
parser.add_argument('--cloudlog', type=str, default=None, help='use cloudlog')

device = None

args = parser.parse_args()

if args.config:
    opt = vars(args)
    yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(yaml_args)

## set visibel gpu
gpu_ids_str = str(args.gpu_ids).replace('[', '').replace(']', '')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)

## select active gpu devices
device = None
if args.gpu_ids is not None and torch.cuda.is_available():
    print('use cuda & cudnn for acceleration!')
    print('the gpu id is: {}'.format(args.gpu_ids))
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    print('use cpu for training!')
    device = torch.device('cpu')
# torch.set_num_threads(args.threads)
args.eval_sets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']
# args.eval_sets = ['Set5']
## create dataset for training and validating
train_dataloader, valid_dataloaders = create_datasets(args)

## definitions of model
try:
    model = utils.import_module('models.{}_network'.format(args.model)).create_model(args)
except Exception:
    raise ValueError('not supported model type! or something')
if args.fp == 16:
    model.half()

## load pretrain
if args.pretrain is not None:
    print('load pretrained model: {}!'.format(args.pretrain))
    ckpt = torch.load(args.pretrain)
    model.load(ckpt['model_state_dict'])

model = nn.DataParallel(model).to(device)

model = model.eval()
torch.set_grad_enabled(False)
save_path = args.log_path
si = save_img()
si.begin_background()
for valid_dataloader in valid_dataloaders:

    avg_psnr, avg_ssim = 0.0, 0.0
    name = valid_dataloader['name']
    loader = valid_dataloader['dataloader']
    for lr, hr, filename in tqdm(loader, ncols=80):
        if args.fp == 16:
            lr, hr = lr.type(torch.HalfTensor), hr.type(torch.HalfTensor)
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)

        # quantize output to [0, 255]
        hr = hr.clamp(0, 255).round()
        sr = sr.clamp(0, 255).round()

        path = save_path + os.sep + name + os.sep
        if not os.path.exists(path):
            os.makedirs(path)
        path += filename[0].replace('.png', '_x' + str(args.scale) + '_SR' + '.png')
        si.save_results(path, sr)
        # tensor_cpu = sr[0].byte().permute(1, 2, 0).cpu()
        # imageio.imwrite(path, tensor_cpu.numpy())

        # conver to ycbcr
        if args.colors == 3:
            hr_ycbcr = utils.rgb_to_ycbcr(hr)
            sr_ycbcr = utils.rgb_to_ycbcr(sr)
            hr = hr_ycbcr[:, 0:1, :, :]
            sr = sr_ycbcr[:, 0:1, :, :]
        # crop image for evaluation
        hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        # calculate psnr and ssim
        psnr = utils.calc_psnr(sr, hr)
        ssim = utils.calc_ssim(sr, hr)
        avg_psnr += psnr
        avg_ssim += ssim
    avg_psnr = round(avg_psnr / len(loader), 2)
    avg_ssim = round(avg_ssim / len(loader), 4)
    test_log = '[{}x{} PSNR/SSIM: {:.2f}/{:.4f}]'.format(name, args.scale, float(avg_psnr), float(avg_ssim))
    print(test_log)
si.end_background()