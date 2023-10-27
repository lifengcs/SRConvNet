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
import os
from utils import ndarray2tensor
import matplotlib.pyplot as plt

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
parser.add_argument('--custom_image_path', type=str, default=None, help='path of the custom image')

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
if len(args.gpu_ids) > 0 and torch.cuda.is_available():
    print('use cuda & cudnn for acceleration!')
    print('the gpu id is: {}'.format(args.gpu_ids))
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    print('use cpu for training!')
    device = torch.device('cpu')
# torch.set_num_threads(args.threads)

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
    ckpt = torch.load(args.pretrain, map_location=device)
    model.load(ckpt['model_state_dict'])
    

model = nn.DataParallel(model).to(device)
print(model)
model = model.eval()
torch.set_grad_enabled(False)
save_path = args.log_path
si = save_img()
si.begin_background()

filePath = args.custom_image_path
for filename in tqdm(os.listdir(filePath), ncols=80):
    lr = imageio.imread(filePath + os.sep + filename)
    lr = ndarray2tensor(lr)
    lr = torch.unsqueeze(lr, 0)
    if args.fp == 16:
        lr = lr.type(torch.HalfTensor)
    lr = lr.to(device)
    sr = model(lr)

    # quantize output to [0, 255]
    sr = sr.clamp(0, 255).round()
    path = save_path + os.sep + 'custom' + os.sep
    if not os.path.exists(path):
        os.makedirs(path)
    fileUname, ext = '.'.join(filename.split('.')[:-1]), filename.split('.')[-1]
    path += (fileUname + '_x' + str(args.scale) + '_SR' + '.' + ext)
    si.save_results(path, sr)

si.end_background()