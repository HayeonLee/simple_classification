  
from __future__ import print_function
from __future__ import division
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as m
# from torch.utils.tensorboard import SummaryWriter

from parser import get_parser
args = get_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
from utils import *
from loader import get_loaders

# Seed
SEED = args.seed
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True


args = get_parser()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
device = torch.device("cuda:0")
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
print(args)


'''Prepare result path'''
if args.debug:
  args.savedir = './results/debug'
elif args.savedir is None:
  args.savedir = './results/trial1'
model_dir = os.path.join(args.savedir, 'model')

if not os.path.exists(args.savedir):
  os.makedirs(model_dir)


net = m.resnet50(pretrained=False)
cuda(net)

trloader, valloader = get_loaders()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                  optimizer, 'min', factor=0.1, patience=10, verbose=True)

# log
# summary = SummaryWriter(args.savedir)
trlog = Log(args, open(args.savedir+f'/train.log', 'w')) #, summary)
vallog = Log(args, open(args.savedir+f'/val.log', 'w')) #, summary)
trlog.print_args()

trlogger = Accumulator('loss', 'acc')
vallogger = Accumulator('loss', 'acc')


def train(epoch):
  net.train()
  total_loss = 0
  for x, y in tqdm(trloader, 'train'):
    x, y = cuda(x), cuda(y)
    optimizer.zero_grad()
    predicts = net(x)
    mp, pred = torch.max(predicts, 1)
    ox = pred.eq(y).cpu()
    loss = criterion(predicts, y)
    loss.backward()
    optimizer.step()
    total_loss += loss.cpu().item()
    trlogger.accum([loss.item(), torch.mean(ox.float()).item()])
  return total_loss


def eval(epoch):
  net.eval()
  total_loss = 0
  with torch.no_grad():
    for x, y in tqdm(valloader, 'eval'):
      x, y = cuda(x), cuda(y)
      predicts = net(x)
      mp, pred = torch.max(predicts, 1)
      ox = pred.eq(y).cpu()

      loss = criterion(predicts, y)
      total_loss += loss.cpu().item()
      vallogger.accum([loss.item(), torch.mean(ox.float()).item()])


def main():
  sttime = time.time()
  for epoch in range(1, args.nepoch + 1):
    loss = train(epoch)
    scheduler.step(loss)
    trlog.print(trlogger, epoch, tag='train')

    eval(epoch)
    vallog.print(vallogger, epoch, tag='eval')

    # if epoch % args.save_epoch == 0:
    #   torch.save(model.state_dict(), model_dir + f'/ckpt_{epoch}.pth')



if __name__ == '__main__':
  main()
