import argparse
from utils import str2bool, str2intlist, str2strlist


def get_parser():
  parser = argparse.ArgumentParser(add_help=False)
  # basic
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--debug', '--d', action='store_true')
  parser.add_argument('--gpu', type=str, default=0, help='')
  parser.add_argument('--mode', type=str, default='train', help='train')
  parser.add_argument('--savedir', type=str, default=None, help='')
  parser.add_argument('--nepoch', type=int, default=200)
  parser.add_argument('--save_epoch', type=int, default=100)  
  parser.add_argument('--lr', type=float, default=0.01)

  return parser.parse_args()