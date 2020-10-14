import os 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms


def cifar10_preprocessing(mode='train'):
	dpath = '/st1/dataset/CIFAR-10-images'
	import glob
	dlst = {}
	clst = glob.glob(dpath+'/'+mode+'/*')
	for cpath in clst:
		foldername = cpath.split('/')[-1]
		dlst[foldername] = []

		for filename in glob.glob(cpath+'/*'):
			im = Image.open(filename)
			tensor_im = transforms.ToTensor()(im)
			dlst[foldername].append(tensor_im)
	torch.save(dlst, f'cifar10_{mode}.pth')

# cifar10_preprocessing('train')
# cifar10_preprocessing('test')

def get_transforms():
    # Data transformation with augmentation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    T = []
    # T.append(transforms.Resize(args.imsz + 4))
    # T.append(transforms.CenterCrop(args.imsz))
    # T.append(transforms.RandomResizedCrop(args.imsz))
    # T.append(transforms.RandomHorizontalFlip())
    # T.append(transforms.ColorJitter(
    #   brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
    T.append(transforms.ToTensor())
    T.append(transforms.Normalize(mean, std))
    data_transforms = {
      'train': transforms.Compose(T),
      'val': transforms.Compose([
        # transforms.Resize(args.imsz + 4),
        # transforms.CenterCrop(args.imsz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ]),
      'test': transforms.Compose([
        # transforms.Resize(args.imsz + 4),
        # transforms.CenterCrop(args.imsz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ])
    }
    return data_transforms


class Dataset(Dataset):
	def __init__(self, dataset='cifar10', mode='train', T=None):
		x = torch.load(f'cifar10_{mode}.pth')
		self.x = []
		self.y = []
		for i, (k, v) in enumerate(x.items()):
			self.x += v
			self.y += [i] * len(v)
		self.T = T[mode]
		self.topil = transforms.ToPILImage()

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index):
		x = self.x[index]
		if self.T is not None:
			x = self.T(self.topil(x))
		y = self.y[index]
		return x, y


def get_loaders():
	dataset = Dataset(mode='train', T=get_transforms())
	trloader = DataLoader(dataset=dataset,
												batch_size=64,
												shuffle=True,
												num_workers=4)
	dataset = Dataset(mode='test', T=get_transforms())
	valloader = DataLoader(dataset=dataset,
												batch_size=64,
												shuffle=False,
												num_workers=4)
	return trloader, valloader

# trloader, valloader = get_loaders()

# for x, y in trloader:
# 	import pdb; pdb.set_trace()


		
