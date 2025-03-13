# %%
import os
import sys
import argparse
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models_folder.vgg_baseline import VGG,VGG_types
import torch
from optimizer_KLS.dlrt_optimizer import dlr_opt
from optimizer_KLS.train_custom_optimizer import *
from optimizer_KLS.train_experiments import *
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.utils.data as data

device = "cuda" if torch.cuda.is_available() else "cpu"


############################################## parser creation
parser = argparse.ArgumentParser(description='Pytorch dlrt training for vgg of imagenet')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate for dlrt optimizer (default: 0.05)')  
parser.add_argument('--theta', type=float, default=0.0, metavar='THETA',
                    help='threshold for the rank adaption step (default: 0.08)')  
parser.add_argument('--momentum', type=float, default=0.1, metavar='MOMENTUM',
                    help='momentum (default: 0.1)')  
parser.add_argument('--workers', type=int, default=1, metavar='WORKERS',
                    help='number of workers for the dataloaders (default: 1)')      
parser.add_argument('--net_name', type=str, default='vgg_cifar10_baseline_', metavar='NET_NAME',
                    help='network name for the saved results (default: 1)')      
parser.add_argument('--save_weights', type=bool, default=True, metavar='SAVE_WEIGHTS',
                    help='save the weights of the best validation model during the run (default: True)') 
args = parser.parse_args()
############################################## Net creation



num_classes = 10
criterion = torch.nn.CrossEntropyLoss()
NN = VGG(
    in_channels=3, 
    in_height=32, 
    in_width=32, num_classes = num_classes,
    architecture=VGG_types["VGG16"],device = device
).to(device)


optimizer = torch.optim.SGD(NN.parameters(),lr = args.lr,momentum = args.momentum)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
path = './results/vgg/'

########## import 

import pathlib

print(pathlib.Path().resolve())
print("________")

# === data transformation === #
transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataloader = datasets.CIFAR10

trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
validation_loader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


train_baseline(NN = NN,epochs = args.epochs,criterion = criterion,optimizer = optimizer,scheduler = scheduler,
                    train_loader=train_loader,validation_loader=validation_loader,path = path,device = device,
                    net_name = args.net_name,save_weights=args.save_weights)