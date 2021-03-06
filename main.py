""" importing all the necessary libraries"""

###################################################################

from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import models
import shutil

#####################################################################

"""class get_params to store all the training hyper parameters and other details."""

#####################################################################

class get_params():
  def __init__(self,dataset,sparsity_reg,reg_value,fine_tune,path,train_bs,test_bs,epochs,lr,momentum,weight_decay,intervals,arch,depth,resume):
    self.data=dataset       # dataset on which model is trained
    self.sparsity_reg=sparsity_reg      # true if training is done with sparsity regularization
    self.thr=reg_value      # the sparsity regularization hyperparameter value
    self.fine_tune=fine_tune        # true if pruned model is being fine-tuned
    self.path=path      # path from where the pruned model is loaded
    self.resume=resume      # true of we have to resume training of some model whose checkpoint is saved
    self.train_bs=train_bs      # training batch size
    self.test_bs=test_bs        # test batch size
    self.epochs=epochs
    self.lr=lr
    self.momentum=momentum
    self.weight_decay=weight_decay
    self.log_interval=intervals     # number of intervals after which accuracy and loss values are printed during training
    self.arch=arch      # model architecture
    self.depth=depth        # depth of model (if resnet is being used)


args=get_params('CIFAR10',False,0.00001,True,'/content/drive/MyDrive/VGG-16 NS/CIFAR-10/pruned/resnet_pruned1.pth.tar',64,256,100,1e-1,0.9,1e-4,100,'resnet',164,False)

####################################################################

torch.cuda.manual_seed(42)

"""making train loaders and test loaders from given dataset"""
####################################################################

kwargs = {'num_workers': 2, 'pin_memory': True}
if(args.data=='CIFAR10'):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
    batch_size=args.train_bs, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
                batch_size=args.test_bs, shuffle=True, **kwargs)

########################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""the updated count of number of channels after pruning"""
if(args.fine_tune):
    from prunit import cfg
    print(cfg)
#cfg=[6, 14, 8, 9, 6, 6, 8, 6, 4, 10, 14, 10, 5, 9, 4, 11, 9, 11, 7, 6, 8, 12, 11, 11, 11, 14, 6, 13, 16, 12, 4, 7, 5, 5, 7, 9, 13, 15, 14, 5, 9, 11, 6, 4, 8, 2, 2, 4, 5, 7, 10, 11, 11, 14, 35, 31, 31, 18, 18, 29, 23, 22, 30, 20, 24, 32, 11, 14, 31, 19, 24, 30, 22, 22, 29, 21, 27, 30, 20, 18, 29, 20, 21, 26, 17, 23, 29, 15, 17, 29, 27, 22, 28, 25, 24, 31, 25, 23, 28, 24, 23, 28, 19, 15, 30, 13, 15, 22, 108, 64, 64, 24, 40, 64, 26, 39, 64, 32, 53, 60, 38, 51, 62, 45, 60, 64, 53, 61, 64, 53, 59, 61, 58, 64, 63, 71, 62, 60, 55, 58, 54, 80, 62, 60, 59, 56, 53, 76, 61, 50, 73, 59, 50, 60, 51, 45, 69, 51, 42, 45, 38, 38, 68]


"""loading the models to gpu"""
#########################################################################
if args.fine_tune:
    if (args.arch == 'vgg'):
        model = models.vgg(cfg=cfg)
        model = model.to(device)
    elif(args.arch== 'resnet'):
        model = models.__dict__[args.arch](dataset=args.data, depth=args.depth, cfg=cfg)
        model = model.to(device)
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['state_dict'])
else:
    if (args.arch == 'vgg'):
        model = models.vgg()
        model = model.to(device)
    elif(args.arch=='resnet'):
        model = models.__dict__[args.arch](dataset=args.data, depth=args.depth)
        model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criteria=nn.CrossEntropyLoss()

############################################################################


"""resume training of saved checkpoint"""
############################################################################

if args.resume:
    if os.path.isfile(args.resume):
        model.to(device)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

############################################################################

"""separate function to update the values of scaling parameter"""
def updateBN():
  for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
      m.weight.grad.data.add_(args.thr*torch.sign(m.weight.data))

###########################################################################

"""training function"""
def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data=data.to(device)
    target=target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criteria(output, target)
    loss.backward()
    if args.sparsity_reg:  # here, we are updating the values of scaling parameter
      updateBN()
    optimizer.step()
    if (batch_idx % args.log_interval) == 0:
      print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss))

###############################################################################


"""testing function"""
###############################################################################
def test():
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data=data.to(device)
      target=target.to(device)
      output = model(data)
      test_loss += criteria(output, target).item()
      _,pred = torch.max(output,1)
      correct+=(pred == target).sum().item()

  test_loss /= len(test_loader.dataset)
  print('\nTest set: Average loss: {}, Accuracy: {}/{} ({:.1f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
  return correct / float(len(test_loader.dataset))

################################################################################

"""save the details of current checkpoint and also saves information of best model """

def save_checkpoint(state, is_best, filename='/content/drive/My Drive/VGG-16 NS/CIFAR-10/Fine Tuned/{}_checkpoint.pth.tar'.format(args.arch)):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '/content/drive/My Drive/VGG-16 NS/CIFAR-10/Fine Tuned/{}_model_best.pth.tar'.format(args.arch))

#################################################################################

"""putting it all together """
best_prec1 = 0.
for epoch in range(0, args.epochs):
  print(epoch)
  if epoch in [args.epochs*0.5, args.epochs*0.75]:
    for param_group in optimizer.param_groups:
      param_group['lr'] *= 0.1
  train(epoch)
  prec1 = test()
  is_best = prec1 > best_prec1
  best_prec1 = max(prec1, best_prec1)
  save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best)

##################################################################################