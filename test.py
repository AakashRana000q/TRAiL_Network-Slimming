from __future__ import print_function
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchsummary import summary
from prunit import cfg
import models

class get_params():
  def __init__(self,dataset,arch,depth,path,path2):
    self.data=dataset       # dataset on which model is trained
    self.arch=arch
    self.depth=depth        # depth of model (if resnet is being used)
    self.tr_path=path          # path from which model is loaded
    self.ft_path=path2
args=get_params('CIFAR10','resnet',164,'','')

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###################################################################

if (args.arch == 'vgg'):
    trained = models.vgg(cfg=cfg)
    trained = trained.to(device)
elif (args.arch == 'resnet'):
    trained = models.__dict__[args.arch](dataset=args.data, depth=args.depth, cfg=cfg)
    trained = trained.to(device)

checkpoint1 = torch.load(args.tr_path)
best_prec1 = checkpoint1['best_prec1']
trained.load_state_dict(checkpoint1['state_dict'])

###################################################################

#cfg=[37, 63, 'M', 128, 128, 'M', 251, 251, 221, 180, 'M', 134, 44, 37, 32, 'M', 18, 20, 30, 77]

###################################################################

checkpoint = torch.load(args.ft_path)
if (args.arch == 'vgg'):
    pruned = models.vgg(cfg=cfg)
    prened = pruned.to(device)
elif (args.arch == 'resnet'):
    pruned = models.__dict__[args.arch](dataset=args.data, depth=args.depth, cfg=cfg)
    pruned = pruned.to(device)

pruned.load_state_dict(checkpoint['state_dict'])
best_prec1 = checkpoint['best_prec1']

###################################################################

"""loading datasets"""
###################################################################
if(args.data=='CIFAR10'):
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False,download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=64, shuffle=True, num_workers=2,pin_memory=True)

    criteria=nn.CrossEntropyLoss()

###################################################################

"""test function"""

###################################################################
def test(model):
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

###################################################################

summary(trained, (3, 32, 32))  # accuarcy and details of trained model without fine tuning
test(trained)

summary(pruned, (3, 32, 32))  # accuarcy and details of fine tuned model
test(pruned)