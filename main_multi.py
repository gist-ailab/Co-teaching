# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from model import CNN
import argparse, sys
import numpy as np
import datetime
import shutil

from loss import loss_multiteaching

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--n_models', type=int, default = 3)

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr 

# load dataset
if args.dataset=='mnist':
    input_channel=1
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = MNIST(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                         )
    
    test_dataset = MNIST(root='./data/',
                               download=True,  
                               train=False, 
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                        )
    
if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = CIFAR10(root='/SSDc/yyg/cifar10/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                           )
    
    test_dataset = CIFAR10(root='/SSDc/yyg/cifar10/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                          )

if args.dataset=='cifar100':
    input_channel=3
    num_classes=100
    args.top_bn = False
    args.epoch_decay_start = 100
    args.n_epoch = 200
    train_dataset = CIFAR100(root='/SSDc/yyg/cifar100/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )
    
    test_dataset = CIFAR100(root='/SSDc/yyg/cifar100/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not
noise_or_not = torch.tensor(noise_or_not).cuda()

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)
   
save_dir = args.result_dir +'/' +args.dataset+'/multiteaching/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_multiteaching_'+args.noise_type+'_'+str(args.noise_rate)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Train the Model
def train(train_loader, epoch, model_list, optim_list):
    print('Training %s...' % model_str)
    pure_ratio_list=[[] for x in range(len(model_list))]
    
    train_total= [0 for x in range(len(model_list))]
    train_correct= [0 for x in range(len(model_list))]

    prec_ = dict()
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.T
        if i>args.num_iter_per_epoch:
            break
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        logits = dict()
        # Forward + Backward + Optimize
        for n in range(args.n_models):
            model = model_list[n]
            logits[n] = model(images)
            prec, _ = accuracy(logits[n], labels, topk=(1, 5))
            prec_[n] = prec

            train_total[n]+= 1
            train_correct[n] += prec

        loss_, pure_ratio_ = loss_multiteaching(logits, labels, rate_schedule[epoch], ind, noise_or_not)

        for n in range(args.n_models):
            pure_ratio_list[n].append(100*pure_ratio_[n])
            optim_list[n].zero_grad()
            loss_[n].backward()
            optim_list[n].step()

        # print(pure_ratio_list)
        # print(loss_1.d)
        if (i+1) % args.print_freq == 0:
            train_acc_str = ' '.join(str(prec_[i].item()) for i in prec_.keys())
            loss_str = ' '.join(str(loss_[i].item()) for i in loss_.keys())
            pure_ratio_str = ' '.join(str((np.sum(pure_ratio)/len(pure_ratio)).item()) for pure_ratio in pure_ratio_list)
            print ('Epoch [{}/{}], Iter [{}/{}]'.format(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size) + 
                    ' ' + train_acc_str + 
                    ' ' + loss_str + 
                    ' ' + pure_ratio_str)

    train_acc_ = dict()
    for n in range(args.n_models):
        train_acc_[n] = float(train_correct[n])/float(train_total[n])
    return train_acc_, pure_ratio_list

# Evaluate the Model
def evaluate(test_loader, model_list):
    print('Evaluating %s...' % model_str)
    accs = []
    for n in range(len(model_list)):
        model = model_list[n]
        model.eval()    # Change model to 'eval' mode.
        correct1 = 0
        total1 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).cuda()
            logits1 = model(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum() 
        acc = 100*float(correct1)/float(total1)
        accs.append(acc)
    return accs


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')
    model_list = []
    optim_list = []
    for n in range(args.n_models):        
        cnn = CNN(input_channel=input_channel, n_outputs=num_classes)
        cnn.cuda()
        # print(cnn.parameters)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
        
        model_list.append(cnn)
        optim_list.append(optimizer)

    mean_pure_ratios = [0 for x in range(args.n_models)]

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: {} / {} / {} \n'.format(' '.join(str('train_acc{}'.format(x)) for x in range(args.n_models)), 
                                                     ' '.join(str('test_acc{}'.format(x)) for x in range(args.n_models)), 
                                                     ' '.join(str('pure_ratio{}'.format(x)) for x in range(args.n_models))))
    epoch=0
    train_accs = [0 for x in range(args.n_models)]
    # evaluate models with random weights
    test_accs =evaluate(test_loader, model_list)
    accuracy_string = str(int(epoch)) + ': ' + ' '.join(str(x) for x in train_accs) + ' / ' + ' '.join(str(x) for x in test_accs) + ' / ' +' '.join(str(x) for x in mean_pure_ratios)
    print(accuracy_string)

    # print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(accuracy_string + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        for n in range(args.n_models):
            model_list[n].train()
            adjust_learning_rate(optim_list[n], epoch)
        train_accs, pure_ratio_list = train(train_loader, epoch, model_list, optim_list)
        # evaluate models
        test_accs = evaluate(test_loader, model_list)
        # save results
        for n in range(args.n_models):
            mean_pure_ratios[n] = sum(pure_ratio_list[n])/len(pure_ratio_list[n])

        train_acc_str = ' '.join(str(train_accs[i]) for i in train_accs.keys())
        test_acc_str = ' '.join(str(acc) for acc in test_accs)
        pure_ratio_str = ' '.join(str((np.sum(pure_ratio)/len(pure_ratio)).item()) for pure_ratio in pure_ratio_list)

        print('Epoch [{}/{}] Test Accuracy on the {} test images: '.format(epoch+1, args.n_epoch, len(test_dataset)) + ' / ' +test_acc_str + ' / ' +pure_ratio_str)
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + train_acc_str +' / '  + test_acc_str + ' / ' + pure_ratio_str + "\n")

if __name__=='__main__':
    main()
