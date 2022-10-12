import torch
import torch.nn as nn
import ipdb
import torch.multiprocessing as _mp
import torch.nn.functional as F
import sys
sys.path.append('./model_src')

from preactresnet import *
from wideresnet import *
from cnn import *
from resnet import *
from resnet_8x import *
import sys
import time
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import os
import torchvision
import torchvision.transforms as transforms

def get_student_teacher(args):
    w_f = 2 if args.dataset == "CIFAR100" else 1
    net_mapper = {"CIFAR10":WideResNet, "CIFAR100":WideResNet, "AFAD":resnet34, "SVHN":ResNet_8x}
    Net_Arch = net_mapper[args.dataset]
    mode = args.mode
    # ['zero-shot', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher']
    deep_full = 34 if args.dataset in ["SVHN", "AFAD"] else 28
    deep_half = 18 if args.dataset in ["SVHN", "AFAD"] else 16
    if mode in ["teacher", "independent", "pre-act-18"]:
        teacher = None
    else:
        deep = 34 if args.dataset in ["SVHN", "AFAD"] else 28
        teacher = Net_Arch(n_classes = args.num_classes, depth=deep, widen_factor=10, normalize = args.normalize, dropRate = 0.3)
        teacher = nn.DataParallel(teacher).to(args.device) if args.dataset != "SVHN" else teacher.to(args.device)
        teacher_dir = "model_teacher_normalized" if args.normalize else "model_teacher_unnormalized"
        path = f"../models/{args.dataset}/{teacher_dir}/final"
        teacher = load(teacher,path)
        teacher.eval()

    if mode == 'zero-shot':
        student = Net_Arch(n_classes = args.num_classes, depth=deep_half, widen_factor=w_f, normalize = args.normalize)
        path = f"../models/{args.dataset}/wrn-16-1/Base/STUDENT3"
        student.load_state_dict(torch.load(f"{path}.pth", map_location = device))
        student = nn.DataParallel(student).to(args.device)
        student.eval()
        raise("Network needs to be un-normalized")
    elif mode == "prune":
        raise("Not handled")

    elif mode == "fine-tune":
        # python train.py --batch_size 1000 --mode fine-tune --lr_max 0.01 --normalize 0 --model_id fine-tune_unnormalized --pseudo_labels 1 --lr_mode 2 --epochs 5 --dataset CIFAR10
        # python train.py --batch_size 1000 --mode fine-tune --lr_max 0.01 --normalize 1 --model_id fine-tune_normalized --pseudo_labels 1 --lr_mode 2 --epochs 5 --dataset CIFAR10
        student =  Net_Arch(n_classes = args.num_classes, depth=deep_full, widen_factor=10, normalize = args.normalize)
        student = nn.DataParallel(student).to(args.device) if args.dataset != "SVHN" else student.to(args.device)
        teacher_dir = "model_teacher_normalized" if args.normalize else "model_teacher_unnormalized"
        path = f"../models/{args.dataset}/{teacher_dir}/final"
        student = load(student,path)
        student.train()
        assert(args.pseudo_labels)

    elif mode in ["extract-label", "extract-logit"]:
        # python train.py --batch_size 1000 --mode extract-label --normalize 0 --model_id extract-label_unnormalized --pseudo_labels 1 --lr_mode 2 --epochs 20 --dataset CIFAR10
        # python train.py --batch_size 1000 --mode extract-label --normalize 1 --model_id extract-label_normalized --pseudo_labels 1 --lr_mode 2 --epochs 20 --dataset CIFAR10
        # python train.py --batch_size 1000 --mode extract-logit --normalize 0 --model_id extract_unnormalized --pseudo_labels 1 --lr_mode 2 --epochs 20 --dataset CIFAR10
        # python train.py --batch_size 1000 --mode extract-logit --normalize 1 --model_id extract_normalized --pseudo_labels 1 --lr_mode 2 --epochs 20 --dataset CIFAR10
        student =  Net_Arch(n_classes = args.num_classes, depth=deep_half, widen_factor=w_f, normalize = args.normalize)
        student = nn.DataParallel(student).to(args.device)
        student.train()
        assert(args.pseudo_labels)

    elif mode in ["distillation", "independent"]:
        dR = 0.3 if mode == "independent" else 0.0
        # python train.py --batch_size 1000 --mode distillation --normalize 0 --model_id distillation_unnormalized --lr_mode 2 --epochs 50 --dataset CIFAR10
        # python train.py --batch_size 1000 --mode distillation --normalize 1 --model_id distillation_normalized --lr_mode 2 --epochs 50 --dataset CIFAR10
        student =  Net_Arch(n_classes = args.num_classes, depth=deep_half, widen_factor=w_f, normalize = args.normalize, dropRate = dR)
        student = nn.DataParallel(student).to(args.device)
        student.train()

    elif mode == "pre-act-18":
        student = PreActResNet18(num_classes = args.num_classes, normalize = args.normalize)
        student = nn.DataParallel(student).to(args.device)
        student.train()

    else:
        # python train.py --batch_size 1000 --mode teacher --normalize 0 --model_id teacher_unnormalized --lr_mode 2 --epochs 100 --dataset CIFAR10 --dropRate 0.3
        # python train.py --batch_size 1000 --mode teacher --normalize 1 --model_id teacher_normalized --lr_mode 2 --epochs 100 --dataset CIFAR10 --dropRate 0.3
        student =  Net_Arch(n_classes = args.num_classes, depth=deep_full, widen_factor=10, normalize = args.normalize, dropRate = 0.3)
        student = nn.DataParallel(student).to(args.device)
        student.train()
        #Alternate student models: [lr_max = 0.01, epochs = 100], [preactresnet], [dropRate]


    return student, teacher

def load_dataset(args):
    if args.dataset == "CIFAR10":
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   #先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),      #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #R,G,B每层的归一化用到的均值和方差
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                                train=True,
                                                transform=transform_train,
                                                download=True)

        test_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                                train=False,
                                                transform=transform_test)
    if args.dataset == "self":
        # 当数据集为私有时，在此处设置transform
        train_dataset = torchvision.datasets.ImageFolder(root=args.dataset_path+'train',
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ]))
        test_dataset = torchvision.datasets.ImageFolder(root=args.dataset_path+'test',
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0)
    return train_loader, test_loader

def test(model, test_loader, args):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images)
            labels = Variable(labels)
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print("Accuracy: {} %".format(accuracy))
    return accuracy
