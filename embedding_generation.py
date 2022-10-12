import sys
import time

sys.path.append('.')
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import os
from funcs import load_dataset
from funcs import test
import argparse
from funcs import get_student_teacher

parse = argparse.ArgumentParser()
parse = argparse.ArgumentParser()
parse.add_argument('--dataset_path', type=str, default='', help="dataset path")
parse.add_argument('--batch_size', type=int, default=128)
parse.add_argument('--epoch', type=int, default=135)
parse.add_argument('--device', type=str, default='cuda')
parse.add_argument('--model_root', type=str, default='')
parse.add_argument('--num_classes', type=int, default=10)
parse.add_argument('--distance', type=str, default=None)
parse.add_argument('--file_dir', type=str, default='./feature/',help="格式为./feature/数据库名/模型的编号或名称/")
parse.add_argument('--dataset',type=str, default='CIFAR10',choices=['CIFAR10','self'])
parse.add_argument('--mode', type=str, default='teacher')
parse.add_argument('--normalize', type=int, default=1)
args = parse.parse_args()
print(args)

def rand_steps(model, X, y, args, target=None):
    # 盲步
    # X 是一个sample
    # y 是X所对应的标签
    # target 感觉没啥意义
    del target
    start = time.time()
    is_training = model.training
    model.eval()

    # 定义噪声
    uni, std, scale = (0.005, 0.005, 0.01)
    steps = 50
    # 设定随机的步数
    noise_2 = lambda X: torch.normal(0, std, size=X.shape).cuda()
    noise_1 = lambda X: torch.from_numpy(np.random.laplace(loc=0.0, scale=scale, size=X.shape)).float().to(args.device)
    noise_inf = lambda X: torch.empty_like(X).uniform_(-uni, uni)
    noise_map = {"l1": noise_1, "l2": noise_2, "linf": noise_inf}
    mag = 1
    # mag 表示步进的程度，mag逐渐加大

    delta = noise_map[args.distance](X)
    delta_base = delta.clone()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)
    loss = 0
    with torch.no_grad():
        for t in range(steps):
            if t > 0:
                preds = model(X_r + delta_r)
                # X_r + delta_r 表示施加噪声后预测仍然为y的点
                new_remaining = (preds.max(1)[1] == y[remaining])
                remaining_temp = remaining.clone()
                remaining[remaining_temp] = new_remaining
                # 更新remaining状态
            else:
                preds = model(X + delta)
                # preds 表示从model中得到此时X+噪声后的预测分布
                remaining = (preds.max(1)[1] == y)
                # remaining 表示预测概率的第一维中最大值的索引是否为y，即预测未发生改变的点的索引

            if remaining.sum() == 0: break
            # 当remaining中所有预测都与y不同，则表示转换完成，结束rand_step

            X_r = X[remaining]
            # X_r 表示X中仍然预测为y的点
            delta_r = delta[remaining]
            # delta[remaining] 表示在X_r处的噪声
            preds = model(X_r + delta_r)
            mag += 1
            delta_r = delta_base[remaining] * mag
            # 加深预测未发生改变处的噪声
            delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1 - X_r)
            # 截取 X+delta_r[remaining] 至 [0, 1]
            delta[remaining] = delta_r.detach()
            # delta与delta_r共享内存
        # print(
        #    f"Number of steps = {t + 1} | Failed to convert = {(model(X + delta).max(1)[1] == y).sum().item()} | Time taken = {time.time() - start}")
        # 输出结果，Failed to convert表示在施加噪声的最大值后也未改变预测的点的个数
    if is_training:
        model.train()
    return delta


def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

def norms_linf_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]

def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:, None, None, None].squeeze(1).squeeze(1).squeeze(1)

def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)


def get_random_label_only(args, loader, model, num_images = 1000):
    print("getting random attacks")
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    # max_iter
    lp_dist = [[],[],[]]
    # lp_dist 表示三种范数计算下的distance
    ex_skipped = 0
    for i, batch in enumerate(loader):
        # if args.regressor_embed == 1: ##We need an extra set of `distinct images for training the confidence regressor
        #    if(ex_skipped < num_images):
        #        y = batch[1]
        #        ex_skipped += y.shape[0]
        #        continue
        # 原论文中还有上面这部分代码，但我不是很理解作用
        for j, distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(10):
                # 反复计算以增加鲁棒性
                X, y = batch[0].to(args.device), batch[1].to(args.device)
                args.distance = distance
                # 此处distance为None
                preds = model(X)
                targets = None
                delta = rand_steps(model, X, y, args, target = targets)
                # delta 为将sample X中一点的预测发生改变时所需要的最小的噪声
                yp = model(X + delta)
                distance_dict = {"linf": norms_linf_squeezed, "l1":norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                # 不是很理解这个地方的数学意义
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            temp_dist = torch.cat(temp_list, dim=1)
            lp_dist[j].append(temp_dist)
        if i+1 >= max_iter:
            break

    lp_d = [torch.cat(lp_dist[i], dim=0).unsqueeze(-1) for i in range(3)]
    full_d = torch.cat(lp_d, dim=-1)
    print(full_d.shape)
    return full_d


def feature_extractor(args):
    print(args)
    train_loader, test_loader = load_dataset(args)
    student, _ = get_student_teacher(args)
    student.train()
    student = student.to(args.device)
    student.load_state_dict(torch.load(args.model_root), strict=False)
    student.eval()
    print(test(student, test_loader, args))

    file_dir = args.file_dir + args.dataset + '/' + args.model_id + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    test_d = get_random_label_only(args, test_loader, student)
    print(test_d)

    torch.save(test_d, f"{args.file_dir}test_rand.pt")

    train_d = get_random_label_only(args, train_loader, student)
    print(train_d)
    torch.save(train_d, f"{args.file_dir}train_rand.pt")


feature_extractor(args)

