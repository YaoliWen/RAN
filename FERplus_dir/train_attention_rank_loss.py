import argparse
import os,sys,shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import math
from attention_rank_loss import resnet18,resnet34
from new_model import resnet18 as new_resnet18
import attention_rank_loss
#from part_attentioon_sample_fly import MsCelebDataset, CaffeCrop
from part_attention_sample import MsCelebDataset, CaffeCrop
import scipy.io as sio  
import numpy as np
import pdb
from torch.utils.tensorboard import SummaryWriter

summary_dir = "/157Dataset/data-wen.yaoli/RAN/baseline"

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='/media/sdb/xxzeng/MS-1m-subset-cleanning-by-demeng-size-178_218/MS-1m-subset-cleanning-by-demeng-size-178_218/', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
# default 16
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# default = 60
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# default = 128
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-b_t', '--batch-size_t', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='../../Data/Model/resnet34_ferplus.pkl', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_dir','-m', default='./0_1rank_loss_model', type=str)
parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print('img_dir:', args.img_dir)
    print('end2end?:', args.end2end)
    args_img_dir='/home/phd-liu.fang/wyl/data/Alignment' #对齐的面部地址



    # load data and prepare dataset
    train_list_file = '/home/phd-liu.fang/wyl/RAN/FERplus_dir/dlib_ferplus_train_center_crop_range_list.txt' #训练集目录txt
    train_label_file = '/home/phd-liu.fang/wyl/RAN/FERplus_dir/dlib_ferplus_train_center_crop_range_label.txt' #训练集标签txt
    caffe_crop = CaffeCrop('train') # resize到224的大小
    train_dataset =  MsCelebDataset(args_img_dir, train_list_file, train_label_file, 
            transforms.Compose([caffe_crop, transforms.ToTensor()])) # 用PIL.Image读取为“RGB”图像并转换为224大小的ToTensor

    
    args_img_dir_val='/home/phd-liu.fang/wyl/data/Alignment' # 测试集根目录
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True) # 训练集dataloader
   
    caffe_crop = CaffeCrop('test')
    val_list_file = '/home/phd-liu.fang/wyl/RAN/FERplus_dir/dlib_ferplus_val_center_crop_range_list.txt'
    val_label_file = '/home/phd-liu.fang/wyl/RAN/FERplus_dir/dlib_ferplus_val_center_crop_range_label.txt'
    val_dataset =  MsCelebDataset(args_img_dir_val, val_list_file, val_label_file, 
            transforms.Compose([caffe_crop,transforms.ToTensor()])) # 与训练集相同
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_t, shuffle=False,
        num_workers=args.workers, pin_memory=True) # 测试集dataloader

    

    # prepare model
    model = None
    assert(args.arch in ['resnet18','resnet34','resnet101'])
    if args.arch == 'resnet18':
        model = new_resnet18(end2end=args.end2end)
    if args.arch == 'resnet34':
        model = resnet34(end2end=args.end2end)
    if args.arch == 'resnet101':
        pass
    
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    criterion1 = attention_rank_loss.MyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay) # v(t+1) = m*v(t) + g(t+1)  p(t+1) = p(t) - lr*v(t+1)   weight decay (L2 penalty)




   # optionally resume from a checkpoint
    
    if args.pretrained:
        checkpoint = torch.load('/home/lab-wen.yaoli/CODE/git/Region-Attention-Network/checkpoint/ijba_res18_naive.pth.tar')

        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        # pdb.set_trace()
        
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass # 去掉全连接层
            else:    
                model_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(model_state_dict, strict = False)

    if args.resume: #加载原本训练的模型
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #pdb.set_trace()
    print ('args.evaluate',args.evaluate)
    if args.evaluate: 
        #validate(val_loader, model, criterion)
        print('validate')
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch) # epoch的(0.3、0.5、0.8)时lr *= 0.1

        # train for one epoch
        train(train_loader, model, criterion, criterion1, optimizer, epoch, args.batch_size_t)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, criterion1, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        # print(prec1.cuda().item())            
        # best_prec1 = max(prec1.cuda()[0], best_prec1)
        best_prec1 = max(prec1.cuda().item(), best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best.item())

def train(train_loader, model, criterion, criterion1, optimizer, epoch, base_batch_size):
    batch_time = AverageMeter() #统计类
    data_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    weights_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_loss = 0
    batch_size = 0
    total_num = 0
    loss_freq = 128//base_batch_size
    # switch to train mode
    model.train()


    end = time.time()
    for i, (input_first, target_first, input_second, target_second, input_third, target_third, input_forth, target_forth, input_fifth, target_fifth, input_sixth, target_sixth) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        total_num += input_first.size(0)
        input = torch.zeros([input_first.shape[0],input_first.shape[1],input_first.shape[2],input_first.shape[3],6])


        input[:,:,:,:,0] = input_first
        input[:,:,:,:,1] = input_second
        input[:,:,:,:,2] = input_third
        input[:,:,:,:,3] = input_forth
        input[:,:,:,:,4] = input_fifth
        input[:,:,:,:,5] = input_sixth
        #放入6种图片，0为原图
        
        target = target_first
         

        target = target.cuda()
 
        
        input_var = torch.autograd.Variable(input) # [B, 3, 224, 224, 6]
        target_var = torch.autograd.Variable(target) # 放入变量图中（意义？）
        
        
        # compute output
        pred_score, alphas_part_max, alphas_org = model(input_var)
        # pdb.set_trace()
        
        loss = criterion(pred_score, target_var) + criterion1(alphas_part_max, alphas_org) # 计算出crossentropy和margin loss的平均值
        weights_loss = criterion1(alphas_part_max, alphas_org) # 计算margin loss 的平均值
        # pdb.set_trace()
        
        # measure accuracy and record loss
        prec1 = accuracy(pred_score.data, target, topk=(1,)) # scalar tensor 该batch正确的百分比
        # pdb.set_trace()
        losses.update(loss.item(), input.size(0)) # 更新losses为总的平均值
        try:
            weights_losses.update(weights_loss.item(), input.size(0)) # 更新weight_losses为总的平均值
        except:
            pdb.set_trace()
        top1.update(prec1[0], input.size(0)) #更新准确率为总的平均值
        

        # compute gradient and do SGD step
        #set epoch
        batch_loss += loss*input.size(0)
        batch_size += input.size(0)

        if ((i+1) % loss_freq == 0) or (input_first.size(0)<base_batch_size):
            batch_loss = batch_loss/batch_size
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_loss = 0
            batch_size = 0
            
        # measure elapsed ti

        # measure elapsed time
        batch_time.update(time.time() - end) # 计算总的一个batch平均时间
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val} ({batch_time.avg})\t'
                  'Data {data_time.val} ({data_time.avg})\t'
                  'Loss {loss.val} ({loss.avg})\t'
                  'weights_loss {weights_loss.val} ({weights_loss.avg})\t'
                  'Prec@1 {top1.val} ({top1.avg})\t'
                                                              .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, weights_loss=weights_losses, top1=top1))
            with SummaryWriter(summary_dir) as writer: 
                writer.add_scalar('Trian/weight_Loss_val', weights_losses.val, epoch+(i/len(train_loader))) 
                writer.add_scalar('Trian/acc_val', top1.val, epoch+(i/len(train_loader)))
                writer.add_scalar('Trian/Loss_val', losses.val, epoch+(i/len(train_loader)))
    with SummaryWriter(summary_dir) as writer: 
        writer.add_scalar('Trian/Loss_avg', losses.avg, epoch)
        writer.add_scalar('Trian/weight_Loss_avg', weights_losses.avg, epoch)
        writer.add_scalar('Trian/acc_avg', top1.avg, epoch)
            
def validate(val_loader, model, criterion, criterion1, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    weights_losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_first, target_first, input_second,target_second, input_third, target_third, input_forth, target_forth, input_fifth, target_fifth, input_sixth, target_sixth) in enumerate(val_loader):
        # target = target.cuda(async=True)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        input = torch.zeros([input_first.shape[0],input_first.shape[1],input_first.shape[2],input_first.shape[3],6])
        #input = torch.cat((input_first,input_second),0)
        #input = torch.cat((input,input_third),0)


        input[:,:,:,:,0] = input_first
        input[:,:,:,:,1] = input_second
        input[:,:,:,:,2] = input_third
        input[:,:,:,:,3] = input_forth
        input[:,:,:,:,4] = input_fifth
        input[:,:,:,:,5] = input_sixth
        #input[:,:,:,:,6] = input_seventh
        #input[:,:,:,:,7] = input_eigth
        
        target = target_first
         

        target = target.cuda()
 
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        pred_score, alphas_part_max, alphas_org = model(input_var)
        loss = criterion(pred_score, target_var) + criterion1(alphas_part_max, alphas_org)
        # try:
        weights_loss = criterion1(alphas_part_max, alphas_org)

        # measure accuracy and record loss
        prec1 = accuracy(pred_score.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        weights_losses.update(weights_loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val} ({batch_time.avg})\t'
                  'Loss {loss.val} ({loss.avg})\t'
                  'weights_loss {weights_loss.val} ({weights_loss.avg})\t'
                  'Prec@1 {top1.val} ({top1.avg})\t'
                  .format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, weights_loss=weights_losses,  
                   top1=top1))
            with SummaryWriter(summary_dir) as writer:
                writer.add_scalar('Test/Loss_val', losses.val, epoch+(i/len(val_loader)))
                writer.add_scalar('Test/weight_Loss_val', weights_losses.val, epoch+(i/len(val_loader)))
                writer.add_scalar('Test/acc_val', top1.val, epoch+(i/len(val_loader)))

    print(' * Prec@1 {top1.avg} '
          .format(top1=top1))
    
    with SummaryWriter(summary_dir) as writer: 
        writer.add_scalar('Test/Loss_avg', losses.avg, epoch)
        writer.add_scalar('Test/weight_Loss_avg', weights_losses.avg, epoch)
        writer.add_scalar('Test/acc_avg', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    full_filename = os.path.join(args.model_dir, filename)
    full_bestname = os.path.join(args.model_dir, 'model_best.pth.tar')
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save(state, full_filename)
    epoch_num = state['epoch']
    if epoch_num%1==0 and epoch_num>=0:
        torch.save(state, full_filename.replace('checkpoint','checkpoint_'+str(epoch_num)))
    if is_best:
        shutil.copyfile(full_filename, full_bestname)


class AverageMeter(object): 
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in [int(args.epochs*0.3), int(args.epochs*0.5), int(args.epochs*0.8)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output(B*8), target(B*1)
    maxk = max(topk) # 1
    batch_size = target.size(0) # B

    _, pred = output.topk(maxk, 1, True, True) # maxk个， 第1维度（列）上，最大值，按降序排列的，索引 （B*1）
    pred = pred.t() #1*B
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # 1*B bool

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0) #求总正确数目 tensor scalar
        res.append(correct_k.mul_(100.0 / batch_size)) #求百分比 scalar tensor
        # print("res {}".format(res))
    return res


if __name__ == '__main__':
    main()
