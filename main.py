"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import argparse
import os
import random
import time
import warnings
import PIL

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from efficientnet_pytorch import EfficientNet, utils
from utils import save_checkpoint, AverageMeter, ProgressMeter, adjust_learning_rate, accuracy

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (use cifar10, cifar100 to refer to built in datasets)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image-size', default=224, type=int,
                    help='image size')
parser.add_argument('--num-classes', default=1000, type=int)
parser.add_argument('--request-from-nni', default=False, action='store_true')
parser.add_argument('--model-dir', default='/tmp', type=str)
parser.add_argument('--depth-coefficient', default=None, type=float)
parser.add_argument('--width-coefficient', default=None, type=float)
parser.add_argument('--resolution', default=None, type=int)

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.request_from_nni:
        import nni
        tuner_params = nni.get_next_parameter()
        print(tuner_params)

        if "alpha" in tuner_params:
            args.depth_coefficient = tuner_params["alpha"]
            args.width_coefficient = tuner_params["beta"]
            args.resolution = int(tuner_params["gamma"] * 224)
        if "lr" in tuner_params:
            args.learning_rate = tuner_params["lr"]

        # demonstrate that intermediate result is actually sent
        nni.report_intermediate_result(0.)

        args.model_dir = os.environ["NNI_OUTPUT_DIR"]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # Simply call main_worker function
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if 'efficientnet' in args.arch:  # NEW
        if args.request_from_nni:
            block_args, global_params = utils.efficientnet(width_coefficient=args.width_coefficient,
                                                           depth_coefficient=args.depth_coefficient,
                                                           image_size=args.resolution,
                                                           num_classes=args.num_classes)
            model = EfficientNet(block_args, global_params)
            print("=> Creating EfficientNet with configurations from NNI")
        elif args.pretrained:
            model = EfficientNet.from_pretrained(args.arch, num_classes=args.num_classes)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(args.arch)
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Create transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if 'efficientnet' in args.arch:
        if args.request_from_nni:
            image_size = args.resolution
        else:
            image_size = EfficientNet.get_image_size(args.arch)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        print('Using image size', image_size)
    else:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        print('Using image size', 224)

    # Data loading code
    if args.data == "cifar10":
        cifar10_dir = "/tmp/cifar10"
        train_dataset = datasets.CIFAR10(cifar10_dir, train=True, transform=train_transforms, download=True)
        val_dataset = datasets.CIFAR10(cifar10_dir, train=False, transform=val_transforms, download=True)
    elif args.data == "cifar100":
        cifar100_dir = "/tmp/cifar100"
        train_dataset = datasets.CIFAR100(cifar100_dir, train=True, transform=train_transforms, download=True)
        val_dataset = datasets.CIFAR100(cifar100_dir, train=False, transform=val_transforms, download=True)
    else:
        print("Dealing with ImageNet here at %s" % os.path.abspath(args.data))
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        train_dataset = datasets.ImageFolder(traindir, train_transforms)
        val_dataset = datasets.ImageFolder(valdir, val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        res = validate(val_loader, model, criterion, args)
        with open('res.txt', 'w') as f:
            print(res, file=f)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        if args.request_from_nni:
            import nni
            nni.report_intermediate_result(acc1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.model_dir, "checkpoint.pth.tar"))

    try:
        if args.request_from_nni:
            import nni
            nni.report_final_result(acc1)
    except NameError:
        print("No accuracy reported")
        pass


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    main()
