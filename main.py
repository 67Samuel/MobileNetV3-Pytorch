import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import load_data
from model import MobileNetV3

import argparse
from tqdm import tqdm
import time
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_args():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--dataset-mode", type=str, default="IMAGENET", help="(example: CIFAR10, CIFAR100, IMAGENET), (default: IMAGENET)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs, (default: 100)")
    parser.add_argument("--batch-size", type=int, default=512, help="number of batch size, (default, 512)")
    parser.add_argument("--learning-rate", type=float, default=1e-1, help="learning_rate, (default: 1e-1)")
    parser.add_argument("--dropout", type=float, default=0.8, help="dropout rate, not implemented yet, (default: 0.8)")
    parser.add_argument('--model-mode', type=str, default="LARGE", help="(example: LARGE, SMALL), (default: LARGE)")
    parser.add_argument("--load-pretrained", type=bool, default=False, help="(default: False)")
    parser.add_argument('--evaluate', type=bool, default=False, help="Testing time: True, (default: False)")
    parser.add_argument('--multiplier', type=float, default=1.0, help="(default: 1.0)")
    parser.add_argument('--print-interval', type=int, default=5, help="training information and evaluation information output frequency, (default: 5)")
    parser.add_argument('--data', default='D:/ILSVRC/Data/CLS-LOC')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--prune', type=str, default=None)
    parser.add_argument('--snip-percentage', type=int, default=0)

    args = parser.parse_args()

    return args



import torch.nn as nn
import torch.nn.functional as F

import copy
import types

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, keep_ratio, train_dataloader, device, img_size=None, num_channels=3):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    if type(img_size) == int:
        inputs = inputs.view(-1,num_channels,img_size,img_size).float().requires_grad_()
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
    
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)
    
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    print(f"{torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks]))} parameters kept, {torch.sum(torch.cat([torch.flatten(x==0) for x in keep_masks]))} parameters pruned")

    return(keep_masks)

def apply_prune_mask(net, keep_masks):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask
            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask)) #hook masks onto respective weights
  


def initialize_xavier_normal(layer):
	"""
	Function to initialize a layer by picking weights from a xavier normal distribution
	Arguments
	---------
	layer : The layer of the neural network
	Returns
	-------
	None
	"""
	if type(layer) == nn.Conv2d:
		torch.nn.init.xavier_normal_(layer.weight)
		layer.bias.data.fill_(0)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        


# reference,
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Thank you.
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data, target = data.to(device), target.to(device)

        # if args.gpu is not None:
        #     data = data.cuda(args.gpu, non_blocking=True)
        # target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_interval == 0:
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
        for i, (data, target) in enumerate(val_loader):
            # if args.gpu is not None:
            #     input = input.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)
            data, target = data.to(device), target.to(device)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_interval == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
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


def main():
    args = get_args()
    train_loader, test_loader = load_data(args)

    if args.dataset_mode == "CIFAR10":
        num_classes = 10
    elif args.dataset_mode == "CIFAR100":
        num_classes = 100
    elif args.dataset_mode == "IMAGENET":
        num_classes = 1000
    print('num_classes: ', num_classes)

    model = MobileNetV3(model_mode=args.model_mode, num_classes=num_classes, multiplier=args.multiplier, dropout_rate=args.dropout).to(device)
    if torch.cuda.device_count() >= 1:
        print("num GPUs: ", torch.cuda.device_count())
        model = nn.DataParallel(model).to(device)

    if args.load_pretrained or args.evaluate:
        filename = "best_model_" + str(args.model_mode)
        if args.load_pretrained:
            checkpoint = torch.load('./checkpoint/' + filename + '_ckpt.t7')
        else:
            checkpoint = torch.load('./checkpoint/' + filename + '.txt')
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc1 = checkpoint['best_acc1']
        acc5 = checkpoint['best_acc5']
        best_acc1 = acc1
        print("Load Model Accuracy1: ", acc1, " acc5: ", acc5, "Load Model end epoch: ", epoch)
    else:
        print("init model load ...")
        epoch = 1
        best_acc1 = 0
        
    if args.prune:
        print(f"Pruning {args.snip_percentage}% of weights with SNIP...")
        # get snip factor in form required for SNIP function
        snip_factor = (100 - args.snip_percentage)/100
        keep_masks = SNIP(model, snip_factor, train_loader, device)
        apply_prune_mask(model, keep_masks)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-5, momentum=0.9)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.evaluate:
        acc1, acc5 = validate(test_loader, model, criterion, args)
	mask = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                mask.append(torch.abs(layer.weight_mask.grad))
	params_kept = torch.sum(torch.cat([torch.flatten(x == 1) for x in mask]))
	total_params = len(mask)
	print(f"prune percentage: {(total_params-params_kept)*100/total_params}%, {params_kept} parameters kept, {total_params-params_kept} parameters pruned")
        print("Acc1: ", acc1, "Acc5: ", acc5)
        return

    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_time = time.time()
    with open("./reporting/" + "best_model_" + args.model_mode + ".txt", "w") as f:
        for epoch in range(epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args)
            train(train_loader, model, criterion, optimizer, epoch, args)
            acc1, acc5 = validate(test_loader, model, criterion, args)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if is_best:
                print('Saving..')
                best_acc5 = acc5
                state = {
                    'model': model.state_dict(),
                    'best_acc1': best_acc1,
                    'best_acc5': best_acc5,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                filename = "best_model_" + str(args.model_mode)
                torch.save(state, './checkpoint/' + filename + '_ckpt.t7')

            time_interval = time.time() - start_time
            time_split = time.gmtime(time_interval)
            print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ", time_split.tm_sec, end='')
            print(" Test best acc1:", best_acc1, " acc1: ", acc1, " acc5: ", acc5)

            f.write("Epoch: " + str(epoch) + " " + " Best acc: " + str(best_acc1) + " Test acc: " + str(acc1) + "\n")
            f.write("Training time: " + str(time_interval) + " Hour: " + str(time_split.tm_hour) + " Minute: " + str(
                time_split.tm_min) + " Second: " + str(time_split.tm_sec))
            f.write("\n")


if __name__ == "__main__":
    main()
