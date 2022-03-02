import argparse
import re
import os, glob, datetime, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import TrainData
from model import MSFN

# Params
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', default=200, type=int, help='number of train epoches')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--n_feats', default=64, type=int)
parser.add_argument('--n_blocks', default=4, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--save_dir', default='./experiment', type=str)

# Data
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--train_data', default='../../DIV2K', type=str, help='path of train data')
parser.add_argument('--dataset', default='DIV2K', type=str, help='dataset name')
parser.add_argument('--n_colors', default=3, type=int)
parser.add_argument('--sigma', default=50, type=float)
parser.add_argument('--patch_size', default=96, type=int)

# Optim
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--weight_decay', default=1e-8, type=float)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--milestones', type=int, nargs='+', default=[40, 80, 120, 140, 160, 180], help='learning rate decay per N epochs')
parser.add_argument('--clip_grad_norm', default=2.5, type=float)


args = parser.parse_args()

cuda = torch.cuda.is_available()

model_save_dir = os.path.join(args.save_dir, 'models')
optim_save_dir = os.path.join(args.save_dir, 'optim')


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def count_params(model):
    count = 0
    for param in model.parameters():
        param_size = param.size()
        count_of_one_param = 1
        for i in param_size:
            count_of_one_param *= i
        count += count_of_one_param
    print('Total parameters: %d' % count)


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    makedir(model_save_dir)
    makedir(optim_save_dir)

    print('===> Building model')
    model = MSFN(n_colors=args.n_colors, in_channels=args.n_feats, n_blocks=args.n_blocks)
    print(model)
    count_params(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    TrainLoader = TrainData(args).get_loader()

    ###############################################################

    initial_epoch = findLastCheckpoint(save_dir=model_save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(model_save_dir, 'model_%03d.pth' % initial_epoch))
        optimizer = torch.load(os.path.join(optim_save_dir, 'optimizer.pth'))

    model.train()

    criterion = nn.MSELoss(reduction='sum')
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    for epoch in range(initial_epoch, args.epoch):
        time_begin = time.time()
        # scheduler.step(epoch)

        epoch_loss = 0
        start_time = time.time()

        for n_count, (blur, sharp) in enumerate(TrainLoader):
            optimizer.zero_grad()
            if cuda:
                blur, sharp = blur.cuda(), sharp.cuda()
            predict = model(blur)
            loss = criterion(predict, sharp)
            epoch_loss += loss.item()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if (n_count + 1) % 50 == 0:
                log('epcoh = %4d ,iter = %4d ,loss = %4.4f , time = %4.2f s' % (epoch + 1, n_count + 1, loss.item(), time.time() - start_time))
        scheduler.step()
    
        elapsed_time = time.time() - start_time
        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / (n_count + 1), elapsed_time))

            
        torch.save(model, os.path.join(model_save_dir, 'model_%03d.pth' % (epoch + 1)))
        torch.save(optimizer, os.path.join(optim_save_dir, 'optimizer.pth'))
