import argparse
import os
import time
import torch
import numpy as np
import imageio
import util
import pandas as pd
from model import MSFN
from torch.autograd import Variable


# Params
parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--save_dir', default='./experiment', type=str)
parser.add_argument('--test_data', default='../testsets', type=str)
parser.add_argument('--dataset_name', default='Urban100', type=str)
parser.add_argument('--n_colors', default=3, type=int)
parser.add_argument('--sigma', default=70, type=float)
parser.add_argument('--save_results', default=False, type=bool)


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def test_pad(img, model, mod=4):
    n, c, h, w = img.shape
    padding_h = mod - h % mod
    padding_w = mod - w % mod
    if padding_h==mod and padding_w==mod:
        out = model(img)
    else:
        padding_Left = padding_w // 2
        padding_Right = padding_w - padding_Left
        padding_Top = padding_h // 2
        padding_Bottom = padding_h - padding_Top
        padding = torch.nn.ReflectionPad2d([padding_Left, padding_Right, padding_Top, padding_Bottom])
        input = padding(img)
        predict = model(input)
        out = predict[:, :, padding_Top:padding_Top+h, padding_Left:padding_Left+w]
    return out



if __name__ == '__main__':

    args = parser.parse_args()

    # model_path = os.path.join(args.save_dir, 'models')
    model_path = os.path.join(args.save_dir, 'models')
    # data_path = args.test_data
    data_path = os.path.join(args.test_data, args.dataset_name)
    save_path = os.path.join(args.save_dir, 'results', 'DUMRN-B_' + args.dataset_name + '_sigma' + str(int(args.sigma)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cuda = torch.cuda.is_available()

    ############### prepare train data ###############
    data_time = time.time()
    sharp_file_list = []
    for sub, dirs, files in os.walk(data_path):
        if not dirs:
            file_list = [os.path.join(sub, f) for f in files]
            sharp_file_list += file_list

    sharp_file_list.sort()


    print('Finding reading {} test data file path'.format(len(sharp_file_list)))
    print('Reading images to memory..........')

    sharp_image_list = []
    # sharp_file_list = sharp_file_list[:100]
    # blur_file_list = blur_file_list[:100]

    for idx in range(len(sharp_file_list)):
        if args.dataset_name == 'McMaster':
            sharp = imageio.imread(sharp_file_list[idx])
        elif args.n_colors == 3:
            sharp = imageio.imread(sharp_file_list[idx], pilmode='RGB')
        elif args.n_colors == 1:
            sharp = imageio.imread(sharp_file_list[idx], pilmode='L')
            sharp = np.expand_dims(sharp, axis=2)

        sharp_image_list.append(sharp)

    print('prepare test data cost %2.4f s' % (time.time() - data_time))

    ###############################################################


    model = torch.load(os.path.join(model_path, 'RGB_Blind.pth'))
    model.eval()
    if cuda:
        model = model.cuda()

    with torch.no_grad():
        for idx in range(len(sharp_image_list)):
            # print(idx)
            sharp_image = sharp_image_list[idx]
            sharp_image = util.uint2single(sharp_image)
            np.random.seed(seed=0)
            noisy = sharp_image + np.random.normal(0, args.sigma/255., sharp_image.shape).astype('float32')

            noisy_input = util.single2tensor4(noisy)
            if cuda:
                noisy_input = noisy_input.cuda()
            output_img = test_pad(noisy_input, model)
            output_img = util.tensor2single(output_img)

            torch.cuda.synchronize()
            predict = util.single2uint(output_img)

            img_path = sharp_file_list[idx]
            save_name = os.path.split(img_path)[-1]
            imageio.imwrite(os.path.join(save_path, save_name), predict)
