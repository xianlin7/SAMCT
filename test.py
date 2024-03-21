from ast import arg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data import JointTransform2D, ImageToImage2D, ImageToImage2DTest
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from thop import profile


def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='AutoSAMCT', type=str, help='type of model, e.g., SAM, SAMCT, SAMUS, MSA, SAMed, SAMMed2D...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS, SAMCT')  # Modify according to the model
    parser.add_argument('-low_image_size', type=int, default=256, help='the image embedding size, 256 in SAM, SAMCT, and MSA, 128 in SAMed and SAMUS')  # Modify according to the model
    parser.add_argument('--task', default='WORD', help='task or dataset name')  # Modify according to the task
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu') # 8 # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') # True
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=True, help='keep the loss&lr&dice during training or not')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    opt.load_path = "./checkpoints/AutoSAMCT_xxx.pth"
    opt.mode = "val"
    opt.classes=2
    opt.batch_size=1
    opt.test_split = "all_test_WORD_C15" 
    #opt.eval_mode = "camus" #"mask_slice"
    #opt.eval_mode = "mask_slice"
    print(opt.load_path)

    device = torch.device(opt.device)

    #  ============================= add the seed to make sure the results are reproducible ============================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ========================================= model and data initialization ==========================================
    
    # register the sam model
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    val_dataset = ImageToImage2DTest(opt.data_path, opt.test_split, tf_val, img_size=args.encoder_input_size, class_id=1)  # return image, mask, and filename
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.to(device)
    model.train()
    checkpoint = torch.load(opt.load_path)
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    # if args.n_gpu > 1:
    #     #model = nn.DataParallel(model, device_ids = [0,1,2])
    #     model = nn.DataParallel(model)

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    # input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    # points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    # flops, params = profile(model, inputs=(input, points), )
    # print('Gflops:', flops/1000000000, 'params:', params)

    model.eval()
    if opt.mode == "train":
        dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        print("mean dice:", mean_dice)
    else:
        mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        print("dataset:" + args.task + " -----------model name: "+ args.modelname)
        print(mean_dice[1:], mean_hdis[1:], mean_iou[1:], mean_acc[1:], mean_se[1:], mean_sp[1:])
        print(std_dice[1:], std_hdis[1:], std_iou[1:], std_acc[1:], std_se[1:], std_sp[1:])
        with open("experiments.txt", "a+") as file:
            file.write(args.task + " " + opt.test_split + " " + opt.load_path + " " + "\n")
            file.write('%.2f'%(np.mean(mean_dice[1:])*100) + " ")
            file.write('%.2f'%(np.mean(mean_hdis[1:]))  + " ")
            file.write('%.2f'%(np.mean(mean_iou[1:])*100)  + " ")
            file.write('%.2f'%(np.mean(mean_acc[1:])*100)  + " ")
            file.write('%.2f'%(np.mean(mean_se[1:])*100)  + " ")
            file.write('%.2f'%(np.mean(mean_sp[1:])*100)  + "\n")
            file.close()
    

if __name__ == '__main__':
    main()