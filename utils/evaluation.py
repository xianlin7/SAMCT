# this file is utilized to evaluate the models from different mode: 2D-slice level, 2D-patient level, 3D-patient level
from tkinter import image_names
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
from utils.visualization import visual_segmentation, visual_segmentation_binary
from einops import rearrange
from utils.generate_prompts import get_click_prompt
import time
from collections import Counter

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def eval_mask_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = 2*opt.batch_size * (len(valloader) + 1) + 100
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    #print("len", len(valloader))
    for batch_idx, (datapack) in enumerate(valloader):
        #start_time = time.time()
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))

        image_filename = datapack['image_name']
        test_img_path = os.path.join(opt.data_path + '/img', datapack['image_name'][0])
        from utils.imgname import keep_img_name
        keep_img_name(test_img_path)

        pt = get_click_prompt(datapack, opt)
        bbox  = datapack['bbox']
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=opt.device)
        no_pos = torch.any(bbox[:,0]==-1).item()

        with torch.no_grad():
            #pred = model(imgs, pt)
            if no_pos:
                pred = model(imgs, pt)
            else:
                pred = model(imgs, pt, bbox)
            #sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices[eval_number+j, 1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number+j, 1] += iou
            accs[eval_number+j, 1] += acc
            ses[eval_number+j, 1] += se
            sps[eval_number+j, 1] += sp
            hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
        eval_number = eval_number + b
        #print(time.time()-start_time)
    dices = dices[:eval_number, :] 
    hds = hds[:eval_number, :] 
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    #print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def eval_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
        masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
        label = datapack['label'].to(dtype = torch.float32, device=opt.device)

        pt = get_click_prompt(datapack, opt)
        bbox  = datapack['bbox']
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=opt.device)
        no_pos = torch.any(bbox[:,0]==-1).item()

        with torch.no_grad():
            start_time = time.time()
            if no_pos:
                pred = model(imgs, pt)
            else:
                pred = model(imgs, pt, bbox)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict_masks = pred['masks']
        predict_masks = torch.softmax(predict_masks, dim=1)
        pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dices[eval_number+j, 1] += metrics.dice_coefficient(pred_i, gt_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number+j, 1] += iou
            accs[eval_number+j, 1] += acc
            ses[eval_number+j, 1] += se
            sps[eval_number+j, 1] += sp
            hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
        eval_number = eval_number + b
    dices = dices[:eval_number, :] 
    hds = hds[:eval_number, :] 
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def eval_sammedpatient(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 2000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
        masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
        label = datapack['label'].to(dtype = torch.float32, device=opt.device)
        image_filename = datapack['image_name']

        pt = get_click_prompt(datapack, opt)
        bbox  = datapack['bbox']
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=opt.device)
        no_pos = torch.any(bbox[:,0]==-1).item()

        with torch.no_grad():
            start_time = time.time()
            if no_pos:
                pred = model(imgs, pt)
            else:
                pred = model(imgs, pt, bbox)
            #sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict_masks = pred['masks']
        predict_masks = torch.softmax(predict_masks, dim=1)
        pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            patientid = int(obtain_patien_id(image_filename[j])) # xxxx_2CH_xxx
            flag[patientid] = flag[patientid] + 1
            for i in range(1, opt.classes):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, :, :] == i] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, :, :] == i] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
                hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
                if hd > hds[patientid, i]:
                    hds[patientid, i] = hd
                tps[patientid, i] += tp
                fps[patientid, i] += fp
                tns[patientid, i] += tn
                fns[patientid, i] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)

    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :]
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        # data = pd.DataFrame(patient_dices[:, 1:]*100)
        # writer = pd.ExcelWriter('./result/SAMCT/' + opt.test_split + '.xlsx')
        # data.to_excel(writer, 'page_1', float_format='%.2f')
        # writer._save()

        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def eval_patient(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 1200  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        image_filename = datapack['image_name']

        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            pred = model(imgs, pt)

        val_loss = criterion(pred['masks'], label)
        val_losses += val_loss.item()

        gt = label.detach().cpu().numpy()
        predict = F.softmax(pred['masks'], dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            patient_number = int(image_filename[j][:4]) # xxxx_2CH_xxx
            antrum = int(image_filename[j][5])
            if antrum == 2:
                patientid = patient_number
            else:
                patientid = 600 + patient_number
            flag[patientid] = flag[patientid] + 1
            for i in range(1, opt.classes):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, :, :] == i] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, :, :] == i] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
                hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
                hds[patientid, i] += hd
                tps[patientid, i] += tp
                fps[patientid, i] += fp
                tns[patientid, i] += tn
                fns[patientid, i] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    #flag[194], flag[217], flag[231], flag[273], flag[647], flag[791], flag[840], flag[873] = 0, 0, 0, 0, 0, 0, 0, 0
    
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / flag[flag>0][:, None]
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std



# ----------------------------------------------------------------------------------------------------------------------------
def obtain_patien_id(filename):
    if "-" in filename: # filename = "xx-xx-xx_xxx"
        filename = filename.split('-')[-1]
    # filename = xxxxxxx or filename = xx_xxx
    if "_" in filename:
        patientid = filename.split("_")[0]
    else:
        patientid = filename[:3]
    return patientid

class ComputeMetrics(object):
    def __init__(self, class_num=2):
        super().__init__()
        self.tps, self.fps, self.tns, self.fns, self.hd = {}, {}, {}, {}
    def update(self, patient, tp, fp, tn, fn, hd):
        if patient in self.TPS.keys():
            self.tps[patient] += tp
            self.fps[patient] += fp
            self.tns[patient] += tn
            self.fns[patient] += fn
            if hd < self.hd[patient]:
                self.hd[patient] = hd
        else:
            self.tps[patient] = tp
            self.fps[patient] = fp
            self.tns[patient] = tn
            self.fns[patient] = fn
            self.hd[patient] = hd
    def getdice(self):
        patient_dice = dict((2 * Counter(self.tps) + 1e-5) / (2 * Counter(self.tps) + Counter(self.fps) + Counter(self.fns) + 1e-5))
        values = patient_dice.values()
        avg_dice = sum(values)/len(values)
        return avg_dice
    def gethd(self):
        values = self.hd.values()
        avg_hd = sum(values)/len(values)
        return avg_hd


def eval_mask_patient2(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 2000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        image_filename = datapack['image_name']

        pt = get_click_prompt(datapack, opt)
        bbox  = datapack['bbox']
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=opt.device)
        no_pos = torch.any(bbox[:,0]==-1).item()

        with torch.no_grad():
            #pred = model(imgs, pt)
            if no_pos:
                pred = model(imgs, pt)
            else:
                pred = model(imgs, pt, bbox)
            #sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            patientid = int(obtain_patien_id(image_filename[j])) # xxxx_2CH_xxx
            flag[patientid] = flag[patientid] + 1
            for i in range(1, opt.classes):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, :, :] == i] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, :, :] == i] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
                hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
                if hd > hds[patientid, i]:
                    hds[patientid, i] = hd
                tps[patientid, i] += tp
                fps[patientid, i] += fp
                tns[patientid, i] += tn
                fns[patientid, i] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    #flag[194], flag[217], flag[231], flag[273], flag[647], flag[791], flag[840], flag[873] = 0, 0, 0, 0, 0, 0, 0, 0
    
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :]
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def get_eval(valloader, model, criterion, opt, args):
    if args.modelname == "SAMed":
        if opt.eval_mode == "mask_patient":
            return eval_sammedpatient(valloader, model, criterion, opt, args)
        else:
            if "camus" not in opt.eval_mode or "private" not in opt.eval_mode:
                opt.eval_mode = "slice"   
    if opt.eval_mode == "mask_slice":
        return eval_mask_slice(valloader, model, criterion, opt, args)
    if opt.eval_mode == "mask_patient":
        return eval_mask_patient2(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "slice":
        return eval_slice(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "patient":
        return eval_patient(valloader, model, criterion, opt, args)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)