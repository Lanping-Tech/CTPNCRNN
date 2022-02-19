"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: train.py
@time: 2020/4/5 9:24

"""
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch import optim
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import transforms
from dataLoader.dataLoad import IC15Loader,get_bboxes, get_bboxes_crnn, iou_crnn
from models.loss import CTPNLoss
from models.ctpn import CTPN_Model
from utils.rpn_msr.anchor_target_layer import anchor_target_layer
from tools.Log import Logger

import torch.nn.functional as F

from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

from crnn.crnn_train import CRNN_Trainer

from inference import resize_image as resize_image_i
from inference import toTensorImage as toTensorImage_i

random_seed = 2020
torch.random.manual_seed(random_seed)
np.random.seed(random_seed) 

def toTensor(item):
    item = torch.Tensor(item)
    if torch.cuda.is_available():
        item = item.cuda()
    return item

def main(args):
    
    log_write = Logger('./log.txt', 'LogFile')
    log_write.set_names(['Total loss', 'Classified loss','Y location loss','X Refine loss','Learning Rate', 'CRNN Loss'])
    
    data_loader = IC15Loader(args.size_list)
    gt_files = data_loader.gt_paths
    train_loader = torch.utils.data.DataLoader(
                data_loader,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_worker,
                drop_last=True,
                pin_memory=True)

    model = CTPN_Model(base_model=args.base_model,pretrained=args.pretrain).cuda() if torch.cuda.is_available() else CTPN_Model(base_model=args.base_model,pretrained=args.pretrain)
    critetion = CTPNLoss().cuda() if torch.cuda.is_available() else CTPNLoss()
    
    if(args.restore!=''):
        model.load_state_dict(torch.load(args.restore))

    if(args.optimizer=='SGD'):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    model.train()

    crnn_trainer = CRNN_Trainer()

    for epoch in range(args.train_epochs):

        loss_total_list = []
        loss_cls_list = []
        loss_ver_list = []
        loss_refine_list = []
        loss_crnn_list = []

        for batch_idx, (imgs, img_scales,im_shapes, gt_path_indexs,im_infos) in enumerate(train_loader):

            data_loader.get_random_train_size()

            optimizer.zero_grad()

            imgs = imgs.cuda() if torch.cuda.is_available() else imgs

            image = Variable(imgs)

            score_pre, vertical_pred, side_refinement = model(image)

            score_pre = score_pre.permute(0, 2, 3, 1)
            vertical_pred = vertical_pred.permute(0, 2, 3, 1)
            side_refinement = side_refinement.permute(0, 2, 3, 1)

            batch_res_polys = get_bboxes(imgs,gt_files, gt_path_indexs, img_scales,im_shapes)

            batch_loss_tatal = []
            batch_loss_cls = []
            batch_loss_ver = []
            batch_loss_refine = []
            batch_loss_crnn = []
            for i in range(image.shape[0]):

                image_ori =  (imgs[i].numpy()*255).transpose((1,2,0)).copy()

                gt_boxes = np.array(batch_res_polys[i])

                rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(
                    image_ori, score_pre[i].cpu().unsqueeze(0), gt_boxes, im_infos[i].numpy())

                rpn_labels = toTensor(rpn_labels)
                rpn_bbox_targets = toTensor(rpn_bbox_targets)
                rpn_bbox_inside_weights = toTensor(rpn_bbox_inside_weights)
                rpn_bbox_outside_weights = toTensor(rpn_bbox_outside_weights)

                loss_tatal, loss_cls, loss_ver, loss_refine = critetion(score_pre[i].unsqueeze(0), vertical_pred[i].unsqueeze(0), rpn_labels, rpn_bbox_targets)

                batch_loss_tatal.append(loss_tatal)
                batch_loss_cls.append(loss_cls)
                batch_loss_ver.append(loss_ver)
                batch_loss_refine.append(loss_refine)

                del(loss_tatal)
                del(loss_cls)
                del(loss_ver)
                del(loss_refine)

                # ------------------------ crnn ---------------------------

                boxes, text_proposals = ctpn_detect_procedure((imgs[i].numpy()*255).transpose((1,2,0)).copy(), model, args.detect_type)
                if len(boxes) != 0:
                    print('some boxes are detected, crnn training!')
                    boxes = np.array(boxes, dtype=np.int)

                    gt_boxes_crnn, texts = get_bboxes_crnn(gt_files[gt_path_indexs[i]], img_scales[i].numpy()[0],im_shapes[i].numpy()[0])
                    candidate_boxes = []
                    candidate_texts = []
                    for j in range(boxes.shape[0]):
                        c_box = boxes[j, :8].reshape(4, 2)
                        c_x_min = np.min(c_box[:, 0])
                        c_y_min = np.min(c_box[:, 1])
                        c_x_max = np.max(c_box[:, 0])
                        c_y_max = np.max(c_box[:, 1])
                        c_box = np.array([c_x_min, c_y_min, c_x_max, c_y_max])

                        ious = np.array([iou_crnn(c_box, gt_box) for gt_box in gt_boxes_crnn])
                        iou_max_index = np.argmax(ious)
                        if ious[iou_max_index] > 0.5:
                            candidate_boxes.append(c_box)
                            candidate_texts.append(texts[iou_max_index])
                    
                    candidate_boxes = np.array(candidate_boxes)
                    candidate_texts = np.array(candidate_texts)

                    crnn_loss = crnn_trainer.train_batch(image_ori, candidate_boxes, candidate_texts)
                    batch_loss_crnn.append(crnn_loss)


                # ------------------------ crnn ---------------------------


            loss_tatal = sum(batch_loss_tatal)/len(batch_loss_tatal)
            loss_cls = sum(batch_loss_cls)/len(batch_loss_cls)
            loss_ver = sum(batch_loss_ver)/len(batch_loss_ver)
            loss_refine = sum(batch_loss_refine)/len(batch_loss_refine)
            loss_crnn = sum(batch_loss_crnn)/len(batch_loss_crnn) if len(batch_loss_crnn)>0 else 0

            loss_tatal.backward()

            optimizer.step()

            loss_total_list.append(loss_tatal.item())
            loss_cls_list.append(loss_cls.item())
            loss_ver_list.append(loss_ver.item())
            loss_refine_list.append(loss_refine.item())
            loss_crnn_list.append(loss_crnn.item() if len(batch_loss_crnn)>0 else 0)

            if (batch_idx % args.show_step == 0):
                log = '({epoch}/{epochs}/{batch_i}/{all_batch}) | loss_tatal: {loss1:.4f} | loss_cls: {loss2:.4f} | loss_ver: {loss3:.4f} | loss_refine: {loss4:.4f} | Lr: {lr} | loss_crnn: {loss5:.4f}'.format(
                    epoch=epoch, epochs=args.train_epochs, batch_i=batch_idx, all_batch=len(train_loader), loss1=loss_tatal.item(),
                    loss2=loss_cls.item(), loss3=loss_ver.item(), loss4=loss_refine.item(), lr=scheduler.get_lr()[0], loss5=loss_crnn.item() if len(batch_loss_crnn)>0 else 0)
                print(log)
                log_write.append([loss_tatal.item(),loss_cls.item(),loss_ver.item(),loss_refine.item(),scheduler.get_lr()[0], loss_crnn.item() if len(batch_loss_crnn)>0 else 0])

        print('--------------------------------------------------------------------------------------------------------')
        log_write.set_split(['---------','----------','--------','----------','--------','--------'])
        print(
            "epoch_loss_total:{loss1:.4f} | epoch_loss_cls:{loss2:.4f} | epoch_loss_ver:{loss3:.4f} | epoch_loss_ver:{loss4:.4f} | Lr:{lr} | epoch_loss_crnn: {loss5:.4f}".
            format(loss1=np.mean(loss_total_list), loss2=np.mean(loss_cls_list), loss3=np.mean(loss_ver_list),
                   loss4=np.mean(loss_refine_list), lr=scheduler.get_lr()[0], loss5=np.mean(loss_crnn_list)))
        log_write.append([np.mean(loss_total_list),np.mean(loss_cls_list),np.mean(loss_ver_list),np.mean(loss_refine_list),scheduler.get_lr()[0],np.mean(loss_crnn_list)])
        print('-------------------------------------------------------------------------------------------------------')
        log_write.set_split(['---------','----------','--------','----------','--------','--------'])
        if(epoch % args.epoch_save==0 and epoch!=0):
            torch.save(model.state_dict(),os.path.join(args.checkpoint, 'ctpn_' + str(epoch) + '.pth'))
            crnn_trainer.save_model(os.path.join(args.checkpoint, 'crnn_' + str(epoch) + '.pth'))
        scheduler.step()
    log_write.close()

def ctpn_detect_procedure(img, ctpn_model, detect_type):
    img_ori, (rh, rw) = resize_image_i(img)
    h, w, c = img_ori.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    img = toTensorImage_i(img_ori, torch.cuda.is_available())
    with torch.no_grad():
        pre_score, pre_reg, refine_ment = ctpn_model(img)
    score = pre_score.reshape((pre_score.shape[0], 10, 2, pre_score.shape[2], pre_score.shape[3])).squeeze(
        0).permute(0, 2, 3, 1).reshape((-1, 2))
    score = F.softmax(score, dim=1)
    score = score.reshape((10, pre_reg.shape[2], -1, 2))

    pre_score = score.permute(1, 2, 0, 3).reshape(pre_reg.shape[2], pre_reg.shape[3], -1).unsqueeze(
        0).cpu().detach().numpy()
    pre_reg = pre_reg.permute(0, 2, 3, 1).cpu().detach().numpy()
    refine_ment = refine_ment.permute(0, 2, 3, 1).cpu().detach().numpy()

    textsegs, _ = proposal_layer(pre_score, pre_reg, refine_ment, im_info)
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]

    textdetector = TextDetector(DETECT_MODE = detect_type)
    boxes, text_proposals = textdetector.detect(textsegs, scores[:, np.newaxis], img_ori.shape[:2])

    return boxes, text_proposals




if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--base_model', nargs='?', type=str, default='shufflenet_v2_x1_0',help='mobilenet_v3_large,mobilenet_v3_small, shufflenet_v2_x1_0, shufflenet_v2_x0_5, vgg11, vgg11_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, resnet18, resnet34 ,resnet50, resnet101, resnet152')
    parser.add_argument('--optimizer', nargs='?', type=str, default='SGD')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8, help='Batch Size') 
    parser.add_argument('--size_list', nargs='?', type=int, default = [1048], help='img max Size when train') #[768,928,1088,1200,1360]
    parser.add_argument('--num_worker', nargs='?', type=int, default=0, help='num_worker to train')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, help='Learning Rate') 
    parser.add_argument('--step_size', nargs='?', type=int, default=50, help='optimizer step size') 
    parser.add_argument('--gamma', nargs='?', type=float, default=0.1, help='optimizer decay gamma') 
    parser.add_argument('--pretrain', nargs='?', type=bool, default=True, help='If use pre model') 
    parser.add_argument('--restore', nargs='?', type=str, default='', help='If restore to train')
    parser.add_argument('--train_epochs', nargs='?', type=int, default=200, help='how epoch to train')
    parser.add_argument('--show_step', nargs='?', type=int, default=50, help='step to show')
    parser.add_argument('--epoch_save', nargs='?', type=int, default=5, help='how epoch to save')
    parser.add_argument('--checkpoint', default='./model_save', type=str, help='path to save model') 
    parser.add_argument('--detect_type', default='O', type=str, help='detect_type')
    args = parser.parse_args()
    
    main(args)



