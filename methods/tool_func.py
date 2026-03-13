
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
EPS=0.00001
P_THRED = 0.2
START_EPS = 16/255
P_THRED_ATTACK = 0
DECAY = 0.1

def calc_mean_std(feat, eps=EPS):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def fgsm_attack(init_input, epsilon, ori_grad, crop_grad):
    # random start init_input
    input_size = init_input.size()
    init_input = init_input + START_EPS * torch.randn(input_size).cuda()

    num_crops = len(crop_grad)
    crop_grad = sum(crop_grad) / num_crops
    ori_grad_norm = ori_grad / torch.mean(torch.abs(ori_grad), dim=(1, 2, 3), keepdim=True)
    crop_grad_norm = crop_grad / torch.mean(torch.abs(crop_grad), dim=(1, 2, 3), keepdim=True)

    data_grad = ori_grad_norm - DECAY * crop_grad_norm
    sign_data_grad = data_grad.sign()
    adv_input = init_input + epsilon*sign_data_grad
    return adv_input

def changeNewAdvStyle(input_fea, new_styleAug_mean, new_styleAug_std, p_thred):
    if(new_styleAug_mean=='None'):
        return input_fea
    
    p = np.random.uniform()
    if( p < p_thred):
        return input_fea

    feat_size = input_fea.size()
    ori_style_mean, ori_style_std = calc_mean_std(input_fea)
    normalized_fea = (input_fea - ori_style_mean.expand(feat_size)) / ori_style_std.expand(feat_size)
    styleAug_fea  = normalized_fea * new_styleAug_std.expand(feat_size) + new_styleAug_mean.expand(feat_size)
    return styleAug_fea

def consistency_loss(scoresM1, scoresM2, type='euclidean'):
    if(type=='euclidean'):
        avg_pro = (scoresM1 + scoresM2)/2.0
        matrix1 = torch.sqrt(torch.sum((scoresM1 - avg_pro)**2,dim=1))
        matrix2 = torch.sqrt(torch.sum((scoresM2 - avg_pro)**2,dim=1))
        dis1 = torch.mean(matrix1)
        dis2 = torch.mean(matrix2)
        dis = (dis1+dis2)/2.0
    elif(type=='KL1'):
        avg_pro = (scoresM1 + scoresM2)/2.0
        matrix1 = torch.sum( F.softmax(scoresM1,dim=-1) * (F.log_softmax(scoresM1, dim=-1) - F.log_softmax(avg_pro,dim=-1)), 1)
        matrix2 = torch.sum( F.softmax(scoresM2,dim=-1) * (F.log_softmax(scoresM2, dim=-1) - F.log_softmax(avg_pro,dim=-1)), 1)
        dis1 = torch.mean(matrix1)
        dis2 = torch.mean(matrix2)
        dis = (dis1+dis2)/2.0
    elif(type=='KL2'):
        matrix = torch.sum( F.softmax(scoresM2,dim=-1) * (F.log_softmax(scoresM2, dim=-1) - F.log_softmax(scoresM1,dim=-1)), 1)
        dis = torch.mean(matrix)
    elif(type=='KL3'):
        matrix = torch.sum( F.softmax(scoresM1,dim=-1) * (F.log_softmax(scoresM1, dim=-1) - F.log_softmax(scoresM2,dim=-1)), 1)
        dis = torch.mean(matrix)
    else:
        return
    return dis

def Entropy_residual(softmax_residual):
    epsilon = 1e-5
    entropy_residual = -softmax_residual * torch.log(softmax_residual + epsilon)
    entropy_res = torch.sum(entropy_residual, dim=1)
    return entropy_res

def residual_style_loss(x_ori, x_adv, classifier_b, epoch):
    feature_size = x_ori.size()
    # ori_style_mean, ori_style_std = calc_mean_std(x_ori)
    # adv_style_mean, adv_style_std = calc_mean_std(x_adv)
    # ori_style_mean = torch.nn.Parameter(ori_style_mean)
    # ori_style_std = torch.nn.Parameter(ori_style_std)
    # adv_style_mean = torch.nn.Parameter(adv_style_mean)
    # adv_style_std = torch.nn.Parameter(adv_style_std)

    # ori_styles = torch.cat((ori_style_mean.expand(feature_size), ori_style_std.expand(feature_size)), dim=1)
    # adv_styles = torch.cat((adv_style_mean.expand(feature_size), adv_style_std.expand(feature_size)), dim=1)

    lambd_init = torch.tensor([0.7], dtype=torch.float).cuda()
    alpha = torch.tensor([10], dtype=torch.float).cuda()
    belta = torch.tensor([0.75], dtype=torch.float).cuda()
    lambd = lambd_init * (1 - (1 + alpha * (epoch + 1) / 200)**(-belta))

    epsilon = 1e-3
    residual_com = x_adv - x_ori * lambd
    denomi = 1 - lambd

    if denomi < epsilon:
        denomi += epsilon
    if 1 - denomi < epsilon:
        denomi -= epsilon

    residual_component = residual_com / denomi

    residual_component_mean, residual_component_std = calc_mean_std(residual_component)
    residual_styles = torch.cat((residual_component_mean.expand(feature_size), residual_component_std.expand(feature_size)), dim=0)

    output_residual = classifier_b(residual_styles)
    softmax_residual = nn.Softmax(dim=1)(output_residual)
    entropy_residual = torch.mean(Entropy_residual(softmax_residual))
    return entropy_residual