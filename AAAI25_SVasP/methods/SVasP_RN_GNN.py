import torch
import torch.nn as nn
import numpy as np
import random

from methods.gnn import GNN_nl
from methods import backbone_multiblock
from methods.tool_func import *
from methods.meta_template_SVasP_RN_GNN import MetaTemplate

class SVasPGNN(MetaTemplate):
  maml=False
  def __init__(self, model_func, params, n_way, n_support, tf_path=None):
    super(SVasPGNN, self).__init__(model_func, n_way, n_support, tf_path=tf_path)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(backbone_multiblock.Linear_fw(self.feat_dim, 128), backbone_multiblock.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)

    # for global classifier
    self.method = params.method
    self.classifier = nn.Linear(self.feature.final_feat_dim, 64)

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way).cuda()
    
    # init
    self.domain_discriminators = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(self.feature.final_feat_dim, 2))
    # new
    # self.domain_discriminators = nn.Sequential(nn.Linear(self.feature.final_feat_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))

    self.lambd_crop = params.lambd_crop

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.classifier.cuda()
    self.support_label = self.support_label.cuda()
    self.domain_discriminators = self.domain_discriminators.cuda()
    return self

  def set_forward(self,x,is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))
      z = z.view(self.n_way, -1, z.size(1))

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn(z_stack)
    return scores

  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss

  def adversarial_attack_Incre(self, x, y_global_only, epsilon_list):
    if isinstance(x, list):
      total_num = len(x)
      num_crops = len(x) - 1
      x_crops = x[:num_crops]
      x_ori = x[-1]

    else:
      x_ori = x
      x_adv = x

    x_crops =[x_crop.cuda() for x_crop in x_crops]

    x_ori = x_ori.cuda()
    y_global_only = y_global_only.cuda()

    x_size = x_ori.size()
    Episode_batch = x_size[0] * x_size[1]

    x_crops = [x_crop.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4]) for x_crop in x_crops]

    x_ori = x_ori.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    x_total_crops = torch.cat(x_crops, dim=0)
    x_total = torch.cat((x_total_crops, x_ori), dim=0)
    y_global_only = y_global_only.view(x_size[0]*x_size[1])
    y_global = [y_global_only for i in range(total_num)]
    y_global = torch.cat(y_global, dim=0)

    # if not adv, set defalut = 'None'
    adv_style_mean_block1, adv_style_std_block1 = 'None', 'None'
    adv_style_mean_block2, adv_style_std_block2 = 'None', 'None'
    adv_style_mean_block3, adv_style_std_block3 = 'None', 'None'

    # forward and set the grad = True
    blocklist = 'block123'
    
    if('1' in blocklist and epsilon_list[0] != 0):

      # forward block1
      x_total_block1 = self.feature.forward_block1(x_total)
      feat_size_block1 = x_total_block1[num_crops * Episode_batch:].size()

      x_crop_block1 = [x_total_block1[i * Episode_batch: (i + 1) * Episode_batch] for i in range(num_crops)]

      x_ori_block1 = x_total_block1[num_crops * Episode_batch:]

      crop_style_mean_block1 = []
      crop_style_std_block1 = []
      for i in range(num_crops):
        crop_style_mean_block1_i, crop_style_std_block1_i = calc_mean_std(x_crop_block1[i])
        crop_style_mean_block1.append(crop_style_mean_block1_i)
        crop_style_std_block1.append(crop_style_std_block1_i)
  
      ori_style_mean_block1, ori_style_std_block1 = calc_mean_std(x_ori_block1)

      # set them as learnable parameters
      crop_style_mean_block1 = [torch.nn.Parameter(crop_style_mean_block1_i) for crop_style_mean_block1_i in crop_style_mean_block1]
      crop_style_std_block1 = [torch.nn.Parameter(crop_style_std_block1_i) for crop_style_std_block1_i in crop_style_std_block1]

      ori_style_mean_block1  = torch.nn.Parameter(ori_style_mean_block1)
      ori_style_std_block1 = torch.nn.Parameter(ori_style_std_block1)

      for i in range(num_crops):
        crop_style_mean_block1[i].requires_grad_()
        crop_style_std_block1[i].requires_grad_()

      ori_style_mean_block1.requires_grad_()
      ori_style_std_block1.requires_grad_()
      # contain ori_style_mean_block1 in the graph
      x_crop_normalized_block1 = [(x_crop_block1[i] - crop_style_mean_block1[i].detach().expand(feat_size_block1)) / crop_style_std_block1[i].detach().expand(feat_size_block1) for i in range(num_crops)]
      x_crop_block1 = [x_crop_normalized_block1[i] * crop_style_std_block1[i].expand(feat_size_block1) + crop_style_mean_block1[i].expand(feat_size_block1) for i in range(num_crops)]

      x_normalized_block1 = (x_ori_block1 - ori_style_mean_block1.detach().expand(feat_size_block1)) / ori_style_std_block1.detach().expand(feat_size_block1)
      x_ori_block1 = x_normalized_block1 * ori_style_std_block1.expand(feat_size_block1) + ori_style_mean_block1.expand(feat_size_block1)
      
      # pass the rest model
      x_crop_block1 = torch.cat(x_crop_block1, dim=0)
      x_total_block1 = torch.cat((x_crop_block1, x_ori_block1), dim=0)
      x_total_block2 = self.feature.forward_block2(x_total_block1)
      x_total_block3 = self.feature.forward_block3(x_total_block2)
      x_total_block4 = self.feature.forward_block4(x_total_block3)
      x_total_fea = self.feature.forward_rest(x_total_block4)
      x_total_output = self.classifier.forward(x_total_fea)
    
      # calculate initial pred, loss and acc
      ori_loss = self.loss_fn(x_total_output, y_global)

      # zero all the existing gradients
      self.feature.zero_grad()
      self.classifier.zero_grad()
   
      # backward loss
      ori_loss.backward()

      # collect datagrad
      grad_crop_style_mean_block1 = [crop_style_mean_block1[i].grad.detach() for i in range(num_crops)]
      grad_crop_style_std_block1 = [crop_style_std_block1[i].grad.detach() for i in range(num_crops)]
      grad_ori_style_mean_block1 = ori_style_mean_block1.grad.detach()
      grad_ori_style_std_block1 = ori_style_std_block1.grad.detach()

      # fgsm style attack
      index_mean = torch.randint(0, len(epsilon_list), (1, ))[0]
      index_std = torch.randint(0, len(epsilon_list), (1, ))[0]
      epsilon_mean = epsilon_list[index_mean]
      epsilon_std = epsilon_list[index_std]

      # plot_3d_line_two_datasets(grad_ori_style_mean_block1, grad_ori_style_std_block1)
      adv_style_mean_block1 = fgsm_attack(ori_style_mean_block1, epsilon_mean, grad_ori_style_mean_block1, grad_crop_style_mean_block1)
      adv_style_std_block1 = fgsm_attack(ori_style_std_block1, epsilon_std, grad_ori_style_std_block1, grad_crop_style_std_block1)

    # add zero_grad
    self.feature.zero_grad()
    self.classifier.zero_grad()

    if('2' in blocklist and epsilon_list[1] != 0):
      # forward block1
      x_total_block1 = self.feature.forward_block1(x_total)
      x_adv_block1 = changeNewAdvStyle(x_total_block1[num_crops * Episode_batch:], adv_style_mean_block1, adv_style_std_block1, p_thred=P_THRED_ATTACK)
      x_total_block1 = torch.cat((x_total_block1[ :num_crops * Episode_batch], x_adv_block1), dim=0)
      x_total_block2 = self.feature.forward_block2(x_total_block1)

      feat_size_block2 = x_total_block2[num_crops * Episode_batch:].size()

      x_crop_block2 = [x_total_block2[i * Episode_batch: (i + 1) * Episode_batch] for i in range(num_crops)]

      x_ori_block2 = x_total_block2[num_crops * Episode_batch:]

      crop_style_mean_block2 = []
      crop_style_std_block2 = []
      for i in range(num_crops):
        crop_style_mean_block2_i, crop_style_std_block2_i = calc_mean_std(x_crop_block2[i])
        crop_style_mean_block2.append(crop_style_mean_block2_i)
        crop_style_std_block2.append(crop_style_std_block2_i)

      ori_style_mean_block2, ori_style_std_block2 = calc_mean_std(x_ori_block2)

      # set them as learnable parameters
      crop_style_mean_block2 = [torch.nn.Parameter(crop_style_mean_block2_i) for crop_style_mean_block2_i in crop_style_mean_block2]
      crop_style_std_block2 = [torch.nn.Parameter(crop_style_std_block2_i) for crop_style_std_block2_i in crop_style_std_block2]
      ori_style_mean_block2  = torch.nn.Parameter(ori_style_mean_block2)
      ori_style_std_block2 = torch.nn.Parameter(ori_style_std_block2)

      for i in range(num_crops):
        crop_style_mean_block2[i].requires_grad_()
        crop_style_std_block2[i].requires_grad_()

      ori_style_mean_block2.requires_grad_()
      ori_style_std_block2.requires_grad_()

      # contain ori_style_mean_block2 in the graph 
      x_crop_normalized_block2 = [(x_crop_block2[i] - crop_style_mean_block2[i].detach().expand(feat_size_block2)) / crop_style_std_block2[i].detach().expand(feat_size_block2) for i in range(num_crops)]
      x_crop_block2 = [x_crop_normalized_block2[i] * crop_style_std_block2[i].expand(feat_size_block2) + crop_style_mean_block2[i].expand(feat_size_block2) for i in range(num_crops)]

      x_normalized_block2 = (x_ori_block2 - ori_style_mean_block2.detach().expand(feat_size_block2)) / ori_style_std_block2.detach().expand(feat_size_block2)
      x_ori_block2 = x_normalized_block2 * ori_style_std_block2.expand(feat_size_block2) + ori_style_mean_block2.expand(feat_size_block2)
      
      # pass the rest model
      x_crop_block2 = torch.cat(x_crop_block2, dim=0)
      x_total_block2 = torch.cat((x_crop_block2, x_ori_block2), dim=0)
      x_total_block3 = self.feature.forward_block3(x_total_block2)
      x_total_block4 = self.feature.forward_block4(x_total_block3)
      x_total_fea = self.feature.forward_rest(x_total_block4)
      x_total_output = self.classifier.forward(x_total_fea)
    
      # calculate initial pred, loss and acc
      ori_loss = self.loss_fn(x_total_output, y_global)

      # zero all the existing gradients
      self.feature.zero_grad()
      self.classifier.zero_grad()
   
      # backward loss
      ori_loss.backward()

      # collect datagrad
      grad_crop_style_mean_block2 = [crop_style_mean_block2[i].grad.detach() for i in range(num_crops)]
      grad_crop_style_std_block2 = [crop_style_std_block2[i].grad.detach() for i in range(num_crops)]
      grad_ori_style_mean_block2 = ori_style_mean_block2.grad.detach()
      grad_ori_style_std_block2 = ori_style_std_block2.grad.detach()

      # fgsm style attack
      index_mean = torch.randint(0, len(epsilon_list), (1, ))[0]
      index_std = torch.randint(0, len(epsilon_list), (1, ))[0]
      
      epsilon_mean = epsilon_list[index_mean]
      epsilon_std = epsilon_list[index_std]

      adv_style_mean_block2 = fgsm_attack(ori_style_mean_block2, epsilon_mean, grad_ori_style_mean_block2, grad_crop_style_mean_block2)
      adv_style_std_block2 = fgsm_attack(ori_style_std_block2, epsilon_std, grad_ori_style_std_block2, grad_crop_style_std_block2)

    # add zero_grad
    self.feature.zero_grad()
    self.classifier.zero_grad()

    if('3' in blocklist and epsilon_list[2] != 0):
      # forward block1
      x_total_block1 = self.feature.forward_block1(x_total)
      x_adv_block1 = changeNewAdvStyle(x_total_block1[num_crops * Episode_batch:], adv_style_mean_block1, adv_style_std_block1, p_thred=P_THRED_ATTACK)
      x_total_block1 = torch.cat((x_total_block1[ :num_crops * Episode_batch], x_adv_block1), dim=0)
      x_total_block2 = self.feature.forward_block2(x_total_block1)
      x_adv_block2 = changeNewAdvStyle(x_total_block2[num_crops * Episode_batch:], adv_style_mean_block2, adv_style_std_block2, p_thred=P_THRED_ATTACK)
      x_total_block2 = torch.cat((x_total_block2[ :num_crops * Episode_batch], x_adv_block2), dim=0)
      x_total_block3 = self.feature.forward_block3(x_total_block2)

      feat_size_block3 = x_total_block3[num_crops * Episode_batch:].size()
      x_crop_block3 = [x_total_block3[i * Episode_batch: (i + 1) * Episode_batch] for i in range(num_crops)]
      x_ori_block3 = x_total_block3[num_crops * Episode_batch:]

      crop_style_mean_block3 = []
      crop_style_std_block3 = []
      for i in range(num_crops):
        crop_style_mean_block3_i, crop_style_std_block3_i = calc_mean_std(x_crop_block3[i])
        crop_style_mean_block3.append(crop_style_mean_block3_i)
        crop_style_std_block3.append(crop_style_std_block3_i)

      ori_style_mean_block3, ori_style_std_block3 = calc_mean_std(x_ori_block3)

      # set them as learnable parameters
      crop_style_mean_block3 = [torch.nn.Parameter(crop_style_mean_block3_i) for crop_style_mean_block3_i in crop_style_mean_block3]
      crop_style_std_block3 = [torch.nn.Parameter(crop_style_std_block3_i) for crop_style_std_block3_i in crop_style_std_block3]

      ori_style_mean_block3  = torch.nn.Parameter(ori_style_mean_block3)
      ori_style_std_block3 = torch.nn.Parameter(ori_style_std_block3)

      for i in range(num_crops):
        crop_style_mean_block3[i].requires_grad_()
        crop_style_std_block3[i].requires_grad_()

      ori_style_mean_block3.requires_grad_()
      ori_style_std_block3.requires_grad_()

      # contain ori_style_mean_block2 in the graph 
      x_crop_normalized_block3 = [(x_crop_block3[i] - crop_style_mean_block3[i].detach().expand(feat_size_block3)) / crop_style_std_block3[i].detach().expand(feat_size_block3) for i in range(num_crops)]
      x_crop_block3 = [x_crop_normalized_block3[i] * crop_style_std_block3[i].expand(feat_size_block3) + crop_style_mean_block3[i].expand(feat_size_block3) for i in range(num_crops)]
      x_normalized_block3 = (x_ori_block3 - ori_style_mean_block3.detach().expand(feat_size_block3)) / ori_style_std_block3.detach().expand(feat_size_block3)
      x_ori_block3 = x_normalized_block3 * ori_style_std_block3.expand(feat_size_block3) + ori_style_mean_block3.expand(feat_size_block3)
      
      # pass the rest model
      x_crop_block3 = torch.cat(x_crop_block3, dim=0)
      x_total_block3 = torch.cat((x_crop_block3, x_ori_block3), dim=0)
      x_total_block4 = self.feature.forward_block4(x_total_block3)
      x_total_fea = self.feature.forward_rest(x_total_block4)
      x_total_output = self.classifier.forward(x_total_fea)
    
      # calculate initial pred, loss and acc
      ori_loss = self.loss_fn(x_total_output, y_global)

      # zero all the existing gradients
      self.feature.zero_grad()
      self.classifier.zero_grad()
   
      # backward loss
      ori_loss.backward()

      # collect datagrad
      grad_crop_style_mean_block3 = [crop_style_mean_block3[i].grad.detach() for i in range(num_crops)]
      grad_crop_style_std_block3 = [crop_style_std_block3[i].grad.detach() for i in range(num_crops)]
      grad_ori_style_mean_block3 = ori_style_mean_block3.grad.detach()
      grad_ori_style_std_block3 = ori_style_std_block3.grad.detach()

      # fgsm style attack
      index_mean = torch.randint(0, len(epsilon_list), (1, ))[0]
      index_std = torch.randint(0, len(epsilon_list), (1, ))[0]

      epsilon_mean = epsilon_list[index_mean]
      epsilon_std = epsilon_list[index_std]

      adv_style_mean_block3 = fgsm_attack(ori_style_mean_block3, epsilon_mean, grad_ori_style_mean_block3, grad_crop_style_mean_block3)
      adv_style_std_block3 = fgsm_attack(ori_style_std_block3, epsilon_std, grad_ori_style_std_block3, grad_crop_style_std_block3)

    # add zero_grad
    self.feature.zero_grad()
    self.classifier.zero_grad()

    return adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3 

  def set_statues_of_modules(self, flag):
    if(flag=='eval'):
      self.feature.eval()
      self.fc.eval()
      self.gnn.eval()
      self.classifier.eval()
      self.domain_discriminators.eval()
    elif(flag=='train'):
      self.feature.train()
      self.fc.train()
      self.gnn.train()
      self.classifier.train()
      self.domain_discriminators.train()
    return 

  def set_forward_loss_SVasP(self, x, global_y, epsilon_list, epoch):
    ##################################################################
    if isinstance(x, list):
      num_crops = len(x) - 1
      x_crops = x[:num_crops]
      x_ori = x[-1]
      x_adv = x[-1]

    else:
      x_ori = x
      x_adv = x

    ##################################################################
    # 1. SVasP
    self.set_statues_of_modules('eval') 

    adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3 = \
          self.adversarial_attack_Incre(x, global_y, epsilon_list)
 
    self.feature.zero_grad()
    self.fc.zero_grad()
    self.classifier.zero_grad()
    self.gnn.zero_grad()
    self.domain_discriminators.zero_grad()

    #################################################################
    # 2. forward and get loss
    self.set_statues_of_modules('train')

    # define y_query for FSL
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()

    # forward x_ori 
    x_ori = x_ori.cuda()
    x_size = x_ori.size()
    x_ori = x_ori.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    Batch = x_ori.size(0)
    global_y = global_y.view(x_size[0]*x_size[1]).cuda()
    x_ori_block1 = self.feature.forward_block1(x_ori)
    x_ori_block2 = self.feature.forward_block2(x_ori_block1)
    x_ori_block3 = self.feature.forward_block3(x_ori_block2)
    x_ori_block4 = self.feature.forward_block4(x_ori_block3)
    x_ori_fea = self.feature.forward_rest(x_ori_block4)

    # ori cls global loss    
    scores_cls_ori = self.classifier.forward(x_ori_fea)
    scores_cls_ori_glpbal = [scores_cls_ori for i in range(num_crops)]
    scores_cls_ori_global = torch.cat(scores_cls_ori_glpbal, dim=0)
    scores_cls_ori_global_softmax = torch.softmax(scores_cls_ori_global, dim=1)
    loss_cls_ori = self.loss_fn(scores_cls_ori, global_y)

    # ori FSL scores and losses
    x_ori_z = self.fc(x_ori_fea)
    x_ori_z = x_ori_z.view(self.n_way, -1, x_ori_z.size(1))
    x_ori_z_stack = [torch.cat([x_ori_z[:, :self.n_support], x_ori_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, x_ori_z.size(2)) for i in range(self.n_query)]
    assert(x_ori_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores_fsl_ori = self.forward_gnn(x_ori_z_stack)
    loss_fsl_ori = self.loss_fn(scores_fsl_ori, y_query)

    # forward x_adv
    # layer_drop_flag = self.select_layers(layer_wise_prob=self.layer_wise_prob)

    x_adv = x_adv.cuda()
    x_adv = x_adv.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    x_adv_block1 = self.feature.forward_block1(x_adv)
    x_adv_block1_newStyle = changeNewAdvStyle(x_adv_block1, adv_style_mean_block1, adv_style_std_block1, p_thred = P_THRED) 
    x_adv_block2 = self.feature.forward_block2(x_adv_block1_newStyle)
    x_adv_block2_newStyle = changeNewAdvStyle(x_adv_block2, adv_style_mean_block2, adv_style_std_block2, p_thred = P_THRED)
    x_adv_block3 = self.feature.forward_block3(x_adv_block2_newStyle)
    x_adv_block3_newStyle = changeNewAdvStyle(x_adv_block3, adv_style_mean_block3, adv_style_std_block3, p_thred = P_THRED)
    x_adv_block4 = self.feature.forward_block4(x_adv_block3_newStyle)
    x_adv_fea = self.feature.forward_rest(x_adv_block4)

    # adv cls gloabl loss
    scores_cls_adv = self.classifier.forward(x_adv_fea)
    loss_cls_adv = self.loss_fn(scores_cls_adv, global_y)

    # adv FSL scores and losses
    x_adv_z = self.fc(x_adv_fea)
    x_adv_z = x_adv_z.view(self.n_way, -1, x_adv_z.size(1))
    x_adv_z_stack = [torch.cat([x_adv_z[:, :self.n_support], x_adv_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, x_adv_z.size(2)) for i in range(self.n_query)]
    assert(x_adv_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores_fsl_adv = self.forward_gnn(x_adv_z_stack)
    loss_fsl_adv = self.loss_fn(scores_fsl_adv, y_query)

    # forward x_crop

    x_crops = [x_crop.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4]) for x_crop in x_crops]

    x_crop = torch.cat(x_crops, dim=0)
    x_crop = x_crop.cuda()
    x_crop_block1 = self.feature.forward_block1(x_crop)
    x_crop_block2 = self.feature.forward_block2(x_crop_block1)
    x_crop_block3 = self.feature.forward_block3(x_crop_block2)
    x_crop_block4 = self.feature.forward_block4(x_crop_block3)
    x_crop_fea = self.feature.forward_rest(x_crop_block4)

    # x_crop cls global loss
    global_y_crop = [global_y for i in range(num_crops)]
    global_y_crop = torch.cat(global_y_crop, dim=0)
    scores_cls_crop = self.classifier.forward(x_crop_fea)
    scores_cls_crop_softmax = torch.softmax(scores_cls_crop, dim=1)

    #crop-loss-1
    loss_cls_crop_1 = self.loss_fn(scores_cls_crop, global_y_crop)
    #crop-loss-2
    loss_cls_crop_2 = torch.mean(torch.sum(torch.log(scores_cls_crop_softmax**(-scores_cls_ori_global_softmax)), dim=1))
    
    loss_cls_crop = (1 - self.lambd_crop) * loss_cls_crop_1 + self.lambd_crop * loss_cls_crop_2
                                                                                           
    # domain loss
    original_domain_labels_crop = torch.tensor([0], device=x_ori.device).expand(num_crops * Batch).long()
    original_domain_labels = torch.tensor([0], device=x_ori.device).expand(Batch).long()
    adv_domain_labels = torch.tensor([1], device=x_adv.device).expand(Batch).long()

    crop_domain = self.domain_discriminators(x_crop_fea)
    ori_domain = self.domain_discriminators(x_ori_fea)
    adv_domain = self.domain_discriminators(x_adv_fea)

    loss_domain = (self.loss_fn(crop_domain, original_domain_labels_crop) + self.loss_fn(ori_domain, original_domain_labels) + self.loss_fn(adv_domain, adv_domain_labels)) / 3.0

    return scores_fsl_ori, scores_fsl_adv, scores_cls_ori, scores_cls_adv, loss_fsl_ori, loss_cls_ori, loss_fsl_adv, loss_cls_adv, loss_domain, loss_cls_crop
