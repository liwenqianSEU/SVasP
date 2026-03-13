import numpy as np
import torch
import torch.optim
import os
import random 

from methods.backbone_multiblock import model_dict
from data.datamgr import SetDataManager, Eposide_DataManager
from methods.SVasP_RN_GNN import SVasPGNN

from options import parse_args, get_resume_file, load_warmup_state
from test_function_bscdfsl_benchmark import test_bestmodel_bscdfsl


def train(base_loader, val_loader,  model, start_epoch, stop_epoch, params):

  # get optimizer and checkpoint path
  optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # for validation
  max_acc = 0
  total_it = 0

  # start
  for epoch in range(start_epoch, stop_epoch):
    model.train()
    total_it = model.train_loop(epoch, base_loader, optimizer, model, total_it) #model are called by reference, no need to return
    model.eval()

    acc = model.test_loop( val_loader)
    if acc > max_acc :
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    else:
      print("GG! best accuracy {:f}".format(max_acc))

    #if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
    if(epoch == stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

  return model

def record_test_result_bscdfsl(params):
  seed = 0
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False 

  print('hhhhhhh testing for bscdfsl, seed = %d' % seed)
  acc_file_path = os.path.join(params.checkpoint_dir, 'acc_bscdfsl.txt')
  acc_file = open(acc_file_path,'a')
  epoch_id = -1
  name = params.name
  n_shot = params.n_shot
  acc_mean_chestx = test_bestmodel_bscdfsl(acc_file, name, 'ChestX', n_shot, epoch_id)
  acc_mean_isic = test_bestmodel_bscdfsl(acc_file, name, 'ISIC', n_shot, epoch_id)
  acc_mean_eurosat = test_bestmodel_bscdfsl(acc_file, name, 'EuroSAT', n_shot, epoch_id)
  acc_mean_crop = test_bestmodel_bscdfsl(acc_file, name, 'CropDisease', n_shot, epoch_id)
  acc_mean_cub = test_bestmodel_bscdfsl(acc_file, name, 'CUB', n_shot, epoch_id)
  acc_mean_cars = test_bestmodel_bscdfsl(acc_file, name, 'Cars', n_shot, epoch_id)
  acc_mean_places = test_bestmodel_bscdfsl(acc_file, name, 'Places', n_shot, epoch_id)
  acc_mean_plantae = test_bestmodel_bscdfsl(acc_file, name, 'plantae', n_shot, epoch_id)
  acc_mean_mini = test_bestmodel_bscdfsl(acc_file, name, 'miniImagenet', n_shot, epoch_id)

  acc_all_mean_8 = ( acc_mean_chestx + acc_mean_isic + acc_mean_eurosat + acc_mean_crop + acc_mean_cub + acc_mean_cars + acc_mean_places + acc_mean_plantae ) / 8
  acc_all_mean_9 = ( acc_mean_chestx + acc_mean_isic + acc_mean_eurosat + acc_mean_crop + acc_mean_cub + acc_mean_cars + acc_mean_places + acc_mean_plantae + acc_mean_mini ) / 9
  print('ORI Average_8: Acc = %4.2f%%' %(acc_all_mean_8), file = acc_file)
  print('ORI Average_9: Acc = %4.2f%%' %(acc_all_mean_9), file = acc_file)

  acc_file.close()
  return

# --- main function ---
if __name__=='__main__':
  #fix seed 
  seed = 0
  print("set seed = %d" % seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False 

  # parser argument
  params = parse_args('train')

  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')
  print('  train with single seen domain {}'.format(params.dataset))
  base_file  = os.path.join(params.data_dir, params.dataset, 'base.json')
  val_file   = os.path.join(params.data_dir, params.dataset, 'val.json')

  # model
  print('\n--- build model ---')
  image_size = 224
  
  #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
  n_query = max(1, int(16* params.test_n_way/params.train_n_way))

  train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
  base_datamgr            = Eposide_DataManager(base_file, n_crops=params.n_crops, n_episode=params.n_episode, **train_few_shot_params)
  base_loader             = base_datamgr.get_data_loader()

  test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot)
  val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
  val_loader              = val_datamgr.get_data_loader( val_file, aug = False)

  model = SVasPGNN( model_dict[params.model], params, tf_path=params.tf_dir, **train_few_shot_params)
  model = model.cuda()

  # load model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      tmp = torch.load(resume_file)
      start_epoch = tmp['epoch']+1
      model.load_state_dict(tmp['state'])
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  else:
    if params.warmup == 'gg3b0':
      raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
    state = load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup))
    model.feature.load_state_dict(state, strict=False)

  import time
  start =time.perf_counter()
  # training
  print('\n--- start the training ---')
  model = train(base_loader, val_loader, model, start_epoch, stop_epoch, params)
  end=time.perf_counter()
  print('Running time: %s Seconds: %s Min: %s Min per epoch'%(end-start, (end-start)/60, (end-start)/60/params.stop_epoch))

  # testing
  record_test_result_bscdfsl(params)

