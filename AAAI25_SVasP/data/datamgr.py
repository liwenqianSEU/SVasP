# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import json
import torch
import os
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, MultiSetDataset, EpisodicBatchSampler, MultiEpisodicBatchSampler, RandomLabeledTargetDataset
from abc import abstractmethod

identity = lambda x:x

class TransformLoader:
  def __init__(self, image_size,
      normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
    self.image_size = image_size
    self.normalize_param = normalize_param
    self.jitter_param = jitter_param

  def parse_transform(self, transform_type):
    if transform_type=='ImageJitter':
      method = add_transforms.ImageJitter( self.jitter_param )
      return method
    method = getattr(transforms, transform_type)

    if transform_type=='RandomResizedCrop':
      return method(self.image_size)
    elif transform_type=='CenterCrop':
      return method(self.image_size)
    elif transform_type=='Resize':
      return method([int(self.image_size*1.15), int(self.image_size*1.15)])
    elif transform_type=='Normalize':
      return method(**self.normalize_param )
    else:
      return method()

  def get_composed_transform(self, aug = False):
    if aug:
      transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
    else:
      transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

    transform_funcs = [ self.parse_transform(x) for x in transform_list]
    transform = transforms.Compose(transform_funcs)
    return transform




# added by fuyuqian in 2021 0107
class LabeledTargetDataset:
   def __init__(self, data_file,image_size, batch_size = 16, aug=True):
       with open(data_file, 'r') as f:
           self.meta = json.load(f)
       #print('len of labeled target data:', len(self.meta['image_names']))
       # define transform
       self.batch_size = batch_size
       self.trans_loader = TransformLoader(image_size)
       self.transform = self.trans_loader.get_composed_transform(aug)

   def get_epoch(self):
       # return random
       idx_list = [i for i in range(len(self.meta['image_names']))]
       selected_idx_list = random.sample(idx_list, self.batch_size)
       
       img_list = []
       img_label = []
     
       for idx in selected_idx_list:
           image_path = self.meta['image_names'][idx]
           image_label = self.meta['image_labels'][idx]
           img = Image.open(image_path).convert('RGB')
           img = self.transform(img)
           img_list.append(img)
           img_label.append(image_label)
       #print(img_label)
       img_list = torch.stack(img_list)
       #img_label = torch.stack(img_label)
       img_label = torch.LongTensor(img_label)
       #print('img_list:', img_list.size())
       #print('img_label:', img_label.size())
       return img_list, img_label



class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = SimpleDataset(data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader


# added in 20210108
class RandomLabeledTargetDataManager(DataManager):
  def __init__(self, image_size, batch_size):
    super(RandomLabeledTargetDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file, data_file_miniImagenet, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = RandomLabeledTargetDataset(data_file, data_file_miniImagenet, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader

class SetDataManager(DataManager):
  def __init__(self, image_size, n_way, n_support, n_query, n_episode=100):
    super(SetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.batch_size = n_support + n_query
    self.n_episode = n_episode

    self.trans_loader = TransformLoader(image_size)
    #print('datamgr:', 'SetDataManager:', 'n_way:', self.n_way, 'batch_size:', self.batch_size)

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    if isinstance(data_file, list):
      dataset = MultiSetDataset( data_file , self.batch_size, transform )
      sampler = MultiEpisodicBatchSampler(dataset.lens(), self.n_way, self.n_episode )
    else:
      dataset = SetDataset( data_file , self.batch_size, transform )
      sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode )
    data_loader_params = dict(batch_sampler = sampler,  num_workers=4)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

'''

# added in 20210109
class RandomLabeledTargetSetDataManager(DataManager):
  def __init__(self, image_size, n_way, n_support, n_query, n_episode=100):
    super(RandomLabeledTargetSetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.batch_size = n_support + n_query
    self.n_episode = n_episode

    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    if isinstance(data_file, list):
      dataset = MultiSetDataset( data_file , self.batch_size, transform )
      sampler = MultiEpisodicBatchSampler(dataset.lens(), self.n_way, self.n_episode )
    else:
      dataset = SetDataset( data_file , self.batch_size, transform )
      sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode )
    data_loader_params = dict(batch_sampler = sampler,  num_workers=4)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader
 '''
 
class SetDataset_Crop:
  def __init__(self, data_file, batch_size, n_crops):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)

    self.cl_list = np.unique(self.meta['image_labels']).tolist()
    #print('dataset:', 'SetDataset:', 'cl_list:', self.cl_list)

    self.sub_meta = {}
    for cl in self.cl_list:
      self.sub_meta[cl] = []

    for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
      self.sub_meta[y].append(x)
  
    self.sub_dataloader = [] 
    sub_data_loader_params = dict(batch_size = batch_size,
                              shuffle = True,
                              num_workers = 0, #use main thread only or may receive multiple batches
                              pin_memory = False)        
    for cl in self.cl_list:
      sub_dataset = SubDataset_Crop(self.sub_meta[cl], cl, n_crops, min_size = batch_size)
      # sub_data_loader = sub_dataset.__getitem__(0)
      self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

  def __getitem__(self, i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.sub_dataloader)
    
class SubDataset_Crop:
  def __init__(self, sub_meta, cl, n_crops, target_transform=identity, min_size = 50,
    size_crops=[224],
    min_scale_crops=[0.2],
    max_scale_crops=[0.4],
    ):
    self.cl = cl
    self.target_transform = target_transform
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    trans = []
    for i in range(len(size_crops)):
      randomresizedcrop = transforms.RandomResizedCrop(
          size_crops[i],
          scale=(min_scale_crops[i], max_scale_crops[i]),
      )
      trans.extend([transforms.Compose([
          randomresizedcrop,
          transforms.ToTensor(),
          transforms.Normalize(mean=mean, std=std)])
      ] * n_crops)
    self.trans = trans

    self.jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)
    
    self.global_transforms = transforms.Compose([
            transforms.Resize([224,224]),
            add_transforms.ImageJitter(self.jitter_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    
    self.sub_meta = sub_meta
    if len(self.sub_meta) < min_size:
      #print('dataset:', 'SubDataset:', 'len of self_meta:', len(self.sub_meta),' < 50')
      idxs = [i % len(self.sub_meta) for i in range(min_size)]
      #print('dataset:', 'SubDataset:', 'idxs:', idxs)
      self.sub_meta = np.array(self.sub_meta)[idxs].tolist()
      #print('dataset:', 'SubDataset:', 'sub_meat:', self.sub_meta)
      
  def __getitem__(self,i):
    image_path = os.path.join( self.sub_meta[i])
    img = Image.open(image_path).convert('RGB')

    multi_crops = list(map(lambda trans: trans(img), self.trans))
    raw_image = self.global_transforms(img)
    multi_crops.append(raw_image)
    
    target = self.target_transform(self.cl)

    return multi_crops, target
  
  def __len__(self):
      return len(self.sub_meta)

class Eposide_DataManager():
  def __init__(self, data_path, n_way=5, n_support=1, n_query=15, n_episode=100, n_crops=2):        
    super(Eposide_DataManager, self).__init__()
    self.data_path = data_path
    self.n_way = n_way
    self.batch_size = n_support + n_query
    self.n_episode = n_episode
    self.n_crops = n_crops

  def get_data_loader(self): 
    dataset = SetDataset_Crop(self.data_path, self.batch_size, self.n_crops)
    sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)  
    data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=True)   
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader