###
# Experiment 1 : wall drones , receiver center is x=135.27 cm, y=264.77cm, dist between is 6cm
###

import bisect
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.image_utils import (detector_positions_to_theta_grid,
       labels_to_source_images, radio_to_image)
from utils.spf_generate import generate_session
from compress_pickle import dump, load

output_cols={ # maybe this should get moved to the dataset part...
  'src_pos':[0,1],
  'src_theta':[2],
  'src_dist':[3],
  'det_delta':[4,5],
  'det_theta':[6],
  'det_space':[7],
  'src_v':[8,9]
}

input_cols={
  'det_pos':[0,1],
  'time':[2],
  'space_delta':[3,4],
  'space_theta':[5],
  'space_dist':[6],
  'det_theta2':[7],
}

#from stackoverflow
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class SessionsDataset(Dataset):

  def __init__(self, root_dir, snapshots_in_sample=5):
    """
    Arguments:
      root_dir (string): Directory with all the images.
    """
    self.root_dir = root_dir
    self.args=load("/".join([self.root_dir,'args.pkl']),compression="lzma")
    assert(self.args.time_steps>=snapshots_in_sample)
    self.samples_per_session=self.args.time_steps-snapshots_in_sample+1
    self.snapshots_in_sample=snapshots_in_sample
    if not self.args.live:  
      print("NOT LIVE")
      self.filenames=sorted(filter(lambda x : 'args' not in x ,[ "%s/%s" % (self.root_dir,x) for x in  os.listdir(self.root_dir)]))
      if self.args.sessions!=len(self.filenames): # make sure its the right dataset
        print("WARNING DATASET LOOKS LIKE IT IS MISSING SOME SESSIONS!")

  def idx_to_filename_and_start_idx(self,idx):
    assert(idx>=0 and idx<=self.samples_per_session*len(self.filenames))
    return self.filenames[idx//self.samples_per_session],idx%self.samples_per_session

  def __len__(self):
    if self.args.live:
      return self.samples_per_session*self.args.sessions
    else:
      return self.samples_per_session*len(self.filenames)

  def __getitem__(self, idx):
    session_idx=idx//self.samples_per_session
    start_idx=idx%self.samples_per_session

    if self.args.live:
      session=generate_session((self.args,session_idx))  
    else:
      session=load(self.filenames[session_idx],compression="lzma")
    end_idx=start_idx+self.snapshots_in_sample
    print("GET",session_idx,start_idx)
    return { k:session[k][start_idx:end_idx] for k in session.keys()}

class SessionsDatasetReal(Dataset):

  def get_m(self,filename,bypass=False):
    if bypass:
      return np.memmap(
            filename, 
            dtype='float32', 
            mode='r', 
            shape=(self.snapshots_in_file,self.nthetas+5)) 

    if filename not in self.m_cache:
      self.m_cache[filename]=np.memmap(
            filename, 
            dtype='float32', 
            mode='r', 
            shape=(self.snapshots_in_file,self.nthetas+5)) 
    return self.m_cache[filename]

  def check_file(self,filename):
    try:
      m=self.get_m(filename,bypass=True)
    except:
      print("DROP",filename)
      return False
    status=not (np.abs(m).mean(axis=1)==0).any()
    if status==False:
      print("DROP",filename)
    return status

  def __init__(self, 
               root_dir, 
               snapshots_in_file=400000, 
               nthetas=65,
               snapshots_in_sample=128,
               nsources=1,
               width=3000,
               receiver_pos_x=1352.7,
               receiver_pos_y=2647.7,
               receiver_spacing=60.0,
               step_size=1
              ):
    #time_step,x,y,mean_angle,_mean_angle #0,1,2,3,4
    #m = np.memmap(filename, dtype='float32', mode='r', shape=(,70))
    """
    Arguments:
      root_dir (string): Directory with all the images.
    """
    assert(nsources==1) #TODO implement more
    self.root_dir = root_dir
    self.nthetas=nthetas
    self.thetas=np.linspace(-np.pi,np.pi,self.nthetas)
    self.args=dotdict({
        'width':width,
    })
    self.m_cache={}
    self.receiver_pos=np.array([
        [receiver_pos_x-receiver_spacing/2,receiver_pos_y],
        [receiver_pos_x+receiver_spacing/2,receiver_pos_y]
    ])
    self.detector_position=np.array([[receiver_pos_x,receiver_pos_y]])
    self.snapshots_in_file=snapshots_in_file
    self.snapshots_in_sample=snapshots_in_sample
    self.step_size=step_size
    self.filenames=sorted(
      filter( self.check_file, 
      filter(
        lambda x : '.npy' in x ,[ "%s/%s" % (self.root_dir,x) for x in  os.listdir(self.root_dir)])))
    #self.datas=[
    #    np.memmap(
    #        filename, 
    #        dtype='float32', 
    #        mode='r', 
    #        shape=(self.snapshots_in_file,self.nthetas+5)) for filename in self.filenames
    #]
    self.samples_per_file=[
        self.get_m(filename,bypass=True).shape[0]-(self.snapshots_in_sample*self.step_size) for filename in self.filenames
    ]
    self.cumsum_samples_per_file=np.cumsum([0]+self.samples_per_file)
    self.len=sum(self.samples_per_file)
    self.zeros=np.zeros((self.snapshots_in_sample,5))
    self.ones=np.ones((self.snapshots_in_sample,5))
    self.widths=np.ones((self.snapshots_in_sample,1),dtype=np.int32)*self.args.width
    self.halfpis=np.ones((self.snapshots_in_sample,1))*np.pi/2
    idx_to_fileidx_and_sampleidx={}
    #print("WARNING BY DEFAULT FLIPPING RADIO FEATURE SINCE COORDS WERE WRONG IN PI COLLECT!")
    
  def idx_to_fileidx_and_startidx(self,idx):
    file_idx=bisect.bisect_right(self.cumsum_samples_per_file, idx)-1
    if file_idx>=len(self.samples_per_file):
        return None
    start_idx=(idx-self.cumsum_samples_per_file[file_idx]) #*(self.snapshots_in_sample*self.step_size)
    return file_idx,start_idx
    
  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    fileidx,startidx=self.idx_to_fileidx_and_startidx(idx)
    m=self.get_m(self.filenames[fileidx])[startidx:startidx+self.snapshots_in_sample*self.step_size:self.step_size]
    return {
        'broadcasting_positions_at_t':self.ones[:,[0]][:,None], # TODO multi source
        'source_positions_at_t':m[:,1:3][:,None],
        'source_velocities_at_t':self.zeros[:,:2][:,None], #TODO calc velocity,
        'receiver_positions_at_t':np.broadcast_to(self.receiver_pos[None], (m.shape[0],2, 2)),
        'beam_former_outputs_at_t':m[:,5:],
        'thetas_at_t':np.broadcast_to(self.thetas[None], (m.shape[0],self.thetas.shape[0])), 
        'time_stamps':m[:,[0]], 
        'width_at_t': self.widths,
        'detector_orientation_at_t':self.halfpis,#np.arctan2(1,0)=np.pi/2 
        'detector_position_at_t':np.broadcast_to(self.detector_position, (m.shape[0], 2)),
        'source_theta_at_t':self.zeros[:,[0]][:,None],
        'source_distance_at_t':self.zeros[:,[0]][:,None],
    }


class SessionsDatasetTask1(SessionsDataset):
  def __getitem__(self,idx):
    d=super().__getitem__(idx)
    #featurie a really simple way
    x=torch.Tensor(np.hstack([
      d['receiver_positions_at_t'].reshape(self.snapshots_in_sample,-1),
      d['beam_former_outputs_at_t'].reshape(self.snapshots_in_sample,-1),
      #d['signal_matrixs'].reshape(self.snapshots_in_sample,-1)
      d['time_stamps'].reshape(self.snapshots_in_sample,-1)-d['time_stamps'][0],
      ]))
    y=torch.Tensor(d['source_positions_at_t'][:,0])
    return x,y


def pos_to_rel(p,width):
  return 2*(p/width-0.5)

def rel_to_pos(r,width):
  return ((r/2)+0.5)*width

class SessionsDatasetRealTask2(SessionsDatasetReal):
  def __getitem__(self,idx):
    d=super().__getitem__(idx)
    #normalize these before heading out
    d['source_positions_at_t_normalized_centered']=2*(d['source_positions_at_t']/self.args.width-0.5)
    d['source_velocities_at_t_normalized']=d['source_velocities_at_t']/self.args.width
    d['detector_position_at_t_normalized_centered']=2*(d['detector_position_at_t']/self.args.width-0.5)
    d['source_distance_at_t_normalized']=d['source_distance_at_t'].mean(axis=2)/(self.args.width/2)
    return d #,d['source_positions_at_t']

class SessionsDatasetTask2(SessionsDataset):
  def __getitem__(self,idx):
    d=super().__getitem__(idx)
    #normalize these before heading out
    d['source_positions_at_t_normalized_centered']=2*(d['source_positions_at_t']/self.args.width-0.5)
    d['source_velocities_at_t_normalized']=d['source_velocities_at_t']/self.args.width
    d['detector_position_at_t_normalized_centered']=2*(d['detector_position_at_t']/self.args.width-0.5)
    d['source_distance_at_t_normalized']=d['source_distance_at_t'].mean(axis=2)/(self.args.width/2)
    return d #,d['source_positions_at_t']

class SessionsDatasetTask2WithImages(SessionsDataset):
  def __getitem__(self,idx):
    d=super().__getitem__(idx)
    #normalize these before heading out
    d['source_positions_at_t_normalized']=2*(d['source_positions_at_t']/self.args.width-0.5)
    d['source_velocities_at_t_normalized']=d['source_velocities_at_t']/self.args.width
    d['detector_position_at_t_normalized']=2*(d['detector_position_at_t']/self.args.width-0.5)
    d['source_distance_at_t_normalized']=d['source_distance_at_t'].mean(axis=2)/(self.args.width/2)

    d['source_image_at_t']=labels_to_source_images(d['source_positions_at_t'][None],self.args.width)[0]
    d['detector_theta_image_at_t']=detector_positions_to_theta_grid(d['detector_position_at_t'][None],self.args.width)[0]
    d['radio_image_at_t']=radio_to_image(d['beam_former_outputs_at_t'][None],d['detector_theta_image_at_t'][None],d['detector_orientation_at_t'][None])[0]

    return d #,d['source_positions_at_t']


def collate_fn_beamformer(_in):
  d={ k:torch.from_numpy(np.stack([ x[k] for x in _in ])) for k in _in[0]}
  b,s,n_sources,_=d['source_positions_at_t'].shape

  times=d['time_stamps']/(0.00001+d['time_stamps'].max(axis=2,keepdim=True)[0]) 

  source_theta=d['source_theta_at_t'].mean(axis=2)
  distances=d['source_distance_at_t_normalized'].mean(axis=2,keepdims=True)
  _,_,beam_former_bins=d['beam_former_outputs_at_t'].shape
  perfect_labels=torch.zeros((b,s,beam_former_bins))

  idxs=(beam_former_bins*(d['source_theta_at_t']+np.pi)/(2*np.pi)).int()
  smooth_bins=int(beam_former_bins*0.25*0.5)
  for _b in torch.arange(b):
    for _s in torch.arange(s):
      for smooth in range(-smooth_bins,smooth_bins+1):
        perfect_labels[_b,_s,(idxs[_b,_s]+smooth)%beam_former_bins]=1/(1+smooth**2)
      perfect_labels[_b,_s]/=perfect_labels[_b,_s].sum()+1e-9
  r= {'input':torch.concatenate([
      #d['signal_matrixs_at_t'].reshape(b,s,-1),
      (d['signal_matrixs_at_t']/d['signal_matrixs_at_t'].abs().mean(axis=[2,3],keepdims=True)).reshape(b,s,-1), # normalize the data
      d['signal_matrixs_at_t'].abs().mean(axis=[2,3],keepdims=True).reshape(b,s,-1), # 
      d['detector_orientation_at_t'].to(torch.complex64)],
      axis=2),
    'beamformer':d['beam_former_outputs_at_t'],
    'labels':perfect_labels,
    'thetas':source_theta}
  return r

def collate_fn_transformer_filter(_in):
  d={ k:torch.from_numpy(np.stack([ x[k] for x in _in ])) for k in _in[0]}
  b,s,n_sources,_=d['source_positions_at_t'].shape

  normalized_01_times=d['time_stamps']/(0.0000001+d['time_stamps'].max(axis=1,keepdim=True)[0]) 
  normalized_times=(d['time_stamps']-d['time_stamps'].max(axis=1,keepdims=True)[0])/100

  normalized_pirads_detector_theta=d['detector_orientation_at_t']/np.pi

  space_diffs=(d['detector_position_at_t_normalized_centered'][:,:-1]-d['detector_position_at_t_normalized_centered'][:,1:])
  space_delta=torch.cat([
    torch.zeros(b,1,2),
    space_diffs,
    ],axis=1)

  normalized_pirads_space_theta=torch.cat([  
    torch.zeros(b,1,1),
    (torch.atan2(space_diffs[...,1],space_diffs[...,0]))[:,:,None]/np.pi
  ],axis=1)

  space_dist=torch.cat([  
    torch.zeros(b,1,1),
    torch.sqrt(torch.pow(space_diffs,2).sum(axis=2,keepdim=True))
  ],axis=1)
  return {
    'drone_state':torch.cat(
    [
      d['detector_position_at_t_normalized_centered'], # 2: 2
      #normalized_01_times, #-times.max(axis=2,keepdim=True)[0], # 1: 3
      normalized_times, #-times.max(axis=2,keepdim=True)[0], # 1: 3
      space_delta, # 2: 5
      normalized_pirads_space_theta, # 1: 6
      space_dist, #1: 7
      normalized_pirads_detector_theta, #1: 8
    ],dim=2).float(),
    'times':normalized_times,
    'emitter_position_and_velocity':torch.cat([
      d['source_positions_at_t_normalized_centered'],
      d['source_velocities_at_t_normalized'],
    ],dim=3),
    'emitters_broadcasting':d['broadcasting_positions_at_t'],
    'emitters_n_broadcasts':d['broadcasting_positions_at_t'].cumsum(axis=1),
    'radio_feature':torch.cat(
    [
      torch.log(d['beam_former_outputs_at_t'].mean(axis=2,keepdim=True))/20,
      d['beam_former_outputs_at_t']/d['beam_former_outputs_at_t'].mean(axis=2,keepdim=True), # maybe pass in log values?
    ],
    dim=2
    ).float()

  }

def collate_fn(_in):
  d={ k:torch.from_numpy(np.stack([ x[k] for x in _in ])) for k in _in[0]}
  b,s,n_sources,_=d['source_positions_at_t'].shape

  #times=d['time_stamps']/(0.00001+d['time_stamps'].max(axis=2,keepdim=True)[0]) 
  times=d['time_stamps']/(0.0000001+d['time_stamps'].max(axis=1,keepdim=True)[0]) 

  #deal with source positions
  source_position=d['source_positions_at_t_normalized'][torch.where(d['broadcasting_positions_at_t']==1)[:-1]].reshape(b,s,2).float()
  source_velocity=d['source_velocities_at_t_normalized'][torch.where(d['broadcasting_positions_at_t']==1)[:-1]].reshape(b,s,2).float()

  #deal with detector position features
  #diffs=source_positions-d['detector_position_at_t_normalized']
  #source_thetab=(torch.atan2(diffs[...,1],diffs[...,0]))[:,:,None]/np.pi # batch, snapshot,1, x ,y 
  source_theta=d['source_theta_at_t'].mean(axis=2)/np.pi
  detector_theta=d['detector_orientation_at_t']/np.pi
  #distancesb=torch.sqrt(torch.pow(diffs, 2).sum(axis=2,keepdim=True))
  distances=d['source_distance_at_t_normalized'].mean(axis=2,keepdims=True)
  space_diffs=(d['detector_position_at_t_normalized'][:,:-1]-d['detector_position_at_t_normalized'][:,1:])
  space_delta=torch.cat([
    torch.zeros(b,1,2),
    space_diffs,
    ],axis=1)

  space_theta=torch.cat([  
    torch.zeros(b,1,1),
    (torch.atan2(space_diffs[...,1],space_diffs[...,0]))[:,:,None]/np.pi
  ],axis=1)

  space_dist=torch.cat([  
    torch.zeros(b,1,1),
    torch.sqrt(torch.pow(space_diffs,2).sum(axis=2,keepdim=True))
  ],axis=1)

  #create the labels
  labels=torch.cat([
    source_position, # zero center the positions
    (source_theta+detector_theta+1)%2.0-1, # initialy in units of np.pi?
    distances, # try to zero center?
    space_delta,
    space_theta,
    space_dist,
                source_velocity,
  ], axis=2).float() #.to(device)
  #breakpoint()
  #create the features
  inputs={
    'drone_state':torch.cat(
    [
      d['detector_position_at_t_normalized'],
      times, #-times.max(axis=2,keepdim=True)[0],
      space_delta,
      space_theta,
      space_dist,
      detector_theta,
    ],dim=2).float(),
    'radio_feature':torch.cat(
    [
      torch.log(d['beam_former_outputs_at_t'].mean(axis=2,keepdim=True))/20,
      d['beam_former_outputs_at_t']/d['beam_former_outputs_at_t'].mean(axis=2,keepdim=True), # maybe pass in log values?
    ],
    dim=2
    ).float()
    }
  if 'radio_image_at_t' in d:
    radio_images=d['radio_image_at_t'].float()
    label_images=d['source_image_at_t'].float()
    return {'inputs':inputs,
      'input_images':radio_images,
      'labels':labels,
      'label_images':label_images}
  return {'inputs':inputs,
    'labels':labels}
