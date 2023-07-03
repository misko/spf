import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.image_utils import (detector_positions_to_theta_grid,
                         labels_to_source_images, radio_to_image)
from utils.plot import plot_space
from utils.spf_generate import generate_session
from compress_pickle import dump, load

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
		return { k:session[k][start_idx:end_idx] for k in session.keys()}

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

class SessionsDatasetTask2(SessionsDataset):
	def __getitem__(self,idx):
		d=super().__getitem__(idx)
		#normalize these before heading out
		d['source_positions_at_t_normalized']=2*(d['source_positions_at_t']/self.args.width-0.5)
		d['detector_position_at_t_normalized']=2*(d['detector_position_at_t']/self.args.width-0.5)
		d['source_distance_at_t_normalized']=d['source_distance_at_t'].mean(axis=2)/(self.args.width/2)
		return d #,d['source_positions_at_t']

class SessionsDatasetTask2WithImages(SessionsDataset):
	def __getitem__(self,idx):
		d=super().__getitem__(idx)
		#normalize these before heading out
		d['source_positions_at_t_normalized']=2*(d['source_positions_at_t']/self.args.width-0.5)
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

def collate_fn(_in):
	d={ k:torch.from_numpy(np.stack([ x[k] for x in _in ])) for k in _in[0]}
	b,s,n_sources,_=d['source_positions_at_t'].shape

	times=d['time_stamps']/(0.00001+d['time_stamps'].max(axis=2,keepdim=True)[0]) 

	#deal with source positions
	source_positions=d['source_positions_at_t_normalized'][torch.where(d['broadcasting_positions_at_t']==1)[:-1]].reshape(b,s,2).float()

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
		source_positions, # zero center the positions
		(source_theta+detector_theta+1)%2.0-1, # initialy in units of np.pi?
		distances, # try to zero center?
		space_delta,
		space_theta,
		space_dist
	], axis=2).float() #.to(device)
	#breakpoint()
	#create the features
	radio_inputs=torch.cat(
		[
			d['detector_position_at_t_normalized'],
			times-times.max(axis=2,keepdim=True)[0],
			space_delta,
			space_theta,
			space_dist,
			detector_theta,
			torch.log(d['beam_former_outputs_at_t'].mean(axis=2,keepdim=True))/20,
			d['beam_former_outputs_at_t']/d['beam_former_outputs_at_t'].mean(axis=2,keepdim=True), # maybe pass in log values?
		],
		dim=2
	).float() #.to(device)
	if 'radio_image_at_t' in d:
		radio_images=d['radio_image_at_t'].float()
		label_images=d['source_image_at_t'].float()
		return radio_inputs,radio_images,labels,label_images
	return radio_inputs,None,labels,None
