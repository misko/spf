import os
import pickle
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.image_utils import (detector_positions_to_theta_grid,
                         labels_to_source_images, radio_to_image)
from utils.plot import plot_space


class SessionsDataset(Dataset):

	def __init__(self, root_dir, snapshots_in_sample=5):
		"""
		Arguments:
			root_dir (string): Directory with all the images.
		"""
		self.root_dir = root_dir
		self.filenames=sorted(filter(lambda x : 'args' not in x ,[ "%s/%s" % (self.root_dir,x) for x in  os.listdir(self.root_dir)]))
		self.args=pickle.load(open("/".join([self.root_dir,'args.pkl']),'rb'))
		if self.args.sessions!=len(self.filenames): # make sure its the right dataset
			print("WARNING DATASET LOOKS LIKE IT IS MISSING SOME SESSIONS!")
		assert(self.args.time_steps>=snapshots_in_sample)
		self.samples_per_session=self.args.time_steps-snapshots_in_sample+1
		self.snapshots_in_sample=snapshots_in_sample

	def idx_to_filename_and_start_idx(self,idx):
		assert(idx>=0 and idx<=self.samples_per_session*len(self.filenames))
		return self.filenames[idx//self.samples_per_session],idx%self.samples_per_session

	def __len__(self):
		return self.samples_per_session*len(self.filenames)

	def __getitem__(self, idx):
		filename,start_idx=self.idx_to_filename_and_start_idx(idx)
		session=pickle.load(open(filename,'rb'))
		end_idx=start_idx+self.snapshots_in_sample
		return { k:session[k][start_idx:end_idx] for k in session.keys()}

class SessionsDatasetTask1Simple(SessionsDataset):
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

class SessionsDatasetTask2Simple(SessionsDataset):
	def __getitem__(self,idx):
		d=super().__getitem__(idx)
		d['source_positions_at_t']=d['source_positions_at_t'] #/self.args.width
		d['detector_position_at_t']=d['detector_position_at_t'] #/self.args.width

		d['source_image_at_t']=labels_to_source_images(d['source_positions_at_t'][None],self.args.width)[0]
		d['detector_theta_image_at_t']=detector_positions_to_theta_grid(d['detector_position_at_t'][None],self.args.width)[0]
		d['radio_image_at_t']=radio_to_image(d['beam_former_outputs_at_t'][None],d['detector_theta_image_at_t'][None])[0]
		#featurie a really simple way
		return d #,d['source_positions_at_t']


def collate_fn(_in):
	d={ k:torch.from_numpy(np.stack([ x[k] for x in _in ])) for k in _in[0]}
	b,s,n_sources,_=d['source_positions_at_t'].shape

	times=d['time_stamps']/(0.00001+d['time_stamps'].max(axis=2,keepdim=True)[0]) 

	#deal with source positions
	source_positions=d['source_positions_at_t'][torch.where(d['broadcasting_positions_at_t']==1)[:-1]].reshape(b,s,2).float()
	#d['source_image_at_t']=labels_to_source_images(d['source_positions_at_t'],128)

	#deal with detector position features
	diffs=source_positions-d['detector_position_at_t']
	source_theta=(torch.atan2(diffs[...,1],diffs[...,0]))[:,:,None]/np.pi # batch, snapshot,1, x ,y 
	distances=torch.sqrt(torch.pow(diffs, 2).sum(axis=2,keepdim=True))

	space_diffs=(d['detector_position_at_t'][:,:-1]-d['detector_position_at_t'][:,1:])
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
		source_positions,
		source_theta,
		distances,
		0*space_delta,
		0*space_theta,
		0*space_dist
	], axis=2).float() #.to(device)

	#create the features
	radio_inputs=torch.cat(
		[
			d['beam_former_outputs_at_t'].max(axis=2,keepdim=True)[0],
			d['beam_former_outputs_at_t']/d['beam_former_outputs_at_t'].max(axis=2,keepdim=True)[0],
			times-times.max(axis=2,keepdim=True)[0],
			d['detector_position_at_t'],
			space_delta,
			space_theta,
			space_dist
		],
		dim=2
	).float() #.to(device)
	radio_images=d['radio_image_at_t'].float()
	
	label_images=d['source_image_at_t'].float()
	return radio_inputs,radio_images,labels,label_images

class SessionsDatasetTask2(SessionsDataset):
	def __getitem__(self,idx):
		d=super().__getitem__(idx)
		#featurie a really simple way
			