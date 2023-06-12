
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
from utils.spf_dataset import *

def plot_single_session(d):
	fig=plt.figure(figsize=(16,4))
	axs=fig.subplots(1,4)
	det_x,det_y=d['detector_position_at_t'][0] #*self.width
	detector_pos_str="Det (X=%d,Y=%d)" % (det_x,det_y)

	axs[0].imshow(d['source_image_at_t'][0,0].T)
	axs[0].set_title("Source emitters")

	axs[1].imshow(d['detector_theta_image_at_t'][0,0].T)
	axs[1].set_title("Detector theta angle")

	axs[2].imshow(d['radio_image_at_t'][0,0].T)
	axs[2].set_title("Radio feature (steering)")
	
	for idx in range(3):
		axs[idx].set_xlabel("X")
		axs[idx].set_ylabel("Y")

	axs[3].plot(d['thetas_at_t'][0],d['beam_former_outputs_at_t'][0])
	axs[3].set_title("Radio feature (steering)")
	axs[3].set_xlabel("Theta")
	axs[3].set_ylabel("Power")
	fig.suptitle(detector_pos_str)
	fig.tight_layout()
	fig.show()
	breakpoint()

if __name__=='__main__':
	sessions_dir='./sessions-default'	
	#test task1
	if False:
		#load a dataset
		ds=SessionsDatasetTask1Simple(sessions_dir)
		ds[253]
		ds=SessionsDataset(sessions_dir)

		r_idxs=np.arange(len(ds))
		np.random.shuffle(r_idxs)
		fig=plt.figure(figsize=(8,8))
		for r_idx in r_idxs[:4]:
			fig.clf()
			axs=fig.subplots(3,3)
			x=ds[r_idx]
			theta_at_pos=detector_positions_to_theta_grid(x['detector_position_at_t'][None],ds.args.width) # batch, snapshot, 1, x ,y 
			dist_at_pos=detector_positions_to_distance(x['detector_position_at_t'][None],ds.args.width) # batch, snapshot, 1, x ,y
			theta_idxs=(((theta_at_pos+np.pi)/(2*np.pi))*257) #.int().float() #~ [2, 8, 1, 128, 128])	
		
			label_images=labels_to_source_images(x['source_positions_at_t'][None],ds.args.width)
		
			_,s,_,_x,_y=dist_at_pos.shape
			for s_idx in np.arange(s):
				m=theta_idxs[0,s_idx]
				for idx in np.arange(257):
					m[m==idx]=np.abs(x['beam_former_outputs_at_t'][s_idx,idx]) #.abs() #.log()
			for idx in np.arange(3):
				axs[idx,0].imshow(theta_idxs[0,idx,0])
				axs[idx,1].imshow(label_images[0,idx,0])
				axs[idx,2].imshow( (theta_idxs[0,:idx+1,0].mean(axis=0)) )
			plt.pause(1)
		#plot the space diagram for some samples
		fig,ax=plt.subplots(2,2,figsize=(8,8))
		[ plot_space(ax[i//2,i%2], ds[r_idxs[i]]) for i in np.arange(4) ]
		plt.title("Task1")
		plt.show()

	#test task2
	if True:
		#load a dataset
		ds=SessionsDatasetTask2Simple(sessions_dir)
		ds[253]
		plot_single_session(ds[0])
		plot_single_session(ds[200])
		plot_single_session(ds[100])
		ds=SessionsDataset(sessions_dir)

		#plot the space diagram for some samples
		fig,ax=plt.subplots(2,2,figsize=(8,8))
		r_idxs=np.arange(len(ds))
		np.random.shuffle(r_idxs)
		[ plot_space(ax[i//2,i%2], ds[r_idxs[i]]) for i in np.arange(4) ]
		plt.title("Task2")
		plt.show()




