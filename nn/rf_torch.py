import os
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision
from functools import cache
import matplotlib.pyplot as plt
import torch
import numpy as np
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
		assert(self.args.width==self.args.height)
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

def plot_space(ax,session,height=256,width=256):
	ax.set_xlim([0,width])
	ax.set_ylim([0,height])

	markers=['o','v','D']
	colors=['g', 'b', 'y']
	for receiver_idx in np.arange(session['receiver_positions_at_t'].shape[1]):
		ax.scatter(session['receiver_positions_at_t'][:,receiver_idx,0],session['receiver_positions_at_t'][:,receiver_idx,1],label="Receiver %d" % receiver_idx ,facecolors='none',marker=markers[receiver_idx%len(markers)],edgecolor=colors[receiver_idx%len(colors)])
	for source_idx in np.arange(session['source_positions_at_t'].shape[1]):
		ax.scatter(session['source_positions_at_t'][:,source_idx,0],session['source_positions_at_t'][:,source_idx,1],label="Source %d" % source_idx ,facecolors='none',marker=markers[source_idx%len(markers)],edgecolor='r')
	ax.legend()
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")

class SessionsDatasetTask1Simple(SessionsDataset):
	def __getitem__(self,idx):
		d=super().__getitem__(idx)
		#featurie a really simple way
		x=torch.Tensor(np.hstack([
			d['receiver_positions_at_t'].reshape(self.snapshots_in_sample,-1)/float(self.args.width),
			d['beam_former_outputs_at_t'].reshape(self.snapshots_in_sample,-1),
			#d['signal_matrixs'].reshape(self.snapshots_in_sample,-1)
			d['time_stamps'].reshape(self.snapshots_in_sample,-1)-d['time_stamps'][0],
			]))
		y=torch.Tensor(d['source_positions_at_t'][:,0]/float(self.args.width))
		return x,y
@cache
def get_grid(width):
	_xy=torch.arange(width)
	_x,_y=torch.meshgrid(_xy,_xy)
	return torch.cat([_y[None],_x[None]]).transpose(0,2)

def detector_positions_to_distance(detector_positions,width):
	diffs=get_grid(ds.args.width)[None,None]-detector_positions[:,:,None,None] 
	return (torch.sqrt(torch.pow(diffs, 2).sum(axis=4))/width)[:,:,None] # batch, snapshot, 1,x ,y 

def detector_positions_to_theta_grid(detector_positions,width):
	diffs=get_grid(ds.args.width)[None,None]-detector_positions[:,:,None,None] 
	return (torch.atan2(diffs[...,1],diffs[...,0]))[:,:,None] # batch, snapshot,1, x ,y `

def blur5(img):
	blur=torchvision.transforms.GaussianBlur(11, sigma=8.0)
	return blur(blur(blur(blur(blur(img)))))
def blur10(img):
	return blur5(blur5(img))
			
def labels_to_source_images(labels):
	b,s,n_sources,_=labels.shape
	label_images=torch.zeros((b,s,ds.args.width,ds.args.width))
	for b_idx in np.arange(b):
		for s_idx in np.arange(s):
			for source_idx in np.arange(n_sources):
				source_x,source_y=labels[b_idx,s_idx,source_idx]
				label_images[b_idx,s_idx,source_x,source_y]=1
	#label_images=torchvision.transforms.GaussianBlur(51, sigma=5.0)(label_images.reshape(b*s,1,ds.args.width,ds.args.width)).reshape(b,s,1,ds.args.width,ds.args.width)		
	label_images=blur10(label_images.reshape(b*s,1,ds.args.width,ds.args.width)).reshape(b,s,1,ds.args.width,ds.args.width)     
	label_images=label_images/label_images.sum(axis=[3,4],keepdims=True)
	return label_images

class SessionsDatasetTask2Simple(SessionsDataset):
	def __getitem__(self,idx):
		d=super().__getitem__(idx)
		#featurie a really simple way
		return d,d['source_positions_at_t']
			
if __name__=='__main__':
	
	#test task1
	if True:
		#load a dataset
		ds=SessionsDatasetTask1Simple('./sessions_task1')
		ds[253]
		ds=SessionsDataset('./sessions_task1')

		r_idxs=np.arange(len(ds))
		np.random.shuffle(r_idxs)
		fig=plt.figure(figsize=(8,8))
		for r_idx in r_idxs[:4]:
			fig.clf()
			axs=fig.subplots(3,3)
			x=ds[r_idx]
			theta_at_pos=detector_positions_to_theta_grid(x['detector_position_at_t'][None],ds.args.width) # batch, snapshot, 1, x ,y 
			dist_at_pos=detector_positions_to_distance(x['detector_position_at_t'][None],ds.args.width) # batch, snapshot, 1, x ,y
			theta_idxs=(((theta_at_pos+np.pi)/(2*np.pi))*257).int().float() #~ [2, 8, 1, 128, 128])	
		
			label_images=labels_to_source_images(x['source_positions_at_t'][None])
		
			breakpoint()
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
		breakpoint()
		#plot the space diagram for some samples
		fig,ax=plt.subplots(2,2,figsize=(8,8))
		[ plot_space(ax[i//2,i%2], ds[r_idxs[i]]) for i in np.arange(4) ]
		plt.title("Task1")
		plt.show()

	#test task2
	if True:
		#load a dataset
		ds=SessionsDatasetTask2Simple('./sessions_task2')
		ds[253]
		ds=SessionsDataset('./sessions_task2')

		#plot the space diagram for some samples
		fig,ax=plt.subplots(2,2,figsize=(8,8))
		r_idxs=np.arange(len(ds))
		np.random.shuffle(r_idxs)
		[ plot_space(ax[i//2,i%2], ds[r_idxs[i]]) for i in np.arange(4) ]
		plt.title("Task2")
		plt.show()




