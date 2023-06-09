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

def plot_space(ax,session):
	width=session['width_at_t'][0]	
	ax.set_xlim([0,width])
	ax.set_ylim([0,width])

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
			d['receiver_positions_at_t'].reshape(self.snapshots_in_sample,-1),
			d['beam_former_outputs_at_t'].reshape(self.snapshots_in_sample,-1),
			#d['signal_matrixs'].reshape(self.snapshots_in_sample,-1)
			d['time_stamps'].reshape(self.snapshots_in_sample,-1)-d['time_stamps'][0],
			]))
		y=torch.Tensor(d['source_positions_at_t'][:,0])
		return x,y
@cache
def get_grid(width):
	_xy=np.arange(width).astype(np.int16)
	_x,_y=np.meshgrid(_xy,_xy)
	return np.stack((_x,_y)).transpose(2,1,0)
	#return np.concatenate([_y[None],_x[None]]).transpose(0,2)

#input b,s,2   output: b,s,1,width,width
def detector_positions_to_distance(detector_positions,width):
	diffs=get_grid(width)[None,None]-detector_positions[:,:,None,None].astype(np.float32)
	return (np.sqrt(np.power(diffs, 2).sum(axis=4)))[:,:,None] # batch, snapshot, 1,x ,y 

#input b,s,2   output: b,s,1,width,width
def detector_positions_to_theta_grid(detector_positions,width):
	diffs=get_grid(width)[None,None]-detector_positions[:,:,None,None].astype(np.float32) 
	return (np.arctan2(diffs[...,1],diffs[...,0]))[:,:,None] # batch, snapshot,1, x ,y `
def blur2(img):
	blur=torchvision.transforms.GaussianBlur(11, sigma=8.0)
	return blur(blur(img))

def blur5(img):
	blur=torchvision.transforms.GaussianBlur(11, sigma=8.0)
	return blur(blur(blur(blur(blur(img)))))
def blur10(img):
	return blur5(blur5(img))
			
def labels_to_source_images(labels,width):
	b,s,n_sources,_=labels.shape
	offset=0 #50 # takes too much compute!
	label_images=torch.zeros((
			b,s,
			width+2*offset,
			width+2*offset))
	for b_idx in np.arange(b):
		for s_idx in np.arange(s):
			for source_idx in np.arange(n_sources):
				source_x,source_y=labels[b_idx,s_idx,source_idx]
				label_images[b_idx,s_idx,int(source_x)+offset,int(source_y)+offset]=1
	
	label_images=blur5(
		label_images.reshape(
			b*s,1,
			width+2*offset,
			width+2*offset)).reshape(
				b,s,1,
				width+2*offset,
				width+2*offset)    
	#label_images=label_images[...,offset:-offset,offset:-offset] # trim the rest
	assert(label_images.shape[3]==width)
	label_images=label_images/label_images.sum(axis=[3,4],keepdims=True)
	return label_images

def radio_to_image(beam_former_outputs_at_t,theta_at_pos):
	theta_idxs=(((theta_at_pos+np.pi)/(2*np.pi))*(beam_former_outputs_at_t.shape[-1]-1)).round().astype(int)
	b,s,_,width,_=theta_at_pos.shape
	beam_former_outputs_at_t=beam_former_outputs_at_t#[:]/beam_former_outputs_at_t.sum(axis=2,keepdims=True)

	outputs=[]
	for b_idx in np.arange(b):
		for s_idx in np.arange(s):
			outputs.append(
				np.take(
					beam_former_outputs_at_t[b_idx,s_idx],
					theta_idxs[b_idx,s_idx].reshape(-1)).reshape(theta_idxs[b_idx,s_idx].shape))
	return np.stack(outputs).reshape(b,s,1,width,width)

class SessionsDatasetTask2Simple(SessionsDataset):
	def __getitem__(self,idx):
		d=super().__getitem__(idx)
		d['source_positions_at_t']=d['source_positions_at_t'] #/self.args.width
		d['detector_position_at_t']=d['detector_position_at_t'] #/self.args.width

		d['source_image_at_t']=labels_to_source_images(d['source_positions_at_t'][None],self.args.width)[0]
		d['detector_theta_image_at_t']=detector_positions_to_theta_grid(d['detector_position_at_t'][None],self.args.width)[0]
		d['radio_image_at_t']=radio_to_image(d['beam_former_outputs_at_t'][None],d['detector_theta_image_at_t'][None])[0]
		if True:
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
		ds=SessionsDataset(sessions_dir)

		#plot the space diagram for some samples
		fig,ax=plt.subplots(2,2,figsize=(8,8))
		r_idxs=np.arange(len(ds))
		np.random.shuffle(r_idxs)
		[ plot_space(ax[i//2,i%2], ds[r_idxs[i]]) for i in np.arange(4) ]
		plt.title("Task2")
		plt.show()




