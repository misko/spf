import os
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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
		assert(self.args.sessions==len(self.filenames)) # make sure its the right dataset
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

def plot_space(ax,session,height=256,width=256):
	ax.set_xlim([0,width])
	ax.set_ylim([0,height])

	markers=['o','v','D']
	colors=['g', 'b', 'y']
	for receiver_idx in np.arange(session['receiver_positions'].shape[1]):
		ax.scatter(session['receiver_positions'][:,receiver_idx,0],session['receiver_positions'][:,receiver_idx,1],label="Receiver %d" % receiver_idx ,facecolors='none',marker=markers[receiver_idx%len(markers)],edgecolor=colors[receiver_idx%len(colors)])
	for source_idx in np.arange(session['source_positions'].shape[1]):
		ax.scatter(session['source_positions'][:,source_idx,0],session['source_positions'][:,source_idx,1],label="Source %d" % source_idx ,facecolors='none',marker=markers[source_idx%len(markers)],edgecolor='r')
	ax.legend()
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")

if __name__=='__main__':
	#load a dataset
	ds=SessionsDataset('./sessions')

	#plot the space diagram for some samples
	fig,ax=plt.subplots(2,2,figsize=(8,8))
	r_idxs=np.arange(len(ds))
	np.random.shuffle(r_idxs)
	[ plot_space(ax[i//2,i%2], ds[r_idxs[i]]) for i in np.arange(4) ]
	plt.show()
	





