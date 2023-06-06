from rf_torch import SessionsDataset,SessionsDatasetTask2Simple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import torchvision

class Net(nn.Module):
	def __init__(self,n_snapshots,width,beam_former_dim=257,internal_dim=5):
		super().__init__()
		#self.feature_dim=beam_former_dim+1+1
		self.feature_dim=1#+1#+1
		self.internal_dim=internal_dim
		self.width=width
		self.n_snapshots=n_snapshots

		self.conv1 = m = nn.Conv2d(self.feature_dim, 32, 1)
		self.conv2 = m = nn.Conv2d(32, 32, 7, padding=3)#3,padding=1)
		self.conv3 = m = nn.Conv2d(32, 32, 7, padding=3)#3,padding=1)
		self.conv4 = m = nn.Conv2d(32, 32, 7, padding=3)#3,padding=1)
		self.conv5 = m = nn.Conv2d(32, 32, 7, padding=3)
		self.conv6 = m = nn.Conv2d(32, self.internal_dim, 1)
		self.conv7 = m = nn.Conv2d(self.internal_dim,1,1) #,5,stride=1,padding=2)
		self.bn1 = nn.BatchNorm2d(32)#, affine=False)
		self.bn2 = nn.BatchNorm2d(32)#, affine=False)
		self.bn3 = nn.BatchNorm2d(32)#, affine=False)
		self.bn4 = nn.BatchNorm2d(32)#, affine=False)
		self.bn5 = nn.BatchNorm2d(32)#, affine=False)
		self.bn6 = nn.BatchNorm2d(self.internal_dim)#, affine=False)


		self.join_conv1 = nn.Conv2d(self.n_snapshots+3, 32, 7,padding=3)
		self.join_conv2 = nn.Conv2d(32, 32, 7,padding=3)
		self.join_conv3 = nn.Conv2d(32, 32, 7,padding=3)
		self.join_conv4 = nn.Conv2d(32, 32, 7,padding=3)
		self.join_conv5 = nn.Conv2d(32, 1, 3,padding=1) 
		self.join_bn1 = nn.BatchNorm2d(32)#, affine=False)
		self.join_bn2 = nn.BatchNorm2d(32)#, affine=False)
		self.join_bn3 = nn.BatchNorm2d(32)#, affine=False)
		self.join_bn4 = nn.BatchNorm2d(32)#, affine=False)
		self.join_bn5 = nn.BatchNorm2d(1)#, affine=False)
		#self.join_bn3 = nn.BatchNorm2d(32)#, affine=False)


	def forward(self, x):
		theta_at_pos=detector_positions_to_theta_grid(x['detector_position_at_t'],self.width) # batch, snapshot, 1, x ,y 
		dist_at_pos=detector_positions_to_distance(x['detector_position_at_t'],ds.args.width) # batch, snapshot, 1, x ,y
		theta_idxs=(((theta_at_pos+np.pi)/(2*np.pi))*257).int().float() #~ [2, 8, 1, 128, 128])	
		#print(theta_idxs.shape,"SHA")	
		#img=theta_idxs[0,0,0]
		beam_former_outputs_at_t=x['beam_former_outputs_at_t'][:]/x['beam_former_outputs_at_t'].sum(axis=[2],keepdim=True)

		b,s,_,_x,_y=dist_at_pos.shape
		for b_idx in np.arange(b):
			for s_idx in np.arange(s):
				m=theta_idxs[b_idx,s_idx]
				for idx in np.arange(257):
					m[m==idx]=beam_former_outputs_at_t[b_idx,s_idx,idx] #).abs().log()
		#theta_idxs/=(dist_at_pos+0.0000000001)
		#img[img==idx]=(x['beam_former_outputs_at_t'][0,0,idx]).abs()
		
		#source_pos=x['source_positions_at_t'][0,0,0]
		#img=theta_idxs[0,0,0]
		#img[source_pos[0],source_pos[1]]=img.max()*1.2
		#plt.imshow(img)
		#plt.pause(0.1)	
		#breakpoint()
		#_theta_at_pos=theta_at_pos.reshape(-1,1) # b*s*x*y,1
		#_dist_at_pos=dist_at_pos.reshape(-1,1) # b*s*x*y,1

		#beam_former_feature_size=x['beam_former_outputs_at_t'].shape[2]
		#_beam_former_output=x['beam_former_outputs_at_t'][:,:,:,None,None].expand((b,s,beam_former_feature_size,_x,_y)) #.reshape(b*s*_x*_y,beam_former_feature_size) #
		#base_input=torch.Tensor(np.concatenate(
		#	[
		#		#theta_at_pos,
		#		theta_idxs,
		#		dist_at_pos,
		#		_beam_former_output],axis=2)).reshape(b*s,self.feature_dim,_x,_y) #.reshape(-1,self.feature_dim)

		#torch.Size([16, 8, 1, 128, 128]) 
		base_input=torch.Tensor(np.concatenate(
			[
				theta_idxs,
				#dist_at_pos,
				#theta_idxs/(dist_at_pos+1e-10),
			], axis=2
			).reshape(b*s,self.feature_dim,_x,_y))	
		x=base_input
		x = F.relu(self.bn1(self.conv1(base_input)))
		#x = F.relu(self.bn2(self.conv2(x)))
		#x = F.relu(self.bn3(self.conv3(x)))
		#x = F.relu(self.bn4(self.conv4(x)))
		#x = F.relu(self.bn5(self.conv5(x)))
		x = F.relu(self.bn6(self.conv6(x)))
		x = self.conv7(x)
		b_s_detections = x.reshape(b,s,1,_x,_y)
		

		#join the levels
		x = x.reshape(b,s,_x,_y)
		x = torch.cat(
			[
				x,
				x.max(axis=1,keepdim=True)[0],
				x.min(axis=1,keepdim=True)[0],
				x.mean(axis=1,keepdim=True)
			], dim=1)
		#breakpoint()
		x = F.relu(self.join_bn1(self.join_conv1(x)))
		x = F.relu(self.join_bn2(self.join_conv2(x)))
		x = F.relu(self.join_bn3(self.join_conv3(x)))
		x = F.relu(self.join_bn4(self.join_conv4(x)))
		x = self.join_conv5(x)

		b_s_detections=b_s_detections/b_s_detections.sum(axis=[3,4],keepdims=True)
		x=x/x.sum(axis=[2,3],keepdims=True)
		#breakpoint()
		return b_s_detections,x

from functools import cache

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

if __name__=='__main__': 
	blur=torchvision.transforms.GaussianBlur(11, sigma=8.0)
	def blur5(img):
		return blur(blur(blur(blur(blur(img)))))
	def blur10(img):
		return blur5(blur5(img))
	print("init dataset")
	snapshots_per_sample=8
	ds=SessionsDatasetTask2Simple('./sessions_task2',snapshots_in_sample=snapshots_per_sample)
	
	print("init dataloader")
	trainloader = torch.utils.data.DataLoader(
			ds, 
			batch_size=32,
			shuffle=True, 
			num_workers=4)
	print("init network")
	
	nets=(
		#{'name':'one snapshot','net':Net(1,ds.args.width), 'snapshots_per_sample':1},
		{'name':'%d snapshots' % snapshots_per_sample, 'net':Net(snapshots_per_sample,ds.args.width), 'snapshots_per_sample':snapshots_per_sample},
	)

	for d_net in nets:
		d_net['optimizer']=optim.Adam(d_net['net'].parameters(),lr=0.001)
	criterion = nn.MSELoss()

	print("training loop")
	print_every=5
	losses_to_plot={ d_net['name']:[] for d_net in nets }
	losses_to_plot['baseline']=[]
	fig=plt.figure(figsize=(8,6))
	axs=fig.subplots(2,2)

	axs[0,0].set_title("Source (snap=0)")
	axs[0,1].set_title("Pred (snap=0 | snap=0)")
	axs[1,0].set_title("Pred (snap=Last | snap=Last)")
	axs[1,1].set_title("Pred (snap=Last | snap=All)")
	for epoch in range(200):  # loop over the dataset multiple times
		running_losses={ d['name']:0.0 for d in nets }
		baseline_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			for _ in np.arange(1): # for debugging
				inputs, labels = data
				#labels ~ b,s,sources,2
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
				for d_net in nets:
					d_net['optimizer'].zero_grad()
					b_s_dets,final_dets = d_net['net']({ k:v[:,:d_net['snapshots_per_sample']] for k,v in inputs.items()}) #[:,:d_net['snapshots_per_sample']])	
					
					single_snapshot_loss = criterion(b_s_dets, label_images)
					full_snapshot_loss = criterion(final_dets, label_images[:,-1])
					loss=single_snapshot_loss+full_snapshot_loss
				
						
					#axs[0,0].imshow(label_images.detach().numpy()[0,0,0]) # [b_idx,s_idx,channel]
					#plt.show()
					#breakpoint()
					print(b_s_dets.mean(),label_images.mean(),"MEANS")
					loss.backward()
					print(loss.item())
					running_losses[d_net['name']] += ds.args.width*np.sqrt(loss.item())
					d_net['optimizer'].step()
					#breakpoint()
					axs[0,0].imshow(label_images.detach().numpy()[0,0,0]) # [b_idx,s_idx,channel]
					axs[0,1].imshow(b_s_dets.detach().numpy()[0,0,0]) # [b_idx,s_idx,channel]
					axs[1,0].imshow(b_s_dets.detach().numpy()[0,-1,0]) # [b_idx,s_idx,channel]
					axs[1,1].imshow(final_dets.detach().numpy()[0,0]) # [b_idx,channel]
					#plt.imshow(label_images.detach().numpy()[0][0][0])
					#plt.pause(1)
					#plt.imshow(outputs.detach().numpy()[0][0][0])
					plt.pause(0.1)
				#baseline_loss += ds.args.width*np.sqrt(criterion(torch.zeros(label_images.shape)+label_images.mean(), label_images).item())
				baseline_loss += ds.args.width*np.sqrt(criterion(torch.zeros(label_images.shape), label_images).item())


			if i % print_every == print_every-1:
				for d_net in nets:
					losses_to_plot[d_net['name']].append(running_losses[d_net['name']]/print_every)
				losses_to_plot['baseline'].append(baseline_loss / print_every)
				loss_str=",".join([ "%s: %0.3f" % (d_net['name'],running_losses[d_net['name']]/print_every) for d_net in nets ])
				print(f'[{epoch + 1}, {i + 1:5d}] err_in_meters {loss_str} baseline: {baseline_loss / print_every:.3f}')
				running_losses={ d['name']:0.0 for d in nets }
				baseline_loss = 0.0
				if False:
					plt.clf()
					for d_net in nets:
						plt.plot(losses_to_plot[d_net['name']],label=d_net['name'])
					plt.plot(losses_to_plot['baseline'],label='baseline')
					plt.xlabel("time")
					plt.ylabel("error in m")
					plt.legend()
					plt.pause(0.001)
		
	print('Finished Training')
