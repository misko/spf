from rf_torch import SessionsDataset,SessionsDatasetTask2Simple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tformer import Transformer, TransformerModel
import torchvision


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
	print("init dataset")
	snapshots_per_sample=16
	ds=SessionsDatasetTask2Simple('./sessions_task1',snapshots_in_sample=snapshots_per_sample)
	
	print("init dataloader")
	trainloader = torch.utils.data.DataLoader(
			ds, 
			batch_size=128,
			shuffle=True, 
			num_workers=4)
	print("init network")

	n_heads=4
	dim_model=512
	layers=6
	d_radio=261
	net_factory = lambda :  TransformerModel(d_radio=d_radio,d_model=dim_model,nhead=n_heads,d_hid=1024,nlayers=layers,dropout=0.0)	
	nets=(
		#{'name':'one snapshot','net':Net(1,ds.args.width), 'snapshots_per_sample':1},
		{'name':'%d snapshots' % 1, 
		'net':net_factory(),
		 'snapshots_per_sample':1},
		{'name':'%d snapshots' % (snapshots_per_sample//2), 
		'net':net_factory(),
		 'snapshots_per_sample':snapshots_per_sample//2},
		{'name':'%d snapshots' % snapshots_per_sample, 
		'net':net_factory(),
		 'snapshots_per_sample':snapshots_per_sample},
	)
	#nets=nets[-1:]

	for d_net in nets:
		d_net['optimizer']=optim.Adam(d_net['net'].parameters(),lr=0.00001)
	criterion = nn.MSELoss()

	print("training loop")
	print_every=5
	losses_to_plot={ d_net['name']:[] for d_net in nets }
	losses_to_plot['baseline']=[]

	for epoch in range(200):  # loop over the dataset multiple times
		running_losses={ d['name']:0.0 for d in nets }
		baseline_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			for _ in np.arange(1): # for debugging
				inputs, labels = data
				b,s,n_sources,_=labels.shape
				source_positions=inputs['source_positions_at_t'][torch.where(inputs['broadcasting_positions_at_t']==1)[:-1]].reshape(b,s,2).float()/ds.args.width
				times=inputs['time_stamps']/(0.00001+inputs['time_stamps'].max(axis=2,keepdim=True)[0]) #(ds.args.time_interval*ds.args.time_steps)
				radio_inputs=torch.cat(
					[
						inputs['beam_former_outputs_at_t'].max(axis=2,keepdim=True)[0],
						inputs['beam_former_outputs_at_t']/inputs['beam_former_outputs_at_t'].max(axis=2,keepdim=True)[0],
						times-times.max(axis=2,keepdim=True)[0],
						inputs['detector_position_at_t']/ds.args.width,
					],
					dim=2
				).float()
				for d_net in nets:
					d_net['optimizer'].zero_grad()

					_radio_inputs=torch.cat([
						radio_inputs[:,:d_net['snapshots_per_sample']],
						],dim=2)
					_source_positions=source_positions[:,:d_net['snapshots_per_sample']]

				
					preds=d_net['net'](
						_radio_inputs) # 8,32,2
					loss = criterion(
						preds,
						_source_positions)
					if i%100==0:
						print(preds[0])
						print(_source_positions[0])
					loss.backward()
					running_losses[d_net['name']] += np.log(loss.item())
					d_net['optimizer'].step()
				#baseline_loss += ds.args.width*np.sqrt(criterion(torch.zeros(label_images.shape)+label_images.mean(), label_images).item())
				baseline_loss += np.log(criterion(source_positions*0+source_positions.mean(), source_positions).item())


			if i % print_every == print_every-1:
				loss_str=",".join([ "%s: %0.3f" % (d_net['name'],running_losses[d_net['name']]/print_every) for d_net in nets ])
				print(f'[{epoch + 1}, {i + 1:5d}] err_in_meters {loss_str} baseline: {baseline_loss / print_every:.3f}')
				if i//print_every>2:
					for d_net in nets:
						losses_to_plot[d_net['name']].append(running_losses[d_net['name']]/print_every)
					losses_to_plot['baseline'].append(baseline_loss / print_every)
					plt.clf()
					for d_net in nets:
						plt.plot(losses_to_plot[d_net['name']],label=d_net['name'])
					plt.plot(losses_to_plot['baseline'],label='baseline')
					plt.xlabel("time")
					plt.ylabel("error in m")
					plt.legend()
					plt.pause(0.001)
				running_losses={ d['name']:0.0 for d in nets }
				baseline_loss = 0.0
		
	print('Finished Training')
