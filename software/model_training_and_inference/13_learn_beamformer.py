import os
import argparse
import time
from functools import cache
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset, random_split

from utils.image_utils import labels_to_source_images
from models.models import (SingleSnapshotNet, SnapshotNet, Task1Net, TransformerModel,
                    UNet, ComplexFFNN, HybridFFNN)
from utils.spf_dataset import SessionsDataset, SessionsDatasetTask2, collate_fn_beamformer


torch.set_printoptions(precision=5,sci_mode=False,linewidth=1000)

output_cols={ # maybe this should get moved to the dataset part...
	'src_pos':[0,1],
	'src_theta':[2],
	'src_dist':[3],
	'det_delta':[4,5],
	'det_theta':[6],
	'det_space':[7]
}

input_cols={
	'det_pos':[0,1],
	'time':[2],
	'space_delta':[3,4],
	'space_theta':[5],
	'space_dist':[6]
}

def src_pos_from_radial(inputs,outputs):
	det_pos=inputs[:,:,input_cols['det_pos']]
	theta=outputs[:,:,output_cols['src_theta']]
	dist=outputs[:,:,output_cols['src_dist']]
	return torch.stack([torch.cos(theta*np.pi),torch.sin(theta*np.pi)],axis=2)[...,0]*dist+det_pos

#def complexMSE(output, target):
#    loss = torch.mean((output - target).abs()**2)
#    return loss

def model_forward(d_model,inputs,outputs,args,beamformer):
	d_model['optimizer'].zero_grad()
	b,s,_=inputs.shape
	losses={}
	preds=d_model['model'](inputs.reshape(b*s,-1))
	loss = criterion(preds,outputs.reshape(b*s,-1))
	losses['beamformer_loss']=loss.item()
	if (i%args.print_every)==args.print_every-1:
		d_model['fig'].clf()
		axs=d_model['fig'].subplots(1,2,sharex=True,sharey=True)
		axs[0].plot(preds[0].detach().numpy(),label='prediction')
		axs[1].plot(beamformer[0,0].detach().numpy(),label='beamformer')
		x=outputs[0,0].argmax().item()
		for idx in [0,1]:
			axs[idx].axvline(x=x,c='r')
			axs[idx].legend()
		d_model['fig'].savefig('%s%s_%d.png' % (args.output_prefix,d_model['name'],i))
		d_model['fig'].canvas.draw_idle()
	d_model['dead']=np.isclose(preds.var(axis=1).mean().item(),0.0)
	return loss,losses

def model_to_losses(running_loss,mean_chunk):
	if len(running_loss)==0:
		return {}
	losses={}
	for k in ['baseline','baseline_image','beamformer_loss']:
		if k in running_loss[0]:
			if '_stats' not in k:
				losses[k]=np.log(np.array( [ np.mean([ l[k] for l in running_loss[idx*mean_chunk:(idx+1)*mean_chunk]])  
					for idx in range(len(running_loss)//mean_chunk) ]))
			else:
				losses[k]=[ torch.stack([ l[k] for l in running_loss[idx*mean_chunk:(idx+1)*mean_chunk] ]).mean(axis=0)
					for idx in range(len(running_loss)//mean_chunk) ]
	return losses

def model_to_loss_str(running_loss,mean_chunk):
	if len(running_loss)==0:
		return ""
	loss_str=[]
	losses=model_to_losses(running_loss,mean_chunk)
	for k in ['beamformer_loss']:
		if k in losses:
			loss_str.append("%s:%0.4f" % (k,losses[k][-1]))
	return ",".join(loss_str)

def stats_title():
	title_str=[]
	for col in args.losses.split(','):
		for _ in range(len(output_cols[col])):
			title_str.append(col)	
	return "\t".join(title_str)
	

def model_to_stats_str(running_loss,mean_chunk):
	if len(running_loss)==0:
		return ""
	losses=model_to_losses(running_loss,mean_chunk)
	loss_str=[]
	for k in ['transformer_stats','single_snapshot_stats']:
		if k in losses:
			loss_str.append("\t\t%s\t%s" % (k,"\t".join([ "%0.4f" % v.item() for v in  losses[k][-1][cols_for_loss]])))
	return "\n".join(loss_str)

def save(args,running_losses,models,iteration,keep_n_saves):
	fn='%ssave_%d.pkl' % (args.output_prefix,iteration)
	pickle.dump({
		'models':models,
		'args':args,
		'running_losses':running_losses},open(fn,'wb'))
	saves.append(fn)
	while len(saves)>keep_n_saves:
		fn=saves.pop(0)
		if os.path.exists(fn):
			os.remove(fn)	

def plot_loss(running_losses,
		baseline_loss,
		xtick_spacing,
		mean_chunk,
		output_prefix,
		fig,
		title):
	fig.clf()
	fig.suptitle(title)
	ax=fig.subplots(1,1,sharex=True)
	xs=np.arange(len(baseline_loss['baseline']))*xtick_spacing
	ax.plot(xs,baseline_loss['baseline'],label='baseline')
	ax.set_xlabel("time")
	ax.set_ylabel("log loss")
	ax.set_title("Loss")
	#mn=baseline_loss['baseline'].max()-baseline_loss['baseline'].std()
	#print(baseline_loss['baseline'].std())
	for d_model in models:
		losses=model_to_losses(running_losses[d_model['name']],mean_chunk)
		if 'beamformer_loss' in losses:
			ax.plot(xs[2:],losses['beamformer_loss'][2:],label=d_model['name'])
			#_mn=np.min(losses['beamformer_loss'])
			#if _mn<mn:
			#	mn=_mn
	#ax.set_ylim([baseline_loss['baseline'].max(),None]) #*0.9])
	ax.legend()
	fig.tight_layout()
	fig.savefig('%sloss_%s_%d.png' % (output_prefix,title,i))
	fig.canvas.draw_idle()

if __name__=='__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, required=False, default='cpu')
	parser.add_argument('--embedding-warmup', type=int, required=False, default=0)
	parser.add_argument('--snapshots-per-sample', type=int, required=False, default=[1,4,8], nargs="+")
	parser.add_argument('--print-every', type=int, required=False, default=100)
	parser.add_argument('--save-every', type=int, required=False, default=1000)
	parser.add_argument('--test-mbs', type=int, required=False, default=8)
	parser.add_argument('--output-prefix', type=str, required=False, default='model_out')
	parser.add_argument('--test-fraction', type=float, required=False, default=0.2)
	parser.add_argument('--seed', type=int, required=False, default=0)
	parser.add_argument('--keep-n-saves', type=int, required=False, default=2)
	parser.add_argument('--epochs', type=int, required=False, default=20000)
	parser.add_argument('--mb', type=int, required=False, default=64)
	parser.add_argument('--workers', type=int, required=False, default=4)
	parser.add_argument('--dataset', type=str, required=False, default='./sessions-default')
	parser.add_argument('--lr-image', type=float, required=False, default=0.05)
	parser.add_argument('--lr-direct', type=float, required=False, default=0.01)
	parser.add_argument('--lr-transformer', type=float, required=False, default=0.00001)
	parser.add_argument('--plot', type=bool, required=False, default=False)
	parser.add_argument('--clip', type=float, required=False, default=0.5)
	parser.add_argument('--losses', type=str, required=False, default="src_theta") #,src_theta,src_dist,det_delta,det_theta,det_space")
	args = parser.parse_args()
	
	if args.plot==False:
		import matplotlib
		matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	start_time=time.time()

	#lets see if output dir exists , if not make it
	basename=os.path.basename(args.output_prefix)
	if not os.path.exists(basename):
		os.makedirs(basename)


	cols_for_loss=[]
	for k in output_cols:
		if k in args.losses.split(','):
			cols_for_loss+=output_cols[k]

	device=torch.device(args.device)
	print("init dataset")
	ds=SessionsDatasetTask2(args.dataset,snapshots_in_sample=max(args.snapshots_per_sample))
	#ds_test=SessionsDatasetTask2(args.test_dataset,snapshots_in_sample=max(args.snapshots_per_sample))
	train_size=int(len(ds)*args.test_fraction)
	test_size=len(ds)-train_size

	#need to generate separate files for this not to train test leak
	#ds_train, ds_test = random_split(ds, [1-args.test_fraction, args.test_fraction])

	ds_train = torch.utils.data.Subset(ds, np.arange(train_size))
	ds_test = torch.utils.data.Subset(ds, np.arange(train_size, train_size + test_size))

	print("init dataloader")
	trainloader = torch.utils.data.DataLoader(
			ds_train, 
			batch_size=args.mb,
			shuffle=True, 
			num_workers=args.workers,
			collate_fn=collate_fn_beamformer)
	testloader = torch.utils.data.DataLoader(
			ds_test, 
			batch_size=args.mb,
			shuffle=True, 
			num_workers=args.workers,
			collate_fn=collate_fn_beamformer)

	print("init network")
	models=[]

	#signal_matrix ~ (snapshots_per_sample,n_antennas,samples_per_snapshot)
	
	_,n_receivers,samples_per_snapshot=ds_train[0]['signal_matrixs_at_t'].shape
	_,beam_former_bins=ds_train[0]['beam_former_outputs_at_t'].shape

	for snapshots_per_sample in [1]:
		for n_complex_layers in [2,4,8]:
			for norm in [True,False]:
				models.append(
					{
						'name':'ThetaNet(Complex%d) %s snaps:%d' % (n_complex_layers,"Norm" if norm else "",snapshots_per_sample),
						'model':ComplexFFNN(
							d_inputs=n_receivers*samples_per_snapshot+2,
							d_outputs=beam_former_bins,
							d_hidden=beam_former_bins*2,
							n_layers=n_complex_layers,norm=norm),
						'snapshots_per_sample':snapshots_per_sample,
						'images':False,
						'lr':args.lr_direct,
						'fig':plt.figure(figsize=(18,4)),
						'dead':False
					}
				)
				models.append(
					{
						'name':'ThetaNet(Hybrid%d) %s snaps:%d' % (n_complex_layers,"Norm" if norm else "",snapshots_per_sample),
						'model':HybridFFNN(
							d_inputs=n_receivers*samples_per_snapshot+2,
							d_outputs=beam_former_bins,
							n_complex_layers=n_complex_layers,
							n_real_layers=8,
							d_hidden=beam_former_bins*2,
							norm=norm),
						'snapshots_per_sample':snapshots_per_sample,
						'images':False,
						'lr':args.lr_direct,
						'fig':plt.figure(figsize=(18,4)),
						'dead':False
					}
				)


	#move the models to the device
	for d_net in models:
		d_net['model']=d_net['model'].to(device)
	loss_figs={
		'train':plt.figure(figsize=(9,6)),
		'test':plt.figure(figsize=(9,6))}

	for d_model in models:
		d_model['optimizer']=optim.Adam(d_model['model'].parameters(),lr=d_model['lr'])

	criterion = nn.MSELoss()

	print("training loop")
	running_losses={'train':{},'test':{}}
	for k in ['train','test']:
		running_losses[k]={ d['name']:[] for d in models}
		running_losses[k]['baseline']=[]
		running_losses[k]['baseline_image']=[]

	saves=[]
	

	def prep_data(data):
		data['beamformer']=data['beamformer']/data['beamformer'].sum(axis=2,keepdims=True)
		return { k:data[k].to(device) for k in data }

	test_iterator = iter(testloader)
	for epoch in range(args.epochs): 
		for i, data in enumerate(trainloader, 0):
			#move to device, do final prep
			data=prep_data(data)
				
			for d_model in models:
				loss,losses=model_forward(d_model,data['input'],data['labels'],args,data['beamformer'])
				loss.backward()
				running_losses['train'][d_model['name']].append(losses) 
				if args.clip>0:
					torch.nn.utils.clip_grad_norm_(d_net['model'].parameters(), args.clip) # clip gradients
				d_model['optimizer'].step()
			running_losses['train']['baseline'].append(
						{'baseline':criterion(data['beamformer'], data['labels']).item() } )
		
			if i%args.print_every==args.print_every-1:
				for idx in np.arange(args.test_mbs):
					try:
						data = next(test_iterator)
					except StopIteration:
						test_iterator = iter(testloader)
						data = next(test_iterator)
					data=prep_data(data)
					with torch.no_grad():
						for d_model in models:
							loss,losses=model_forward(d_model,data['input'],data['labels'],args,data['beamformer'])
							running_losses['test'][d_model['name']].append(losses) 
					running_losses['test']['baseline'].append( 
						{'baseline':criterion(data['beamformer'], data['labels']).item() } )
				
	
			if i==0 or i%args.save_every==args.save_every-1:
				save(args,running_losses,models,i,args.keep_n_saves)

			if i % args.print_every == args.print_every-1:

				train_baseline_loss=model_to_losses(running_losses['train']['baseline'],args.print_every)
				test_baseline_loss=model_to_losses(running_losses['test']['baseline'],args.test_mbs)
				print(f'[{epoch + 1}, {i + 1:5d}]')
				print(f'\tTrain: baseline: {train_baseline_loss["baseline"][-1]:.3f} , time { (time.time()-start_time)/(i+1) :.3f} / batch' )
				print(f'\tTest: baseline: {test_baseline_loss["baseline"][-1]:.3f}, time { (time.time()-start_time)/(i+1) :.3f} / batch' )
				loss_str="\t"+"\n\t".join(
					[ "%s(%s):(tr)%s,(ts)%s" % (d['name'],str(d['dead']),
						model_to_loss_str(running_losses['train'][d['name']],args.print_every),
						model_to_loss_str(running_losses['test'][d['name']],args.test_mbs)
					) for d in models ])
				print(loss_str)
				if i//args.print_every>0:
					plot_loss(running_losses=running_losses['train'],
						baseline_loss=train_baseline_loss,
						xtick_spacing=args.print_every,
						mean_chunk=args.print_every,
						output_prefix=args.output_prefix,
						fig=loss_figs['train'],
						title='Train')
					plot_loss(running_losses=running_losses['test'],
						baseline_loss=test_baseline_loss,
						xtick_spacing=args.print_every,
						mean_chunk=args.test_mbs,
						output_prefix=args.output_prefix,
						fig=loss_figs['test'],
						title='Test')
					if args.plot:
						plt.pause(0.5)

	print('Finished Training') # but do we ever really get here?
