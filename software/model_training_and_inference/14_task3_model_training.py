import os
import argparse
import time
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

from models.models import (SingleSnapshotNet, SnapshotNet, Task1Net, TransformerEncOnlyModel,
			UNet,TrajectoryNet)
from utils.spf_dataset import SessionsDataset, SessionsDatasetTask2, collate_fn_transformer_filter, output_cols, input_cols

torch.set_printoptions(precision=5,sci_mode=False,linewidth=1000)


def src_pos_from_radial(inputs,outputs):
	det_pos=inputs[:,:,input_cols['det_pos']]

	theta=outputs[:,:,[cols_for_loss.index(output_cols['src_theta'][0])]]*np.pi
	dist=outputs[:,:,[cols_for_loss.index(output_cols['src_dist'][0])]]

	theta=theta.float()
	dist=dist.float()
	det_pos=det_pos.float()
	return torch.stack([torch.sin(theta),torch.cos(theta)],axis=2)[...,0]*dist+det_pos

def model_forward(d_model,data,args,train_test_label,update,plot=True):
	batch_size,time_steps,d_drone_state=data['drone_state'].shape
	_,_,n_sources,d_emitter_state=data['emitter_position_and_velocity'].shape

	d_model['optimizer'].zero_grad()
	losses={}
	preds=d_model['model'](data)#.detach().cpu()
	#for k in preds:
	#	preds[k]=preds[k].detach().cpu()
	transformer_loss=0.0
	single_snapshot_loss=0.0
	fc_loss=0.0
	if 'transformer_pred' in preds:
		#if preds['transformer_pred'].mean(axis=1).var(axis=0).mean().item()<1e-13:
		#	d_model['dead']=True
		#else:
		#	d_model['dead']=False
		labels=torch.cat([
			data['emitter_position_and_velocity'],
			data['emitters_broadcasting'],
			],dim=3)[:,1:]


		#print("TODO PLUMBING FOR B OF TRACKING TRACK")

		transformer_preds=preds['transformer_pred'][:,:-1] # trim off last time step, dont have gt for it
		transformer_pred_means=transformer_preds[...,:4] 
		transformer_pred_vars=transformer_preds[...,4:4+4] 
		transformer_pred_angles=transformer_preds[...,8:8+2]
		#for n in points
		#  for p_idx in 2: #xy output
		# 	 for output_col in 2: # xy input
		#      out[n,p_idx]+=points[n,output_col]*rot[n,output_col,p_idx]

		#for i 
		#   for k 

		#torch.einsum('ik,kj->ij', [a, b]) , matmul

		#torch.einsum('nik,nkj->nij', [a, b]) , matmul
		# need to expand b,t,2 angles into b,t,2,(2x2) 

		#point wise mul and sum
		# then matmul with predictions b,t,2,2
		
		
		breakpoint()
		transformer_loss = criterion(preds['transformer_pred'][:,:-1,:-1],labels)
		#transformer_loss = criterion(preds['transformer_pred'][:,:-1,:-1],labels)
		#pred_means=preds['transformer_pred'][:,:,
		#		d['emitter_position_and_velocity'],
		#		d['emitter_position_and_velocity']*0+1, # the variance
		#TODO add in gaussian scoring here?
		losses['transformer_loss']=transformer_loss.detach().item()
		losses['transformer_stats']=(preds['transformer_pred'][:,:-1,:-1]-labels).pow(2).mean(axis=[0,1]).detach().cpu()
		_p=preds['transformer_pred'].detach().cpu()
		assert(not preds['transformer_pred'].isnan().any())
	if False and 'single_snapshot_pred' in preds:
		single_snapshot_loss = criterion(preds['single_snapshot_pred'],_data['labels'])
		losses['single_snapshot_loss']=single_snapshot_loss.item()
		losses['single_snapshot_stats']=(preds['single_snapshot_pred']-_data['labels']).pow(2).mean(axis=[0,1]).detach().cpu()
	loss=(1.0-args.transformer_loss_balance)*transformer_loss+args.transformer_loss_balance*single_snapshot_loss+fc_loss
	if i<args.embedding_warmup:
		loss=single_snapshot_loss+fc_loss
	return loss,losses

def model_to_losses(running_loss,mean_chunk):
	if len(running_loss)==0:
		return {}
	losses={}
	for k in ['baseline','baseline_image','image_loss','transformer_loss','single_snapshot_loss','fc_loss','transformer_stats','single_snapshot_stats']:
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
	for k in ['image_loss','transformer_loss','single_snapshot_loss','fc_loss']:
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
			loss_str.append("\t\t%s\t%s" % (k,"\t".join([ "%0.4f" % v.item() for v in  losses[k][-1]])))
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
		baseline_image_loss,
		xtick_spacing,
		mean_chunk,
		output_prefix,
		fig,
		title,
		update):
	fig.clf()
	fig.suptitle(title)
	axs=fig.subplots(1,4,sharex=True)
	axs[1].sharex(axs[0])
	axs[2].sharex(axs[0])
	xs=np.arange(len(baseline_loss['baseline']))*xtick_spacing
	for i in range(3):
		axs[i].plot(xs,baseline_loss['baseline'],label='baseline')
		axs[i].set_xlabel("time")
		axs[i].set_ylabel("log loss")
	if baseline_image_loss is not None:
		axs[3].plot(xs,baseline_image_loss['baseline_image'],label='baseline image')
	axs[0].set_title("Transformer loss")
	axs[1].set_title("Single snapshot loss")
	axs[2].set_title("FC loss")
	axs[3].set_title("Image loss")
	for d_model in models:
		losses=model_to_losses(running_losses[d_model['name']],mean_chunk)
		if 'transformer_loss' in losses:
			axs[0].plot(xs,losses['transformer_loss'],label=d_model['name'])
		if 'single_snapshot_loss' in losses:
			axs[1].plot(xs,losses['single_snapshot_loss'],label=d_model['name'])
		if 'fc_loss' in losses:
			axs[2].plot(xs,losses['fc_loss'],label=d_model['name'])
		if 'image_loss' in losses:
			axs[3].plot(xs,losses['image_loss'],label=d_model['name'])
	for i in range(4):
		axs[i].legend()
	fig.tight_layout()
	fig.savefig('%sloss_%s_%d.png' % (output_prefix,title,update))
	fig.canvas.draw_idle()

if __name__=='__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, required=False, default='cpu')
	parser.add_argument('--embedding-warmup', type=int, required=False, default=0)
	parser.add_argument('--snapshots-per-sample', type=int, required=False, default=[1,4,8], nargs="+")
	parser.add_argument('--print-every', type=int, required=False, default=100)
	parser.add_argument('--lr-scheduler-every', type=int, required=False, default=256)
	parser.add_argument('--plot-every', type=int, required=False, default=1024)
	parser.add_argument('--save-every', type=int, required=False, default=1000)
	parser.add_argument('--test-mbs', type=int, required=False, default=8)
	parser.add_argument('--output-prefix', type=str, required=False, default='model_out')
	parser.add_argument('--test-fraction', type=float, required=False, default=0.2)
	parser.add_argument('--weight-decay', type=float, required=False, default=0.0)
	parser.add_argument('--transformer-loss-balance', type=float, required=False, default=0.1)
	parser.add_argument('--type', type=str, required=False, default=32)
	parser.add_argument('--seed', type=int, required=False, default=0)
	parser.add_argument('--keep-n-saves', type=int, required=False, default=2)
	parser.add_argument('--epochs', type=int, required=False, default=20000)
	parser.add_argument('--positional-encoding-len', type=int, required=False, default=0)
	parser.add_argument('--mb', type=int, required=False, default=64)
	parser.add_argument('--workers', type=int, required=False, default=4)
	parser.add_argument('--dataset', type=str, required=False, default='./sessions-default')
	parser.add_argument('--lr-image', type=float, required=False, default=0.05)
	parser.add_argument('--lr-direct', type=float, required=False, default=0.01)
	parser.add_argument('--lr-transformer', type=float, required=False, default=0.00001)
	parser.add_argument('--plot', type=bool, required=False, default=False)
	parser.add_argument('--transformer-input', type=str, required=False, default=['drone_state','embedding','single_snapshot_pred'],nargs="+")
	parser.add_argument('--transformer-dmodel', type=int, required=False, default=64)
	parser.add_argument('--clip', type=float, required=False, default=0.5)
	parser.add_argument('--losses', type=str, required=False, default="src_pos,src_theta,src_dist") #,src_theta,src_dist,det_delta,det_theta,det_space")
	args = parser.parse_args()

	dtype=torch.float32
	if args.type=='16':
		dtype=torch.float16
	
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
			collate_fn=collate_fn_transformer_filter)
	testloader = torch.utils.data.DataLoader(
			ds_test, 
			batch_size=args.mb,
			shuffle=True, 
			num_workers=args.workers,
			collate_fn=collate_fn_transformer_filter)

	print("init network")
	models=[]

	if True:
		n_layers=2
		models.append({
			'name':'TrajectoryNet',
			'model':TrajectoryNet(),
			'dead':False,
						'lr':args.lr_transformer,
		})
			

	if False:
		for n_layers in [2,4,8,16,32]:#,32,64]: #,32,64]:
			for snapshots_per_sample in args.snapshots_per_sample:
				if snapshots_per_sample==1:
					continue
				models.append( 
					{
						'name':'%d snapshots (l%d)' % (snapshots_per_sample,n_layers), 
						'model':SnapshotNet(
							snapshots_per_sample,
							n_layers=n_layers,
							d_model=args.transformer_dmodel,
							n_outputs=len(cols_for_loss),
							ssn_n_outputs=len(cols_for_loss),
							dropout=0.0,
							positional_encoding_len=args.positional_encoding_len,
							tformer_input=args.transformer_input),
						'snapshots_per_sample':snapshots_per_sample,
						'images':False,
						'lr':args.lr_transformer,
						'images':False,'fig':plt.figure(figsize=(18,4)),
						'dead':False,
					}
				)



	#move the models to the device
	for d_net in models:
		d_net['model']=d_net['model'].to(dtype).to(device)
	loss_figs={
		'train':plt.figure(figsize=(14*3,6)),
		'test':plt.figure(figsize=(14*3,6))}

	for d_model in models:
		d_model['optimizer']=optim.Adam(d_model['model'].parameters(),lr=d_model['lr'],weight_decay=args.weight_decay)
		d_model['scheduler']=optim.lr_scheduler.LinearLR(
			d_model['optimizer'], 
			start_factor=0.001, 
			end_factor=1.0, 
			total_iters=30, 
			verbose=False)

	criterion = nn.MSELoss().to(device)

	print("training loop")
	running_losses={'train':{},'test':{}}
	for k in ['train','test']:
		running_losses[k]={ d['name']:[] for d in models}
		running_losses[k]['baseline']=[]
		running_losses[k]['baseline_image']=[]

	saves=[]
	

	def prep_data(data):
		#todo add data augmentation
		#split one source to two (merge)
		#drop sources from input (birth)
		#add sources not seen (death)
		d={ k:data[k].to(dtype).to(device) for k in data}

		batch_size,time_steps,n_sources,_=d['emitter_position_and_velocity'].shape
		d['emitter_position_and_velocity']=torch.cat([
				d['emitter_position_and_velocity'],
				torch.zeros(batch_size,time_steps,n_sources,4,device=d['emitter_position_and_velocity'].device)+1, # the variance
				torch.zeros(batch_size,time_steps,n_sources,2,device=d['emitter_position_and_velocity'].device), # the angle
			],dim=3)

		for k in d:
			assert(not d[k].isnan().any())
		
		return d

	test_iterator = iter(testloader)
	for epoch in range(args.epochs): 
		for i, data in enumerate(trainloader, 0):
			#move to device, do final prep
			prepared_data=prep_data(data)
			if True: #torch.cuda.amp.autocast():
				for d_model in models:
					for p in d_net['model'].parameters():
						if p.isnan().any():
							breakpoint()
					loss,losses=model_forward(
						d_model,
						prepared_data,
						args,
						'train',
						update=i,
						plot=True)
					if i%args.lr_scheduler_every==args.lr_scheduler_every-1:
						d_model['scheduler'].step()
					loss.backward()
					running_losses['train'][d_model['name']].append(losses) 
					if args.clip>0:
						torch.nn.utils.clip_grad_norm_(d_net['model'].parameters(), args.clip) # clip gradients
					d_model['optimizer'].step()
					for p in d_net['model'].parameters():
						if p.isnan().any():
							breakpoint()
			#labels=prepared_data['labels']
			running_losses['train']['baseline'].append({'baseline':1e-5}) # {'baseline':criterion(labels*0+labels.mean(axis=[0,1],keepdim=True), labels).item() } )
		
			if i%args.print_every==args.print_every-1:
				for idx in np.arange(args.test_mbs):
					try:
						data = next(test_iterator)
					except StopIteration:
						test_iterator = iter(testloader)
						data = next(test_iterator)
					prepared_data=prep_data(data)
					with torch.no_grad():
						for d_model in models:
							loss,losses=model_forward(
								d_model,
								prepared_data,
								args,
								'test',
								update=i,
								plot=idx==0)
							running_losses['test'][d_model['name']].append(losses) 
					#labels=prepared_data['labels']
					running_losses['test']['baseline'].append({'baseline':1e-5})# {'baseline':criterion(labels*0+labels.mean(axis=[0,1],keepdim=True), labels).item() } )
				
	
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
				#loss_str="\n".join(
				#	[ "\t%s:(tr)\n%s\n\t%s:(ts)\n%s" % (d['name'],
				#		model_to_stats_str(running_losses['train'][d['name']],args.print_every),
				#		d['name'],
				#		model_to_stats_str(running_losses['test'][d['name']],args.test_mbs)
				#	) for d in models ])
				#print("\t\t\t%s" % stats_title())
				#print(loss_str)
			if i//args.print_every>2 and i % args.plot_every == args.plot_every-1:
				plot_loss(running_losses=running_losses['train'],
					baseline_loss=train_baseline_loss,
					xtick_spacing=args.print_every,
					mean_chunk=args.print_every,
					output_prefix=args.output_prefix,
					fig=loss_figs['train'],
					title='Train',update=i)
				plot_loss(running_losses=running_losses['test'],
					baseline_loss=test_baseline_loss,
					xtick_spacing=args.print_every,
					mean_chunk=args.test_mbs,
					output_prefix=args.output_prefix,
					fig=loss_figs['test'],
					title='Test',update=i)
				if args.plot:
					plt.pause(0.5)

	print('Finished Training') # but do we ever really get here?
