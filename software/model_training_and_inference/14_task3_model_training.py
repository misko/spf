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
from matplotlib.patches import Ellipse

from models.models import (SingleSnapshotNet, SnapshotNet, Task1Net, TransformerEncOnlyModel,
			UNet,TrajectoryNet)
from utils.spf_dataset import SessionsDataset, SessionsDatasetTask2, collate_fn_transformer_filter, output_cols, input_cols

torch.set_printoptions(precision=5,sci_mode=False,linewidth=1000)


def get_rot_mats(theta):
	assert(theta.dim()==2 and theta.shape[1]==1)
	s = torch.sin(theta)
	c = torch.cos(theta)
	return torch.cat([c , -s, s, c],axis=1).reshape(theta.shape[0],2,2)

def rotate_points_by_thetas(points,thetas):
	return torch.einsum('ik,ijk->ij',points,get_rot_mats(thetas))

def unpack_mean_cov_angle(x):
	return x[:,:2],x[:,2:4],x[:,[4]]


#points (n,2)
#means (n,2)
#sigmas (n,2)
#thetas (n,1)
def convert_sigmas(sigmas,min_sigma,max_sigma):
	return torch.sigmoid(sigmas)*(max_sigma-min_sigma)+min_sigma

def points_to_nll(points,means,sigmas,thetas,ellipse=False): #,min_sigma=0.01,max_sigma=0.3,ellipse=False):
	#sigmas=torch.clamp(sigmas.abs(),min=min_sigma,max=None) # TODO clamp hides the gradient?
	#sigmas=torch.sigmoid(sigmas)*(max_sigma-min_sigma)+min_sigma #.abs()+min_sigma
	if ellipse:
		p=rotate_points_by_thetas(points-means,thetas)/sigmas
	else:
		p=(points-means)/sigmas.mean(axis=1,keepdim=True)
	return p.pow(2).sum(axis=1)*0.5 + torch.log(2*torch.pi*sigmas[:,0]*sigmas[:,1])

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

	min_sigma=0.01
	max_sigma=0.3
	d_model['optimizer'].zero_grad()
	losses={}
	preds=d_model['model'](data)#.detach().cpu()

	positions=data['emitter_position_and_velocity'][...,:2].reshape(-1,2)
	velocities=data['emitter_position_and_velocity'][...,2:4].reshape(-1,2)

	pos_mean,pos_cov,pos_angle=unpack_mean_cov_angle(preds['trajectory_predictions'][:,:,:,:5].reshape(-1,5))
	vel_mean,vel_cov,vel_angle=unpack_mean_cov_angle(preds['trajectory_predictions'][:,:,:,5:].reshape(-1,5))

	nll_position_reconstruction_loss=points_to_nll(positions,
											pos_mean,
											convert_sigmas(pos_cov,min_sigma=min_sigma,max_sigma=max_sigma),
											pos_angle).mean()
	nll_velocity_reconstruction_loss=points_to_nll(velocities,
											vel_mean,
											convert_sigmas(vel_cov,min_sigma=min_sigma,max_sigma=max_sigma),
											vel_angle).mean()

	ss_mean,ss_cov,ss_angle=unpack_mean_cov_angle(preds['single_snapshot_predictions'].reshape(-1,5))
	emitting_positions=data['emitter_position_and_velocity'][data['emitters_broadcasting'][...,0].to(bool)][:,:2]

	nll_ss_position_reconstruction_loss=points_to_nll(emitting_positions,
													ss_mean,
													convert_sigmas(ss_cov,min_sigma=min_sigma,max_sigma=max_sigma),
													ss_angle).mean()
	if plot and (update%args.plot_every)==args.plot_every-1:
		t=128
		d_model['fig'].clf()
		axs=d_model['fig'].subplots(1,3,sharex=True,sharey=True)
		axs[0].set_xlim([-1,1])
		axs[0].set_ylim([-1,1])
		#axs[0].scatter(_l[0,:,src_pos_idxs[0]],_l[0,:,src_pos_idxs[1]],label='source positions',c='r')
		_emitting_positions=emitting_positions[:t]
		axs[0].scatter(data['drone_state'][0,:t,0],data['drone_state'][0,:t,1],label='detector positions',s=1)
		axs[0].scatter(_emitting_positions[:,0],_emitting_positions[:,1],label='source positions',c='r',alpha=0.3)
		axs[0].set_title("Ground truth")
		_ss_mean=ss_mean[:t].detach().numpy()
		_ss_cov=convert_sigmas(ss_cov[:t],min_sigma=min_sigma,max_sigma=max_sigma).detach().numpy()
		_ss_angle=ss_angle[:t].detach().numpy()
		axs[1].scatter(_ss_mean[:,0],_ss_mean[:,1],label='pred means',c='r',alpha=0.3)
		print("PLOT",_ss_cov[:t].mean(),_ss_cov[:t].max())
		for idx in range(t):
			ellipse = Ellipse((_ss_mean[idx,0], _ss_mean[idx,1]),
					width=_ss_cov[idx,0]*3,
					height=_ss_cov[idx,1]*3,
					facecolor='none',edgecolor='red',
					angle=-_ss_angle[idx]*(360.0/(2*torch.pi)))
			axs[1].add_patch(ellipse)

		axs[1].set_title("Single snapshot predictions")

		_pred_trajectory=preds['trajectory_predictions'].detach().numpy()
		for source_idx in range(n_sources):
			trajectory_mean,_,_=unpack_mean_cov_angle(_pred_trajectory[0,:t,source_idx,:5].reshape(-1,5))
			axs[2].scatter(trajectory_mean[:,0],trajectory_mean[:,1],label='trajectory prediction',s=20)
		#for idx in [0,1,2]:
		#	axs[idx].legend()
		#
		#	axs[2].scatter(_l[0,:,src_pos_idxs[0]],_l[0,:,src_pos_idxs[1]],label='real positions',c='b',alpha=0.1,s=7)
		#	axs[2].scatter(_p[0,:,src_pos_idxs[0]],_p[0,:,src_pos_idxs[1]],label='predicted positions',c='r',alpha=0.3,s=7)
		d_model['fig'].tight_layout()
		d_model['fig'].canvas.draw_idle()
		d_model['fig'].savefig('%s%s_%d_%s.png' % (args.output_prefix,d_model['name'],update,train_test_label))

	losses={
		'nll_position_reconstruction_loss':nll_position_reconstruction_loss.item(),
		'nll_velocity_reconstruction_loss':nll_velocity_reconstruction_loss.item(),
		'nll_ss_position_reconstruction_loss':nll_ss_position_reconstruction_loss.item()
	}
	#loss=nll_position_reconstruction_loss+nll_velocity_reconstruction_loss+nll_ss_position_reconstruction_loss
	lm=torch.tensor([0.4,0.4,0.1])
	lm/=lm.sum()
	loss=lm[0]*nll_ss_position_reconstruction_loss+lm[1]*nll_position_reconstruction_loss+lm[2]*nll_velocity_reconstruction_loss
	return loss,losses

def model_to_losses(running_loss,mean_chunk):
	if len(running_loss)==0:
		return {}
	losses={}
	for k in ['baseline','nll_position_reconstruction_loss','nll_velocity_reconstruction_loss','nll_ss_position_reconstruction_loss']:
		if k in running_loss[0]:
			if '_stats' not in k:
				#losses[k]=np.log(np.array( [ np.mean([ l[k] for l in running_loss[idx*mean_chunk:(idx+1)*mean_chunk]])  
				#	for idx in range(len(running_loss)//mean_chunk) ]))
				losses[k]=np.array( [ np.mean([ l[k] for l in running_loss[idx*mean_chunk:(idx+1)*mean_chunk]])  
					for idx in range(len(running_loss)//mean_chunk) ])
			else:
				losses[k]=[ torch.stack([ l[k] for l in running_loss[idx*mean_chunk:(idx+1)*mean_chunk] ]).mean(axis=0)
					for idx in range(len(running_loss)//mean_chunk) ]
	return losses

def model_to_loss_str(running_loss,mean_chunk):
	if len(running_loss)==0:
		return ""
	loss_str=[]
	losses=model_to_losses(running_loss,mean_chunk)
	for k in ['nll_position_reconstruction_loss','nll_velocity_reconstruction_loss','nll_ss_position_reconstruction_loss']:
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
		#axs[i].plot(xs,baseline_loss['baseline'],label='baseline')
		axs[i].set_xlabel("time")
		axs[i].set_ylabel("log loss")
	#for k in ['nll_position_reconstruction_loss','nll_velocity_reconstruction_loss','nll_ss_position_reconstruction_loss']:
	axs[0].set_title("nll_ss_position_reconstruction_loss")
	axs[1].set_title("nll_position_reconstruction_loss")
	axs[2].set_title("nll_velocity_reconstruction_loss")
	#axs[3].set_title("Image loss")
	for d_model in models:
		losses=model_to_losses(running_losses[d_model['name']],mean_chunk)
		if 'nll_ss_position_reconstruction_loss' in losses:
			axs[0].plot(xs,losses['nll_ss_position_reconstruction_loss'],label=d_model['name'])
		if 'nll_position_reconstruction_loss' in losses:
			axs[1].plot(xs,losses['nll_position_reconstruction_loss'],label=d_model['name'])
		if 'nll_velocity_reconstruction_loss' in losses:
			axs[2].plot(xs,losses['nll_velocity_reconstruction_loss'],label=d_model['name'])
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
			'fig':plt.figure(figsize=(18,4)),
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
				#torch.zeros(batch_size,time_steps,n_sources,4,device=d['emitter_position_and_velocity'].device)+1, # the variance
				#torch.zeros(batch_size,time_steps,n_sources,2,device=d['emitter_position_and_velocity'].device), # the angle
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
			if i//args.print_every>5 and i % args.plot_every == args.plot_every-1:
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
