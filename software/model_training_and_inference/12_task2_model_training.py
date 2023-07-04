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

from utils.image_utils import labels_to_source_images
from models.models import (SingleSnapshotNet, SnapshotNet, Task1Net, TransformerModel,
		    UNet)
from utils.spf_dataset import SessionsDataset, SessionsDatasetTask2, collate_fn


torch.set_printoptions(precision=5,sci_mode=False,linewidth=1000)

output_cols={ # maybe this should get moved to the dataset part...
	'src_pos':[0,1],
	'src_theta':[2],
	'src_dist':[3],
	'det_delta':[4,5],
	'det_theta':[6],
	'det_space':[7],
}

input_cols={
	'det_pos':[0,1],
	'time':[2],
	'space_delta':[3,4],
	'space_theta':[5],
	'space_dist':[6],
	'det_theta2':[7],
}

def src_pos_from_radial(inputs,outputs):
	det_pos=inputs[:,:,input_cols['det_pos']]

	theta=outputs[:,:,[cols_for_loss.index(output_cols['src_theta'][0])]]*np.pi
	dist=outputs[:,:,[cols_for_loss.index(output_cols['src_dist'][0])]]

	theta=theta.float()
	dist=dist.float()
	det_pos=det_pos.float()
	return torch.stack([torch.cos(theta),torch.sin(theta)],axis=2)[...,0]*dist+det_pos

def model_forward(d_model,radio_inputs,radio_images,labels,label_images,args,train_test_label,update,plot=True):
	if radio_inputs.isnan().any():
		breakpoint()
	d_model['optimizer'].zero_grad()
	use_all_data=True
	if use_all_data:
		b,s,_=radio_inputs.shape
		model_s=d_model['snapshots_per_sample']
		#assert(s%model_s==0)
		_radio_inputs=radio_inputs.reshape(b*s//model_s,model_s,radio_inputs.shape[-1])
		_,_,l=labels.shape
		_labels=labels.reshape(b*s//model_s,model_s,l)
		if radio_images is not None:
			_,_,_,w,h=radio_images.shape	
			_radio_images=radio_images.reshape(b*s//model_s,model_s,w,h)
			#_,_,_,_,_=label_images.shape	
			_label_images=label_images.reshape(b*s//model_s,model_s,1,w,h)[:,model_s-1]
	else:
		_radio_inputs=radio_inputs[:,:d_model['snapshots_per_sample']]
		_radio_images=radio_images[:,:d_model['snapshots_per_sample'],0] # reshape to b,s,w,h
		_labels=labels[:,:d_model['snapshots_per_sample']] # how does network learn the order?!
		_label_images=label_images[:,d_model['snapshots_per_sample']-1] # reshape to b,1,w,h
	losses={}
	if d_model['images']==False:
		preds=d_model['model'](_radio_inputs)#.detach().cpu()
		assert(not _radio_inputs.isnan().any())
		#for k in preds:
		#	preds[k]=preds[k].detach().cpu()
		transformer_loss=0.0
		single_snapshot_loss=0.0
		fc_loss=0.0
		if 'transformer_pred' in preds:
			if preds['transformer_pred'].mean(axis=1).var(axis=0).mean().item()<1e-13:
				d_model['dead']=True
			else:
				d_model['dead']=False
			transformer_loss = criterion(preds['transformer_pred'],_labels)
			losses['transformer_loss']=transformer_loss.detach().item()
			losses['transformer_stats']=(preds['transformer_pred']-_labels).pow(2).mean(axis=[0,1]).detach().cpu()
			_p=preds['transformer_pred'].detach().cpu()
			assert(not preds['transformer_pred'].isnan().any())
		if 'single_snapshot_pred' in preds:
			single_snapshot_loss = criterion(preds['single_snapshot_pred'],_labels)
			losses['single_snapshot_loss']=single_snapshot_loss.item()
			losses['single_snapshot_stats']=(preds['single_snapshot_pred']-_labels).pow(2).mean(axis=[0,1]).detach().cpu()
		elif 'fc_pred' in preds:
			fc_loss = criterion(preds['fc_pred'],_labels[:,[-1]])
			losses['fc_loss']=fc_loss.item()
			_p=preds['fc_pred'].detach().cpu()
			#losses['fc_stats']=(preds['fc_pred']-_labels).pow(2).mean(axis=[0,1]).cpu()
		if plot and (update%args.plot_every)==args.plot_every-1:
			d_model['fig'].clf()
			_ri=_radio_inputs.detach().cpu()
			_l=_labels.detach().cpu()
			axs=d_model['fig'].subplots(1,4,sharex=True,sharey=True)
			d_model['fig'].suptitle(d_model['name'])
			axs[0].scatter(_ri[0,:,input_cols['det_pos'][0]],_ri[0,:,input_cols['det_pos'][1]],label='detector positions',s=1)
			if 'src_pos' in args.losses:
				src_pos_idxs=[]
				for idx in range(2):
					src_pos_idxs.append(cols_for_loss.index(output_cols['src_pos'][idx]))	
				axs[1].scatter(_l[0,:,src_pos_idxs[0]],_l[0,:,src_pos_idxs[1]],label='source positions',c='r')

				axs[2].scatter(_l[0,:,src_pos_idxs[0]],_l[0,:,src_pos_idxs[1]],label='real positions',c='b',alpha=0.1,s=7)
				axs[2].scatter(_p[0,:,src_pos_idxs[0]],_p[0,:,src_pos_idxs[1]],label='predicted positions',c='r',alpha=0.3,s=7)
			axs[2].legend()
			pos_from_preds=src_pos_from_radial(_ri,_l)
			axs[3].scatter(pos_from_preds[0,:,0],pos_from_preds[0,:,1],label='real radial positions',c='b',alpha=0.1,s=7)
			pos_from_preds=src_pos_from_radial(_ri,_p)
			axs[3].scatter(pos_from_preds[0,:,0],pos_from_preds[0,:,1],label='predicted radial positions',c='r',alpha=0.3,s=7)

			axs[3].legend()
			
			for idx in [0,1,2]:
				axs[idx].legend()
				axs[idx].set_xlim([-1,1])
				axs[idx].set_ylim([-1,1])
			d_model['fig'].savefig('%s%s_%d_%s.png' % (args.output_prefix,d_model['name'],update,train_test_label))
			d_model['fig'].canvas.draw_idle()
		loss=transformer_loss+single_snapshot_loss+fc_loss
		if i<args.embedding_warmup:
			loss=single_snapshot_loss+fc_loss
	else:
		if d_model['normalize_input']:
			_radio_images=_radio_images/_radio_images.sum(axis=[2,3],keepdim=True)
		preds=d_model['model'](_radio_images)
		loss=criterion(preds['image_preds'],_label_images)
		losses['image_loss']=loss.item()
		if plot and (update%args.plot_every)==args.plot_every-1:
			d_model['fig'].clf()
			axs=d_model['fig'].subplots(1,3,sharex=True)
			axs[0].set_title('Target label')
			axs[1].set_title('Predictions')
			axs[2].set_title('Mean of input channels')
			d_model['fig'].suptitle("%s iteration: %d" % (d_model['name'],i))
			#axs=d_model['fig'].subplots(1,3,sharex=True)
			axs[0].imshow(_label_images[0,0].cpu())
			axs[1].imshow(preds['image_preds'][0,0].detach().cpu())
			axs[2].imshow(_radio_images[0].mean(axis=0).cpu())
			d_model['fig'].savefig('%s%s_%d_%s.png' % (args.output_prefix,d_model['name'],update,train_test_label))
			d_model['fig'].canvas.draw_idle()
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
	parser.add_argument('--type', type=str, required=False, default=32)
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
			collate_fn=collate_fn)
	testloader = torch.utils.data.DataLoader(
			ds_test, 
			batch_size=args.mb,
			shuffle=True, 
			num_workers=args.workers,
			collate_fn=collate_fn)

	print("init network")
	models=[]
	if True:
		for n_layers in [2,4,8,16,32]: #,32,64]:
			for snapshots_per_sample in args.snapshots_per_sample:
				models.append( 
					{
						'name':'%d snapshots (l%d)' % (snapshots_per_sample,n_layers), 
						'model':SnapshotNet(
							snapshots_per_sample,
							n_layers=n_layers,
							d_model=128, #128+256,
							n_outputs=len(cols_for_loss),
							ssn_n_outputs=len(cols_for_loss),
							dropout=0.0,
                                                        tformer_input=['embedding']),
						'snapshots_per_sample':snapshots_per_sample,
						'images':False,
						'lr':args.lr_transformer,
						'images':False,'fig':plt.figure(figsize=(18,4)),
						'dead':False,
					}
				)
	if False:
		for snapshots_per_sample in args.snapshots_per_sample:
			models.append(
				{
					'name':'task1net%d' % snapshots_per_sample,
					'model':Task1Net(266*snapshots_per_sample,n_outputs=len(cols_for_loss)),
					'snapshots_per_sample':snapshots_per_sample,
					'images':False,'fig':plt.figure(figsize=(18,4)),
					'lr':args.lr_direct,
					'dead':False,
				}
			)
	if True:
		for snapshots_per_sample in args.snapshots_per_sample:
			models.append(
				{
					'name':'FCNet %d' % snapshots_per_sample,
					'model':SingleSnapshotNet(d_radio_feature=266,
						d_hid=64,
						d_embed=64,
						n_layers=4,
						n_outputs=len(cols_for_loss),
						dropout=0.0,
					snapshots_per_sample=snapshots_per_sample),
					'snapshots_per_sample':snapshots_per_sample,
					'images':False,'fig':plt.figure(figsize=(18,4)),
					'lr':args.lr_direct,
					'dead':False
				}
			)

	if False:
		for snapshots_per_sample in args.snapshots_per_sample: 
			models.append(
				{
					'name':'Unet %d' % snapshots_per_sample,
					'model':UNet(in_channels=snapshots_per_sample,out_channels=1,width=128),
					'snapshots_per_sample':snapshots_per_sample,
					'images':True,'fig':plt.figure(figsize=(14,4)),
					'normalize_input':True,
					'lr':args.lr_image,
					'dead':False
				}
			 )

	using_images=False
	for d_model in models:
		if d_model['images']:
			using_images=True


	#move the models to the device
	for d_net in models:
		d_net['model']=d_net['model'].to(dtype).to(device)
	loss_figs={
		'train':plt.figure(figsize=(14*3,6)),
		'test':plt.figure(figsize=(14*3,6))}

	for d_model in models:
		d_model['optimizer']=optim.Adam(d_model['model'].parameters(),lr=d_model['lr'])
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
		radio_inputs, radio_images, labels, label_images = data
		labels=labels[:,:,:]

		#direct data
		radio_inputs=radio_inputs.to(dtype).to(device)
		labels=labels[...,cols_for_loss].to(dtype).to(device)
		
		assert(not radio_inputs.isnan().any())
		#image data
		if using_images:
			radio_images=radio_images.to(device)
			label_images=label_images.to(device)
			return radio_inputs,labels,radio_images,label_images
		
		return radio_inputs,labels,None,None

	test_iterator = iter(testloader)
	for epoch in range(args.epochs): 
		for i, data in enumerate(trainloader, 0):
			#move to device, do final prep
			radio_inputs,labels,radio_images,label_images=prep_data(data)
			with torch.cuda.amp.autocast():
				for d_model in models:
					for p in d_net['model'].parameters():
						if p.isnan().any():
							breakpoint()
					loss,losses=model_forward(
						d_model,
						radio_inputs,
						radio_images,
						labels,
						label_images,
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
			running_losses['train']['baseline'].append( {'baseline':criterion(labels*0+labels.mean(axis=[0,1],keepdim=True), labels).item() } )
			if using_images:
				running_losses['train']['baseline_image'].append( {'baseline_image':criterion(label_images*0+label_images.mean(), label_images).item() } )
		
			if i%args.print_every==args.print_every-1:
				for idx in np.arange(args.test_mbs):
					try:
						data = next(test_iterator)
					except StopIteration:
						test_iterator = iter(testloader)
						data = next(test_iterator)
					radio_inputs,labels,radio_images,label_images=prep_data(data)
					with torch.no_grad():
						for d_model in models:
							loss,losses=model_forward(
								d_model,
								radio_inputs,
								radio_images,
								labels,
								label_images,
								args,
								'test',
								update=i,
								plot=idx==0)
							running_losses['test'][d_model['name']].append(losses) 
					running_losses['test']['baseline'].append( {'baseline':criterion(labels*0+labels.mean(axis=[0,1],keepdim=True), labels).item() } )
					if using_images:
						running_losses['test']['baseline_image'].append( {'baseline_image':criterion(label_images*0+label_images.mean(), label_images).item() } )
				
	
			if i==0 or i%args.save_every==args.save_every-1:
				save(args,running_losses,models,i,args.keep_n_saves)

			if i % args.print_every == args.print_every-1:

				train_baseline_loss=model_to_losses(running_losses['train']['baseline'],args.print_every)
				test_baseline_loss=model_to_losses(running_losses['test']['baseline'],args.test_mbs)
				train_baseline_image_loss=None
				test_baseline_image_loss=None
				print(f'[{epoch + 1}, {i + 1:5d}]')
				if using_images:
					train_baseline_image_loss=model_to_losses(running_losses['train']['baseline_image'],args.print_every)
					test_baseline_image_loss=model_to_losses(running_losses['test']['baseline_image'],args.test_mbs)
					print(f'\tTrain: baseline: {train_baseline_loss["baseline"][-1]:.3f}, baseline_image: {train_baseline_image_loss["baseline_image"][-1]:.3f} , time { (time.time()-start_time)/(i+1) :.3f} / batch' )
					print(f'\tTest: baseline: {test_baseline_loss["baseline"][-1]:.3f}, baseline_image: {test_baseline_image_loss["baseline_image"][-1]:.3f} , time { (time.time()-start_time)/(i+1) :.3f} / batch' )
				else:
					print(f'\tTrain: baseline: {train_baseline_loss["baseline"][-1]:.3f} , time { (time.time()-start_time)/(i+1) :.3f} / batch' )
					print(f'\tTest: baseline: {test_baseline_loss["baseline"][-1]:.3f}, time { (time.time()-start_time)/(i+1) :.3f} / batch' )
				loss_str="\t"+"\n\t".join(
					[ "%s(%s):(tr)%s,(ts)%s" % (d['name'],str(d['dead']),
						model_to_loss_str(running_losses['train'][d['name']],args.print_every),
						model_to_loss_str(running_losses['test'][d['name']],args.test_mbs)
					) for d in models ])
				print(loss_str)
				loss_str="\n".join(
					[ "\t%s:(tr)\n%s\n\t%s:(ts)\n%s" % (d['name'],
						model_to_stats_str(running_losses['train'][d['name']],args.print_every),
						d['name'],
						model_to_stats_str(running_losses['test'][d['name']],args.test_mbs)
					) for d in models ])
				print("\t\t\t%s" % stats_title())
				print(loss_str)
			if i//args.print_every>2 and i % args.plot_every == args.plot_every-1:
				plot_loss(running_losses=running_losses['train'],
					baseline_loss=train_baseline_loss,
					baseline_image_loss=train_baseline_image_loss,
					xtick_spacing=args.print_every,
					mean_chunk=args.print_every,
					output_prefix=args.output_prefix,
					fig=loss_figs['train'],
					title='Train',update=i)
				plot_loss(running_losses=running_losses['test'],
					baseline_loss=test_baseline_loss,
					baseline_image_loss=test_baseline_image_loss,
					xtick_spacing=args.print_every,
					mean_chunk=args.test_mbs,
					output_prefix=args.output_prefix,
					fig=loss_figs['test'],
					title='Test',update=i)
				if args.plot:
					plt.pause(0.5)

	print('Finished Training') # but do we ever really get here?
