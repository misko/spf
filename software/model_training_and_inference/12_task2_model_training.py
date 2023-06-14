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
                    UNet)
from utils.spf_dataset import SessionsDataset, SessionsDatasetTask2, collate_fn

torch.set_printoptions(precision=5,sci_mode=False,linewidth=1000)


def model_forward(d_model,radio_inputs,radio_images,labels,label_images,args):
	d_model['optimizer'].zero_grad()

	_radio_inputs=radio_inputs[:,:d_model['snapshots_per_sample']]
	_radio_images=radio_images[:,:d_model['snapshots_per_sample']][:,:,0] # reshape to b,s,w,h
	_labels=labels[:,:d_model['snapshots_per_sample']]
	_label_images=label_images[:,d_model['snapshots_per_sample']-1] # reshape to b,1,w,h
	losses={}
	if d_model['images']==False:
		preds=d_model['net'](_radio_inputs)
		for k in preds:
			preds[k]=preds[k][:,:,cols_for_loss]
		transformer_loss=0.0
		single_snapshot_loss=0.0
		fc_loss=0.0
		if 'transformer_pred' in preds:
			if preds['transformer_pred'].mean(axis=1).var(axis=0).mean().item()<1e-13:
				d_model['dead']=True
			else:
				d_model['dead']=False
			transformer_loss = criterion(preds['transformer_pred'],_labels)
			losses['transformer_loss']=transformer_loss.item()
		elif 'single_snapshot_pred' in preds:
			single_snapshot_loss = criterion(preds['single_snapshot_pred'],_labels)
			losses['single_snapshot_loss']=single_snapshot_loss.item()
		elif 'fc_pred' in preds:
			fc_loss = criterion(preds['fc_pred'],_labels[:,[-1]])
			losses['fc_loss']=fc_loss.item()
		loss=transformer_loss+single_snapshot_loss+fc_loss
		if i<args.embedding_warmup:
			loss=single_snapshot_loss+fc_loss
	else:
		if d_model['normalize_input']:
			_radio_images=_radio_images/_radio_images.sum(axis=[2,3],keepdim=True)
		preds=d_model['net'](_radio_images)
		loss=criterion(preds['image_preds'],_label_images)
		losses['image_loss']=loss.item()
		if (i%args.print_every)==args.print_every-1:
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
			d_model['fig'].savefig('%s%s_%d.png' % (args.output_prefix,d_model['name'],i))
			d_model['fig'].canvas.draw_idle()
	return loss,losses

def net_to_losses(running_loss,mean_chunk):
	if len(running_loss)==0:
		return {}
	losses={}
	for k in ['baseline','baseline_image','image_loss','transformer_loss','single_snapshot_loss','fc_loss']:
		if k in running_loss[0]:
			losses[k]=np.log(np.array( [ np.mean([ l[k] for l in running_loss[idx*mean_chunk:(idx+1)*mean_chunk]])  
				for idx in range(len(running_loss)//mean_chunk) ]))
	return losses

def net_to_loss_str(running_loss,mean_chunk):
	if len(running_loss)==0:
		return ""
	loss_str=[]
	losses=net_to_losses(running_loss,mean_chunk)
	for k in ['image_loss','transformer_loss','single_snapshot_loss','fc_loss']:
		if k in losses:
			loss_str.append("%s:%0.4f" % (k,losses[k][-1]))
	return ",".join(loss_str)

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
		title):
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
	axs[3].plot(xs,baseline_image_loss['baseline_image'],label='baseline image')
	axs[0].set_title("Transformer loss")
	axs[1].set_title("Single snapshot loss")
	axs[2].set_title("FC loss")
	axs[3].set_title("Image loss")
	for d_model in models:
		losses=net_to_losses(running_losses[d_model['name']],mean_chunk)
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
	parser.add_argument('--output-prefix', type=str, required=False, default='net_out')
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
	parser.add_argument('--losses', type=str, required=False, default="src_pos") #,src_theta,src_dist,det_delta,det_theta,det_space")
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
	losses=args.losses.split(',')
	if 'src_pos' in losses:
		cols_for_loss+=[0,1]
	if 'src_theta' in losses:
		cols_for_loss+=[2]
	if 'src_dist' in losses:
		cols_for_loss+=[3]
	if 'det_delta' in losses:
		cols_for_loss+=[4,5]
	if 'det_theta' in losses:
		cols_for_loss+=[6]
	if 'det_space' in losses:
		cols_for_loss+=[7]

	device=torch.device(args.device)
	print("init dataset")
	ds=SessionsDatasetTask2(args.dataset,snapshots_in_sample=max(args.snapshots_per_sample))
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
	models=[ 
		{
			'name':'%d snapshots' % snapshots_per_sample, 
			'net':SnapshotNet(snapshots_per_sample),
			'snapshots_per_sample':snapshots_per_sample,
			'images':False,
			'lr':args.lr_transformer,
			'dead':False,
		}
		for snapshots_per_sample in args.snapshots_per_sample ]
	for snapshots_per_sample in args.snapshots_per_sample:
		models.append(
			{
				'name':'task1net%d' % snapshots_per_sample,
				'net':Task1Net(265*snapshots_per_sample),
				'snapshots_per_sample':snapshots_per_sample,
				'images':False,
				'lr':args.lr_direct,
				'dead':False,
			}
		)
	for snapshots_per_sample in args.snapshots_per_sample:
		models.append(
			{
				'name':'FCNet %d' % snapshots_per_sample,
				'net':SingleSnapshotNet(d_radio_feature=265,
					d_hid=64,
					d_embed=64,
					n_layers=4,
					n_outputs=8,
					dropout=0.0,
				snapshots_per_sample=snapshots_per_sample),
				'snapshots_per_sample':snapshots_per_sample,
				'images':False,
				'lr':args.lr_direct,
				'dead':False
			}
		)

	if False:
		for snapshots_per_sample in args.snapshots_per_sample: 
			models.append(
				{
					'name':'Unet %d' % snapshots_per_sample,
					'net':UNet(in_channels=snapshots_per_sample,out_channels=1,width=128),
					'snapshots_per_sample':snapshots_per_sample,
					'images':True,'fig':plt.figure(figsize=(14,4)),
					'normalize_input':True,
					'lr':args.lr_image,
					'dead':False
				}
			 )


	#move the models to the device
	for d_net in models:
		d_net['net']=d_net['net'].to(device)
	loss_figs={
		'train':plt.figure(figsize=(14,4)),
		'test':plt.figure(figsize=(14,4))}

	for d_model in models:
		d_model['optimizer']=optim.Adam(d_model['net'].parameters(),lr=d_model['lr'])

	criterion = nn.MSELoss()

	print("training loop")
	running_losses={'train':{},'test':{}}
	for k in ['train','test']:
		running_losses[k]={ d['name']:[] for d in models}
		running_losses[k]['baseline']=[]
		running_losses[k]['baseline_image']=[]

	saves=[]
	

	def prep_data(data):
		radio_inputs, radio_images, labels, label_images = data
		labels=labels[:,:,cols_for_loss]

		#direct data
		radio_inputs=radio_inputs.to(device)
		labels=labels.to(device)
		
		#image data
		radio_images=radio_images.to(device)
		label_images=label_images.to(device)
		
		return radio_inputs,labels,radio_images,label_images

	test_iterator = iter(testloader)
	for epoch in range(args.epochs): 
		for i, data in enumerate(trainloader, 0):
			#move to device, do final prep
			radio_inputs,labels,radio_images,label_images=prep_data(data)
				
			for d_model in models:
				loss,losses=model_forward(d_model,radio_inputs,radio_images,labels,label_images,args)
				loss.backward()
				running_losses['train'][d_model['name']].append(losses) 
				d_model['optimizer'].step()
			running_losses['train']['baseline'].append( {'baseline':criterion(labels*0+labels.mean(axis=[0,1],keepdim=True), labels).item() } )
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
							loss,losses=model_forward(d_model,radio_inputs,radio_images,labels,label_images,args)
							running_losses['test'][d_model['name']].append(losses) 
					running_losses['test']['baseline'].append( {'baseline':criterion(labels*0+labels.mean(axis=[0,1],keepdim=True), labels).item() } )
					running_losses['test']['baseline_image'].append( {'baseline_image':criterion(label_images*0+label_images.mean(), label_images).item() } )
				
	
			if i==0 or i%args.save_every==args.save_every-1:
				save(args,running_losses,models,i,args.keep_n_saves)

			if i % args.print_every == args.print_every-1:

				train_baseline_loss=net_to_losses(running_losses['train']['baseline'],args.print_every)
				train_baseline_image_loss=net_to_losses(running_losses['train']['baseline_image'],args.print_every)
				test_baseline_loss=net_to_losses(running_losses['test']['baseline'],args.test_mbs)
				test_baseline_image_loss=net_to_losses(running_losses['test']['baseline_image'],args.test_mbs)

				print(f'[{epoch + 1}, {i + 1:5d}]')
				print(f'\tTrain: baseline: {train_baseline_loss["baseline"][-1]:.3f}, baseline_image: {train_baseline_image_loss["baseline_image"][-1]:.3f} , time { (time.time()-start_time)/(i+1) :.3f} / batch' )
				print(f'\tTest: baseline: {test_baseline_loss["baseline"][-1]:.3f}, baseline_image: {test_baseline_image_loss["baseline_image"][-1]:.3f} , time { (time.time()-start_time)/(i+1) :.3f} / batch' )
				loss_str="\t"+"\n\t".join(
					[ "%s(%s):(tr)%s,(ts)%s" % (d['name'],str(d['dead']),
						net_to_loss_str(running_losses['train'][d['name']],args.print_every),
						net_to_loss_str(running_losses['test'][d['name']],args.test_mbs)
					) for d in models ])
				print(loss_str)
				if i//args.print_every>2:
					plot_loss(running_losses=running_losses['train'],
						baseline_loss=train_baseline_loss,
						baseline_image_loss=train_baseline_image_loss,
						xtick_spacing=args.print_every,
						mean_chunk=args.print_every,
						output_prefix=args.output_prefix,
						fig=loss_figs['train'],
						title='Train')
					plot_loss(running_losses=running_losses['test'],
						baseline_loss=test_baseline_loss,
						baseline_image_loss=test_baseline_image_loss,
						xtick_spacing=args.print_every,
						mean_chunk=args.test_mbs,
						output_prefix=args.output_prefix,
						fig=loss_figs['test'],
						title='Test')
					if args.plot:
						plt.pause(0.5)

	print('Finished Training') # but do we ever really get here?
