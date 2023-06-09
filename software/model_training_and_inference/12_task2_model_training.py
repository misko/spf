import argparse
import time
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from utils.image_utils import labels_to_source_images
from models.models import (SingleSnapshotNet, SnapshotNet, Task1Net, TransformerModel,
                    UNet)
from utils.spf_dataset import SessionsDataset, SessionsDatasetTask2Simple, collate_fn

torch.set_printoptions(precision=5,sci_mode=False,linewidth=1000)

if __name__=='__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, required=False, default='cpu')
	parser.add_argument('--embedding-warmup', type=int, required=False, default=256*32)
	parser.add_argument('--snapshots-per-sample', type=int, required=False, default=[1,4,8], nargs="+")
	parser.add_argument('--print-every', type=int, required=False, default=100)
	parser.add_argument('--mb', type=int, required=False, default=64)
	parser.add_argument('--workers', type=int, required=False, default=4)
	parser.add_argument('--dataset', type=str, required=False, default='./sessions_task1')
	parser.add_argument('--lr', type=float, required=False, default=0.000001)
	parser.add_argument('--plot', type=bool, required=False, default=False)
	parser.add_argument('--losses', type=str, required=False, default="src_pos,src_theta,src_dist,det_delta,det_theta,det_space")
	args = parser.parse_args()

	start_time=time.time()

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
	ds=SessionsDatasetTask2Simple(args.dataset,snapshots_in_sample=max(args.snapshots_per_sample))
	
	print("init dataloader")
	trainloader = torch.utils.data.DataLoader(
			ds, 
			batch_size=args.mb,
			shuffle=True, 
			num_workers=args.workers,
			collate_fn=collate_fn)

	print("init network")
	nets=[ {'name':'%d snapshots' % snapshots_per_sample, 
                'net':SnapshotNet(snapshots_per_sample).to(device),
                 'snapshots_per_sample':snapshots_per_sample,
		'images':False}
		for snapshots_per_sample in args.snapshots_per_sample ]
	#for snapshots_per_sample in args.snapshots_per_sample:
	#	nets.append(
	#		{'name':'task1net%d' % snapshots_per_sample,
	#		'net':Task1Net(265*snapshots_per_sample), 'snapshots_per_sample':snapshots_per_sample}
	#	)
	for snapshots_per_sample in args.snapshots_per_sample:
		nets.append(
			{'name':'FCNet %d' % snapshots_per_sample,
			'net':SingleSnapshotNet(d_radio_feature=265,
                        	d_hid=64,
                        	d_embed=64,
                        	n_layers=4,
                        	n_outputs=8,
                        	dropout=0.0,
                        snapshots_per_sample=snapshots_per_sample).to(device),
			'snapshots_per_sample':snapshots_per_sample,
			'images':False,
			}
		)

	nets=[]
	for snapshots_per_sample in args.snapshots_per_sample: 
		nets.append({'name':'Unet %d' % snapshots_per_sample,
		'net':UNet(in_channels=snapshots_per_sample,out_channels=1,width=128).to(device),
		'snapshots_per_sample':snapshots_per_sample,
		'images':True}
		 )


	imfig=plt.figure(figsize=(14,4))
	fig=plt.figure(figsize=(14,4))

	for d_net in nets:
		d_net['optimizer']=optim.Adam(d_net['net'].parameters(),lr=args.lr)
	criterion = nn.MSELoss()

	print("training loop")
	running_losses={ d['name']:[] for d in nets}
	running_losses['baseline']=[]
	running_losses['baseline_image']=[]

	for epoch in range(200):  # loop over the dataset multiple times
		for i, data in enumerate(trainloader, 0):
			radio_inputs, radio_images, labels, label_images = data
			labels=labels[:,:,cols_for_loss]

	
			radio_inputs=radio_inputs.to(device)
			labels=labels.to(device)
			radio_images=radio_images.to(device)
			label_images=label_images.to(device)
			for d_net in nets:
				d_net['optimizer'].zero_grad()

				_radio_inputs=radio_inputs[:,:d_net['snapshots_per_sample']]
				_radio_images=radio_images[:,:d_net['snapshots_per_sample']][:,:,0] # reshape to b,s,w,h
				_labels=labels[:,:d_net['snapshots_per_sample']]
				_label_images=label_images[:,d_net['snapshots_per_sample']-1] # reshape to b,1,w,h
				losses={}
				if d_net['images']==False:
					preds=d_net['net'](_radio_inputs)
					for k in preds:
						preds[k]=preds[k][:,:,cols_for_loss]
					transformer_loss=0.0
					single_snapshot_loss=0.0
					fc_loss=0.0
					if 'transformer_pred' in preds:
						transformer_loss = criterion(preds['transformer_pred'],_labels)
						losses['transformer_loss']=transformer_loss.item()
					if 'single_snapshot_pred' in preds:
						single_snapshot_loss = criterion(preds['single_snapshot_pred'],_labels)
						losses['single_snapshot_loss']=single_snapshot_loss.item()
					if 'fc_pred' in preds:
						fc_loss = criterion(preds['fc_pred'],_labels[:,[-1]])
						losses['fc_loss']=fc_loss.item()
					loss=transformer_loss+single_snapshot_loss+fc_loss
					if i<args.embedding_warmup:
						loss=single_snapshot_loss+fc_loss
				else:
					preds=d_net['net'](_radio_images)
					loss=criterion(preds['image_preds'],_label_images)
					losses['image_loss']=loss.item()
					if args.plot and (i%args.print_every)==args.print_every-1:
						imfig.clf()
						#imfig=plt.figure()
						axs=imfig.subplots(1,3,sharex=True)
						axs[0].imshow(_label_images[0,0].cpu())
						axs[1].imshow(preds['image_preds'][0,0].detach().cpu())
						axs[2].imshow(_radio_images[0].mean(axis=0).cpu())
						imfig.canvas.draw_idle()
						plt.pause(0.1)
				#if i%1000==0:
				#	print("TFORMER",tformer_preds[0])
				#	print("SINGLE",single_snapshot_preds[0])
				#	print("LABEL",_labels[0])
				loss.backward()
				running_losses[d_net['name']].append(losses) # += np.log(np.array([single_snapshot_loss.item(),tformer_loss.item()]))
				d_net['optimizer'].step()
			running_losses['baseline'].append( {'baseline':criterion(labels*0+labels.mean(axis=[0,1],keepdim=True), labels).item() } )
			running_losses['baseline_image'].append( {'baseline_image':criterion(label_images*0+label_images.mean(), label_images).item() } )


			def net_to_losses(name):
				rl=running_losses[name]
				if len(rl)==0:
					return {}
				losses={}
				for k in ['baseline','baseline_image','image_loss','transformer_loss','single_snapshot_loss','fc_loss']:
					if k in rl[0]:
						losses[k]=np.log(np.array( [ np.mean([ l[k] for l in rl[idx*args.print_every:(idx+1)*args.print_every]])  
							for idx in range(len(rl)//args.print_every) ]))
				return losses
				

			def net_to_loss_str(name):
				rl=running_losses[name]
				if len(rl)==0:
					return ""
				loss_str=[name]
				losses=net_to_losses(name)
				for k in ['image_loss','transformer_loss','single_snapshot_loss','fc_loss']:
					if k in losses:
						loss_str.append("%s:%0.4f" % (k,losses[k][-1]))
				return ",".join(loss_str)

			if i % args.print_every == args.print_every-1:
				loss_str="\t"+"\n\t".join([ net_to_loss_str(d['name']) for d in nets ])
				baseline_loss=net_to_losses('baseline')
				baseline_image_loss=net_to_losses('baseline_image')
				print(f'[{epoch + 1}, {i + 1:5d}]\n\tbaseline: {baseline_loss["baseline"][-1]:.3f}, baseline_image: {baseline_image_loss["baseline_image"][-1]:.3f} , time { (time.time()-start_time)/i :.3f} / batch' )
				print(loss_str)
				if args.plot and i//args.print_every>2:
					fig.clf()
					axs=fig.subplots(1,4,sharex=True)
					axs[0].get_shared_y_axes().join(axs[0], axs[1])
					axs[0].get_shared_y_axes().join(axs[0], axs[2])
					xs=np.arange(len(baseline_loss['baseline']))*args.print_every
					for i in range(3):
						axs[i].plot(xs,baseline_loss['baseline'],label='baseline')
						axs[i].set_xlabel("time")
						axs[i].set_ylabel("log loss")
					axs[3].plot(xs,baseline_image_loss['baseline_image'],label='baseline image')
					axs[0].set_title("Transformer loss")
					axs[1].set_title("Single snapshot loss")
					axs[2].set_title("FC loss")
					axs[3].set_title("Image loss")
					for d_net in nets:
						losses=net_to_losses(d_net['name'])
						if 'transformer_loss' in losses:
							axs[0].plot(xs,losses['transformer_loss'],label=d_net['name'])
						if 'single_snapshot_loss' in losses:
							axs[1].plot(xs,losses['single_snapshot_loss'],label=d_net['name'])
						if 'fc_loss' in losses:
							axs[2].plot(xs,losses['fc_loss'],label=d_net['name'])
						if 'image_loss' in losses:
							axs[3].plot(xs,losses['image_loss'],label=d_net['name'])
					for i in range(4):
						axs[i].legend()
					fig.tight_layout()
					#fig.pause(0.001)
					#plt.ion()     # turns on interactive mode
					plt.pause(0.1)
				#running_losses={ d['name']:np.zeros(2) for d in nets }
		
	print('Finished Training')
