import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from utils.image_utils import (detector_positions_to_theta_grid,
                         labels_to_source_images, radio_to_image)


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

def plot_trajectory(ax,
	 positions,
	 width,
	 ms=30,
	 steps_per_fade=10,
	 fadep=0.8,
	 c='b',
	 rings=False,
	 label=None):
	ax.set_xlim([0,width])
	ax.set_ylim([0,width])
	n_steps=positions.shape[0]//steps_per_fade
	if positions.shape[0]%steps_per_fade!=0:
		n_steps+=1

	alpha=fadep
	for n in np.arange(1,n_steps):
		start=positions.shape[0]-(n+1)*steps_per_fade
		end=start+steps_per_fade
		start=max(0,start)
		ax.plot( positions[start:end,0], positions[start:end,1],'--',alpha=alpha,color=c,label=label)
		alpha*=fadep
	start=positions.shape[0]-steps_per_fade
	end=start+steps_per_fade
	ax.plot( positions[start:end,0], positions[start:end,1],'--',alpha=1.0,color=c,label=label)
	
	ax.plot( positions[-1,0], positions[-1,1],'.',ms=ms,c=c)
	if rings:
		n=4
		for x in range(n):
			ax.plot( positions[-1,0], positions[-1,1],'.',ms=ms*(1.8**x),c=c,alpha=1/n)


#generate the images for the session
def plot_full_session(session,steps,output_prefix):
	width=session['width_at_t'][0][0]
	
	#extract the images
	d={}
	d['source_image_at_t']=labels_to_source_images(torch.from_numpy(session['source_positions_at_t'])[None],width)[0]
	d['detector_theta_image_at_t']=detector_positions_to_theta_grid(session['detector_position_at_t'][None],width)[0]
	d['radio_image_at_t']=radio_to_image(session['beam_former_outputs_at_t'][None],d['detector_theta_image_at_t'][None],session['detector_orientation_at_t'][None])[0]
	d['radio_image_at_t_normed']=d['radio_image_at_t']/d['radio_image_at_t'].sum(axis=2,keepdims=True).sum(axis=3,keepdims=True)
	filenames=[]
	plt.ioff()
	for idx in np.arange(1,steps):
		fig=plt.figure(figsize=(12,12))
		axs=fig.subplots(2,2)
		for _a in [0,1]:
			for _b in [0,1]:
				if _a==0 and _b==1: 
					continue
				axs[_a,_b].set_xlabel("X (m)")
				axs[_a,_b].set_ylabel("Y (m)")

		axs[0,0].set_title("Position map")
		plot_trajectory(axs[0,0],session['detector_position_at_t'][:idx],width,ms=30,label='detector')
		direction=session['detector_position_at_t'][idx-1]+0.25*session['width_at_t'][0]*np.stack(
			[np.cos(session['detector_orientation_at_t'][idx-1]),np.sin(session['detector_orientation_at_t'][idx-1])],axis=1)
		axs[0,0].plot(
			[session['detector_position_at_t'][idx-1][0],direction[0,0]],
			[session['detector_position_at_t'][idx-1][1],direction[0,1]])
		anti_direction=session['detector_position_at_t'][idx-1]+0.25*session['width_at_t'][0]*np.stack(
			[np.cos(session['detector_orientation_at_t'][idx-1]+np.pi/2),np.sin(session['detector_orientation_at_t'][idx-1]+np.pi/2)],axis=1)
		axs[0,0].plot(
			[session['detector_position_at_t'][idx-1][0],anti_direction[0,0]],
			[session['detector_position_at_t'][idx-1][1],anti_direction[0,1]])
		emitter_direction=session['detector_position_at_t'][idx-1]+0.25*session['width_at_t'][0]*np.stack(
			[
				np.cos(session['detector_orientation_at_t'][idx-1]+session['source_theta_at_t'][idx-1,0]),
				np.sin(session['detector_orientation_at_t'][idx-1]+session['source_theta_at_t'][idx-1,0])
			],axis=1)
		axs[0,0].plot(
			[session['detector_position_at_t'][idx-1][0],emitter_direction[0,0]],
			[session['detector_position_at_t'][idx-1][1],emitter_direction[0,1]])
		for n in np.arange(session['source_positions_at_t'].shape[1]):
			rings=(session['broadcasting_positions_at_t'][idx,n,0]==1)
			plot_trajectory(axs[0,0],session['source_positions_at_t'][:idx,n],width,ms=15,c='r',rings=rings,label='emitter %d' % n)

		#axs[0,0].legend()
		handles, labels = axs[0,0].get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		axs[0,0].legend(by_label.values(), by_label.keys())

		#lets draw the radio
		axs[1,0].imshow(d['source_image_at_t'][idx,0].T,origin='lower')
		axs[1,0].set_title("Emitters as image at t=%d" % idx)

		axs[1,1].imshow(d['radio_image_at_t'][idx,0].T,origin='lower')
		axs[1,1].set_title("Radio feature at t=%d" % idx)

		axs[0,1].plot(session['thetas_at_t'][idx],session['beam_former_outputs_at_t'][idx])
		axs[0,1].axvline(x=session['source_theta_at_t'][idx-1,0],c='r')
		axs[0,1].set_title("Beamformer output at t=%d" % idx)
		axs[0,1].set_xlabel("Theta (rel. to detector)")
		axs[0,1].set_ylabel("Signal strength")

		fn='%s_%04d.png' % (output_prefix,idx)
		filenames.append(fn)
		fig.savefig(fn)
		plt.close(fig)
	plt.ion()
	return filenames


def filenames_to_gif(filenames,output_gif_fn):
	images=[]
	for fn in filenames:
		images.append(Image.open(fn).resize((600,600)))

	images[0].save(output_gif_fn,
					 save_all = True, append_images = images[1:], 
					 optimize = False, duration = 200,loop=0)
