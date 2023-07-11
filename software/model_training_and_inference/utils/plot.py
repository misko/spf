import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import skimage
from utils.image_utils import (detector_positions_to_theta_grid,
						 labels_to_source_images, radio_to_image)


def line_to_mx(line):
	(x,y),((m_x,m_y),)=line
	m=m_y/(m_x+1e-9)
	return y-m*x,m,(x,y),(m_x,m_y)

def get_top_n_points(line_representations,n,width,threshold=3,mn=4):
	final_points=[]
	line_to_point_assignments=np.zeros(len(line_representations),dtype=int)-1
	for point_idx in range(n): 
		img=np.zeros((width+1,width+1))
		for line_idx in np.arange(len(line_representations)):
			if line_to_point_assignments[line_idx]==-1:
				line=line_representations[line_idx]
				if len(final_points)>0: # check if its close to the last line
					b,m,point,mvec=line_to_mx(line)
					fp=final_points[-1]
					d=(fp[0]-point[0],fp[1]-point[1])
					if np.sign(mvec[0])==np.sign(d[0]) and np.sign(mvec[1])==np.sign(d[1]): # check if its on the right side
						mvec_orthogonal=(-mvec[1],mvec[0])
						distance_to_line=abs(np.dot(d,mvec_orthogonal)/np.linalg.norm(mvec))
						if distance_to_line<threshold:
							line_to_point_assignments[line_idx]=len(final_points)-1
				if line_to_point_assignments[line_idx]==-1:
					b,m,point,mvec=line_to_mx(line)
					bp=boundary_point(b,m,point,width,mvec)
					rr,cc=skimage.draw.line(
							int(point[0]), int(point[1]), 
							int(bp[0]),int(bp[1]))
					img[rr,cc]+=1
		if img.max()>=mn:
			idx=img.argmax()
			p=(idx//(width+1),idx%(width+1))
			final_points.append(p)
	imgs=np.zeros((n,width+1,width+1))
	for point_idx in range(n): 
		for line_idx in np.arange(len(line_representations)):
			if line_to_point_assignments[line_idx]>=0:
				line=line_representations[line_idx]
				if line_to_point_assignments[line_idx]>=0:
					b,m,point,mvec=line_to_mx(line)
					bp=boundary_point(b,m,point,width,mvec)
					rr,cc=skimage.draw.line(
							int(point[0]), int(point[1]), 
							int(bp[0]),int(bp[1]))
					imgs[line_to_point_assignments[line_idx]][rr,cc]+=1
	return final_points,imgs

def boundary_point(b,m,point,width,mvec):
	y_xmax=m*width+b
	y_xmin=b
	x_ymin=-b/m
	x_ymax=(width-b)/m
	if mvec[0]>0:
		if mvec[1]>0:
			p1=(x_ymax,width)
			p2=(width,y_xmax)
		elif mvec[1]<0:
			p1=(x_ymin,0)
			p2=(width,y_xmax)
		else:
			p1=(width,point[1])
			p2=(width,point[1])
	elif mvec[0]<0:
		if mvec[1]>0:
			p1=(0,y_xmin)
			p2=(x_ymax,width)
		elif mvec[1]<0:
			p1=(0,y_xmin)
			p2=(x_ymin,0)
		else:
			p1=(0,point[1])
			p2=(0,point[1])
	else:
		if mvec[1]>0:
			p1=(point[0],width)
			p2=(point[0],width)
		elif mvec[1]<0:
			p1=(point[0],0)
			p2=(point[0],0)
		else:
			assert(False)
	p=p1
	if p1[0]<0 or p1[0]>width or p1[1]<0 or p1[1]>width:
		p=p2
	assert(p[0]>=0 and p[0]<=width and p[1]>=0 and p[1]<=width)
	return p


def lines_to_points(lines,t):
	lines=[ line_to_mx(line) for line in lines ]
	line_to_points=[]
	for line_idx_a in np.arange(len(lines)):
		rng = np.random.default_rng(12345+line_idx_a*1337)
		a_i,a_m,(x1,y1),_=lines[line_idx_a]
		points_for_this_line=[]
		for line_idx_b in rng.choice(np.arange(t),size=min(30,t),replace=False):
			if line_idx_b>=len(lines):
				continue
			if line_idx_a==line_idx_b:
				continue
			b_i,b_m,(x2,y2),_=lines[line_idx_b]
			#compute the intercept
			if a_i!=b_m and a_m!=b_m:
				_x=(b_i-a_i)/(a_m-b_m)
				_y=a_m*((b_i-a_i)/(a_m-b_m))+a_i
				#check if the point is valid
				if a_m<0 and _x<x1:
					pass
				elif a_m>0 and _x>x1:
					pass
				elif b_m<0 and _x<x2:
					pass
				elif b_m>0 and _x>x2:
					pass
				elif ((x1-_x)**2+(y1-_y)**2)<800:
					pass
				elif ((x2-_x)**2+(y2-_y)**2)<800:
					pass
				else:
					points_for_this_line.append((_x,_y))
		line_to_points.append(points_for_this_line)
	return line_to_points

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

def get_top_n_peaks(bf,n=2,peak_size=15):
	bf=np.copy(bf)
	peaks=np.zeros(n,dtype=int)
	for peak_idx in range(n):
		#find a peak
		peak=bf.argmax()
		for idx in np.arange(-peak_size//2,peak_size//2+1):
			bf[(peak+idx)%bf.shape[0]]=-np.inf
		peaks[peak_idx]=peak
	return peaks

def frac_to_theta(fracs):
	return 2*(fracs-0.5)*np.pi

def plot_lines(session,steps,output_prefix):
	width=session['width_at_t'][0][0]
	
	#extract the images
	d={}
	filenames=[]
	plt.ioff()
	lines=[]
	line_representations=[]
	for idx in np.arange(1,steps):
		fig=plt.figure(figsize=(9*3,9))
		axs=fig.subplots(1,3)

		plot_trajectory(axs[0],session['detector_position_at_t'][:idx],width,ms=30,label='detector')
		plot_trajectory(axs[1],session['detector_position_at_t'][:idx],width,ms=30,label='detector')
		direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*np.stack(
			[np.sin(session['detector_orientation_at_t'][idx]),np.cos(session['detector_orientation_at_t'][idx])],axis=1)
		axs[0].plot(
			[session['detector_position_at_t'][idx][0],direction[0,0]],
			[session['detector_position_at_t'][idx][1],direction[0,1]])
		anti_direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*np.stack(
			[np.sin(session['detector_orientation_at_t'][idx]+np.pi/2),np.cos(session['detector_orientation_at_t'][idx]+np.pi/2)],axis=1)
		axs[0].plot(
			[session['detector_position_at_t'][idx][0],anti_direction[0,0]],
			[session['detector_position_at_t'][idx][1],anti_direction[0,1]])
		_thetas=session['thetas_at_t'][idx][get_top_n_peaks(session['beam_former_outputs_at_t'][idx])]
		for _theta in _thetas:
			direction=np.stack([	  
				np.sin(session['detector_orientation_at_t'][idx]+_theta),
				np.cos(session['detector_orientation_at_t'][idx]+_theta)
			],axis=1)
			emitter_direction=session['detector_position_at_t'][idx]+2.0*session['width_at_t'][0]*direction
			lines.append(([session['detector_position_at_t'][idx][0],emitter_direction[0,0]],
										 [session['detector_position_at_t'][idx][1],emitter_direction[0,1]]))
			line_representations.append((session['detector_position_at_t'][idx],direction))


		for x,y in lines:
			axs[0].plot(x,y,c='blue',linewidth=4,alpha=0.1)
			#print("PLOT",x,y)
		emitter_direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*np.stack(
			[
				np.sin(session['detector_orientation_at_t'][idx]+session['source_theta_at_t'][idx,0]),
				np.cos(session['detector_orientation_at_t'][idx]+session['source_theta_at_t'][idx,0])
			],axis=1)
		axs[0].plot(
			[session['detector_position_at_t'][idx][0],emitter_direction[0,0]],
			[session['detector_position_at_t'][idx][1],emitter_direction[0,1]])
		for n in np.arange(session['source_positions_at_t'].shape[1]):
			rings=(session['broadcasting_positions_at_t'][idx,n,0]==1)
			plot_trajectory(axs[0],session['source_positions_at_t'][:idx,n],width,ms=15,c='r',rings=rings,label='emitter %d' % n)
		handles, labels = axs[0].get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		axs[0].legend(by_label.values(), by_label.keys())

		axs[2].plot(session['thetas_at_t'][idx],session['beam_former_outputs_at_t'][idx])
		axs[2].axvline(x=session['source_theta_at_t'][idx,0],c='r')
		axs[2].set_title("Beamformer output at t=%d" % idx)
		axs[2].set_xlabel("Theta (rel. to detector)")
		axs[2].set_ylabel("Signal strength")
		axs[0].set_title("Position map")
		axs[0].set_xlabel("X (m)")
		axs[0].set_ylabel("Y (m)")
		axs[1].set_title("Guess map")
		axs[1].set_xlabel("X (m)")
		axs[1].set_ylabel("Y (m)")

		fp,imgs=get_top_n_points(line_representations,n=4,width=width,threshold=3)
		axs[1].imshow(imgs[:3].transpose([2,1,0])/imgs.max())
		colors=['r','green','blue']
		for _idx in range(min(3,len(fp))):
			axs[1].scatter([fp[_idx][0]],[fp[_idx][1]],color=colors[_idx],s=900)

		fn='%s_%04d_lines.png' % (output_prefix,idx)
		filenames.append(fn)
		fig.savefig(fn)
		plt.close(fig)
	plt.ion()
	return filenames

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
		direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*np.stack(
			[np.sin(session['detector_orientation_at_t'][idx]),np.cos(session['detector_orientation_at_t'][idx])],axis=1)
		axs[0,0].plot(
			[session['detector_position_at_t'][idx][0],direction[0,0]],
			[session['detector_position_at_t'][idx][1],direction[0,1]])
		anti_direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*np.stack(
			[np.sin(session['detector_orientation_at_t'][idx]+np.pi/2),np.cos(session['detector_orientation_at_t'][idx]+np.pi/2)],axis=1)
		axs[0,0].plot(
			[session['detector_position_at_t'][idx][0],anti_direction[0,0]],
			[session['detector_position_at_t'][idx][1],anti_direction[0,1]])
		emitter_direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*np.stack(
			[
				np.sin(session['detector_orientation_at_t'][idx]+session['source_theta_at_t'][idx,0]),
				np.cos(session['detector_orientation_at_t'][idx]+session['source_theta_at_t'][idx,0])
			],axis=1)
		axs[0,0].plot(
			[session['detector_position_at_t'][idx][0],emitter_direction[0,0]],
			[session['detector_position_at_t'][idx][1],emitter_direction[0,1]])
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
		axs[0,1].axvline(x=session['source_theta_at_t'][idx,0],c='r')
		axs[0,1].set_title("Beamformer output at t=%d" % idx)
		axs[0,1].set_xlabel("Theta (rel. to detector)")
		axs[0,1].set_ylabel("Signal strength")

		fn='%s_%04d.png' % (output_prefix,idx)
		filenames.append(fn)
		fig.savefig(fn)
		plt.close(fig)
	plt.ion()
	return filenames


def filenames_to_gif(filenames,output_gif_fn,size=(600,600)):
	images=[]
	for fn in filenames:
		images.append(Image.open(fn).resize(size))

	images[0].save(output_gif_fn,
					 save_all = True, append_images = images[1:], 
					 optimize = False, duration = 200,loop=0)
