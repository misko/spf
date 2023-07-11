import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import skimage
from utils.image_utils import (detector_positions_to_theta_grid,
						 labels_to_source_images, radio_to_image)
	
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

def line_to_mx(line):
	(x,y),((m_x,m_y),)=line
	m=m_y/(m_x+1e-9)
	return y-m*x,m,(x,y),(m_x,m_y)

def baseline_algorithm(session,width,steps=-1):
	line_representations=[]
	if steps==-1:
		session['beam_former_outputs_at_t'].shape[0]
	for idx in np.arange(steps):
		_thetas=session['thetas_at_t'][idx][get_top_n_peaks(session['beam_former_outputs_at_t'][idx])]
		for _theta in _thetas:
			direction=np.stack([	  
				np.sin(session['detector_orientation_at_t'][idx]+_theta),
				np.cos(session['detector_orientation_at_t'][idx]+_theta)
			],axis=1)
			line_representations.append((session['detector_position_at_t'][idx],direction))
	return get_top_n_points(line_representations,n=4,width=width,threshold=3)


def get_top_n_points(line_representations,n,width,threshold=3,mn=4):
	final_points=[]
	line_to_point_assignments=np.zeros(len(line_representations),dtype=int)-1
	line_to_point_distances=np.zeros(len(line_representations),dtype=int)-1
	for point_idx in range(n): 
		img=np.zeros((width+1,width+1))
		for line_idx in np.arange(len(line_representations)):
			if line_to_point_assignments[line_idx]==-1 or line_to_point_distances[line_idx]>=threshold:
				line=line_representations[line_idx]
				if len(final_points)>0: # check if its close to the last line
					b,m,point,mvec=line_to_mx(line)
					fp=final_points[-1]
					d=(fp[0]-point[0],fp[1]-point[1])
					if np.sign(mvec[0])==np.sign(d[0]) and np.sign(mvec[1])==np.sign(d[1]): # check if its on the right side
						mvec_orthogonal=(-mvec[1],mvec[0])
						distance_to_line=abs(np.dot(d,mvec_orthogonal)/np.linalg.norm(mvec))
						if line_to_point_assignments[line_idx]==-1 or distance_to_line<line_to_point_distances[line_idx]:
							line_to_point_assignments[line_idx]=len(final_points)-1
							line_to_point_distances[line_idx]=distance_to_line
				if line_to_point_assignments[line_idx]==-1 or line_to_point_distances[line_idx]>=threshold:
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
	final_points=np.array(final_points)
	points_per_line=np.zeros((len(line_representations),2))+width/2
	for line_idx in np.arange(len(line_representations)):
		if line_to_point_assignments[line_idx]!=-1:
			points_per_line[line_idx]=final_points[line_to_point_assignments[line_idx]]
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
	return final_points,imgs,points_per_line

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
