import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import skimage
from utils.image_utils import (detector_positions_to_theta_grid,
             labels_to_source_images, radio_to_image)

from utils.baseline_algorithm import get_top_n_peaks, baseline_algorithm


#depends if we are using y=0 or x=0 as origin
#if x=0 (y+) then we want x=sin, y=cos
#if y=0 (x+) then we want x=cos, y=sin
def get_xy_from_theta(theta):
  return np.array([
      np.sin(theta),
      np.cos(theta),
  ]).T

def plot_predictions_and_baseline(session,args,step,pred_a,pred_b):
  width=session['width_at_t'][0][0]

  fig=plt.figure(figsize=(5*2,5*2))
  axs=fig.subplots(2,2)

  plot_trajectory(axs[0,0],session['detector_position_at_t'][:step],width,ms=30,label='detector')
  plot_trajectory(axs[0,1],session['detector_position_at_t'][:step],width,ms=30,label='detector')

  #plot directions on the the space diagram
  direction=session['detector_position_at_t'][step]+0.25*session['width_at_t'][0]*get_xy_from_theta(session['detector_orientation_at_t'][step])
  axs[0,0].plot(
    [session['detector_position_at_t'][step][0],direction[0,0]],
    [session['detector_position_at_t'][step][1],direction[0,1]])
  anti_direction=session['detector_position_at_t'][step]+0.25*session['width_at_t'][0]*get_xy_from_theta(session['detector_orientation_at_t'][step]+np.pi/2)
  axs[0,0].plot(
    [session['detector_position_at_t'][step][0],anti_direction[0,0]],
    [session['detector_position_at_t'][step][1],anti_direction[0,1]])

  #for every time step plot the lines
  lines=[]
  for idx in range(step+1):
    _thetas=session['thetas_at_t'][idx][get_top_n_peaks(session['beam_former_outputs_at_t'][idx])]
    for _theta in _thetas:
      direction=get_xy_from_theta(session['detector_orientation_at_t'][idx]+_theta)
      emitter_direction=session['detector_position_at_t'][idx]+2.0*session['width_at_t'][0]*direction
      lines.append(([session['detector_position_at_t'][idx][0],emitter_direction[0,0]],
                     [session['detector_position_at_t'][idx][1],emitter_direction[0,1]]))

  for x,y in lines:
    axs[0,1].plot(x,y,c='blue',linewidth=4,alpha=0.1)

  for n in np.arange(session['source_positions_at_t'].shape[1]):
    #rings=(session['broadcasting_positions_at_t'][idx,n,0]==1)
    rings=False
    plot_trajectory(axs[0,0],session['source_positions_at_t'][:idx,n],width,ms=15,c='r',rings=rings,label='emitter %d' % n)
  axs[0,0].set_title("Position map")
  axs[0,1].set_title("Direction estimates")
  for x in [0,1]:
    handles, labels = axs[0,x].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0,x].legend(by_label.values(), by_label.keys())
    for y in [0,1]:
      #axs[y,x].set_title("Position map")
      axs[y,x].set_xlabel("X (m)")
      axs[y,x].set_ylabel("Y (m)")
      axs[y,x].set_xlim([0,width])
      axs[y,x].set_ylim([0,width])

  true_positions=session['source_positions_at_t'][session['broadcasting_positions_at_t'].astype(bool)[...,0]]
  true_positions_noise=true_positions+np.random.randn(*true_positions.shape)*3

  for ax_idx,pred in [(0,pred_a),(1,pred_b)]:  
    axs[1,ax_idx].set_title("error in %s" % pred['name'])
    axs[1,ax_idx].scatter(pred['predictions'][:,0],pred['predictions'][:,1],s=15,c='r',alpha=0.1)
    for idx in np.arange(step):
      _x,_y=pred['predictions'][idx]+np.random.randn(2)
      x,y=true_positions_noise[idx]
      axs[1,ax_idx].plot([_x,x],[_y,y],color='black',linewidth=1,alpha=0.1)


  fn='%s_%04d_lines.png' % (args.output_prefix,step)
  fig.savefig(fn)
  plt.close(fig)
  return fn
 
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


def plot_lines(session,steps,output_prefix):
  width=session['width_at_t'][0][0]
  
  #extract the images
  d={}
  filenames=[]
  plt.ioff()
  lines=[]
  for idx in np.arange(1,steps):
    fig=plt.figure(figsize=(9*3,9))
    axs=fig.subplots(1,3)

    plot_trajectory(axs[0],session['detector_position_at_t'][:idx],width,ms=30,label='detector')
    plot_trajectory(axs[1],session['detector_position_at_t'][:idx],width,ms=30,label='detector')
    direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*get_xy_from_theta(session['detector_orientation_at_t'][idx])
    axs[0].plot(
      [session['detector_position_at_t'][idx][0],direction[0,0]],
      [session['detector_position_at_t'][idx][1],direction[0,1]])
    anti_direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*get_xy_from_theta(session['detector_orientation_at_t'][idx]+np.pi/2)
    axs[0].plot(
      [session['detector_position_at_t'][idx][0],anti_direction[0,0]],
      [session['detector_position_at_t'][idx][1],anti_direction[0,1]])
    _thetas=session['thetas_at_t'][idx][get_top_n_peaks(session['beam_former_outputs_at_t'][idx])]
    for _theta in _thetas:
      direction=get_xy_from_theta(session['detector_orientation_at_t'][idx]+_theta)
      emitter_direction=session['detector_position_at_t'][idx]+2.0*session['width_at_t'][0]*direction
      lines.append(([session['detector_position_at_t'][idx][0],emitter_direction[0,0]],
                     [session['detector_position_at_t'][idx][1],emitter_direction[0,1]]))

    for x,y in lines:
      axs[0].plot(x,y,c='blue',linewidth=4,alpha=0.1)

    emitter_direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*get_xy_from_theta(session['detector_orientation_at_t'][idx]+session['source_theta_at_t'][idx,0])
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

    fp,imgs,pred_points=baseline_algorithm(session,width,steps=idx)
    for pred_point in pred_points:
      axs[0].scatter([pred_point[0]],[pred_point[1]])
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
def plot_full_session(session,steps,output_prefix,img_width=128,invert=False):
  width=session['width_at_t'][0][0]
  
  #extract the images
  d={}
  d['source_image_at_t']=labels_to_source_images(torch.from_numpy(session['source_positions_at_t'])[None],width,img_width=img_width)[0]
  d['detector_theta_image_at_t']=detector_positions_to_theta_grid(
    session['detector_position_at_t'][None],width,img_width=img_width)[0]
  d['radio_image_at_t']=radio_to_image(
    session['beam_former_outputs_at_t'][None],
    d['detector_theta_image_at_t'][None],
    session['detector_orientation_at_t'][None])[0]
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
    plot_trajectory(axs[0,0],session['detector_position_at_t'][:idx+1],width,ms=30,label='detector')
    direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*get_xy_from_theta(session['detector_orientation_at_t'][idx])

    axs[0,0].plot(
      [session['detector_position_at_t'][idx][0],direction[0,0]],
      [session['detector_position_at_t'][idx][1],direction[0,1]])

    anti_direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*get_xy_from_theta(session['detector_orientation_at_t'][idx]+np.pi/2)

    axs[0,0].plot(
      [session['detector_position_at_t'][idx][0],anti_direction[0,0]],
      [session['detector_position_at_t'][idx][1],anti_direction[0,1]])

    emitter_direction=session['detector_position_at_t'][idx]+0.25*session['width_at_t'][0]*get_xy_from_theta(session['detector_orientation_at_t'][idx]+session['source_theta_at_t'][idx,0])
    axs[0,0].plot(
      [session['detector_position_at_t'][idx][0],emitter_direction[0,0]],
      [session['detector_position_at_t'][idx][1],emitter_direction[0,1]])
    for n in np.arange(session['source_positions_at_t'].shape[1]):
      rings=(session['broadcasting_positions_at_t'][idx,n,0]==1)
      plot_trajectory(axs[0,0],session['source_positions_at_t'][:idx+1,n],width,ms=15,c='r',rings=rings,label='emitter %d' % n)

    #axs[0,0].legend()
    handles, labels = axs[0,0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0,0].legend(by_label.values(), by_label.keys())

    #lets draw the radio
    #axs[1,0].imshow(d['source_image_at_t'][idx,0].T,
    #  origin='upper',
    #  extent=(0,
    #      d['source_image_at_t'][idx,0].shape[0],
    #      d['source_image_at_t'][idx,0].shape[1],
    #      0)
    #) #,origin='lower')
    axs[1,0].imshow(d['source_image_at_t'][idx,0].T)
    axs[1,0].set_title("Emitters as image at t=%d" % idx)

    axs[1,1].imshow(
      d['radio_image_at_t'][idx,0].T)
    if invert:
      axs[0,0].invert_xaxis()
      axs[0,0].invert_yaxis()
      axs[1,0].invert_xaxis()
      axs[1,1].invert_xaxis()
    else:
      #axs[1,0].invert_yaxis()
      axs[1,1].invert_yaxis()
      #axs[1,1].invert_xaxis()
      pass
    #  origin='upper',
    #  extent=(0,
    #      d['radio_image_at_t'][idx,0].shape[0],
    #      d['radio_image_at_t'][idx,0].shape[1],
    #      0))
    #d['radio_image_at_t']=radio_to_image(session['beam_former_outputs_at_t'][None],d['detector_theta_image_at_t'][None],session['detector_orientation_at_t'][None])[0]
    #axs[1,1].imshow(d['detector_theta_image_at_t'][0,0])
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


def filenames_to_gif(filenames,output_gif_fn,size=(600,600),duration=200):
  images=[]
  for fn in filenames:
    images.append(Image.open(fn).resize(size))

  images[0].save(output_gif_fn,
           save_all = True, append_images = images[1:], 
           optimize = False, duration = duration,loop=0)
