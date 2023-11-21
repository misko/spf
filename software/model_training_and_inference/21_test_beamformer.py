from utils.rf import *
c=3e8 # speed of light
#carrier_frequency=2.4e9
carrier_frequency=100e3
wavelength=c/carrier_frequency
sampling_frequency=10e6

import matplotlib.pyplot as plt

'''
       X (source_pos)
       |
       |
       |
       |
----------------

Lets rotate on the source around to the left(counter clockwise) by theta radians
'''

source_pos=np.array([[0,10000]])
rotations=[-np.pi/2,-np.pi,-np.pi/4,0,np.pi/2,np.pi/2,np.pi]+list(np.random.uniform(-np.pi,np.pi,10))
spacings=[wavelength/4] #,wavelength/2,wavelength,wavelength/3]+list(np.random.uniform(0,1,10))

nelements=[2,3,4]#,16]
np.random.seed(1442)

for Detector in [ULADetector,UCADetector]:
  for nelement in nelements:
    for spacing in spacings:
      d=Detector(sampling_frequency,nelement,spacing,sigma=0.0) 

      for rot_theta in rotations:
          rot_mat=rotation_matrix(rot_theta)
          _source_pos=(rot_mat @ source_pos.T).T  # rotate right by rot_theta

          d.rm_sources()
          d.add_source(
              IQSource(
                _source_pos, # x, y position
                carrier_frequency,100e3))

          signal_matrix=d.get_signal_matrix(
              start_time=100,
              duration=3/d.sampling_frequency)

          thetas_at_t,beam_former_outputs_at_t,_=beamformer(
                  d.all_receiver_pos(),
                  signal_matrix,
                  carrier_frequency,spacing=1024+1)

          closest_angle_answer=np.argmin(np.abs(thetas_at_t-rot_theta))
          potential_error=np.abs(beam_former_outputs_at_t[:-1]-beam_former_outputs_at_t[1:]).max()
    
          assert((beam_former_outputs_at_t.max()-potential_error)<=beam_former_outputs_at_t[closest_angle_answer])

          #plt.figure()
          #plt.plot(thetas_at_t,beam_former_outputs_at_t)
          #plt.axvline(x=-rot_theta)
          #plt.axhline(y=beam_former_outputs_at_t.max()-potential_error)
          #plt.show()
print("PASS!")

