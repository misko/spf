from utils.rf import *
c=3e8 # speed of light
#carrier_frequency=2.4e9
carrier_frequency=1000e3
wavelength=c/carrier_frequency
sampling_frequency=10e6


'''
       X (source_pos)
       |
       |
       |
       |
----------------

Lets rotate everything around to the left (counter clockwise) by theta radians
Then we should have identical signal matricies and beamformer outputs
'''

source_pos=np.array([[0,10]])
rotations=[-np.pi,-np.pi/2,-np.pi/4,0,np.pi/2,np.pi/2,np.pi]+list(np.random.uniform(0,np.pi*2,10))
spacings=[wavelength,wavelength/2,wavelength/4,wavelength/3]+list(np.random.uniform(0,1,10))
nelements=[2,3,4,16]
np.random.seed(1442)

for nelement in nelements:
  for spacing in spacings:
    d=ULADetector(sampling_frequency,nelement,spacing,sigma=0.0) 
    signal_matrixs=[]
    beamformer_outputs=[]

    for rot_theta in rotations:
        rot_mat=rotation_matrix(rot_theta)
        _source_pos=(rot_mat @ source_pos.T).T  # rotate left by rot_theta

        d.rm_sources()
        d.add_source(
            IQSource(
              _source_pos, # x, y position
              carrier_frequency,100e3))
        d.orientation=rot_theta
        signal_matrix=d.get_signal_matrix(
            start_time=100,
            duration=3/d.sampling_frequency)
        signal_matrixs.append(signal_matrix)
        assert(np.isclose(signal_matrixs[0],signal_matrixs[-1]).all())
        thetas_at_t,beam_former_outputs_at_t,_=beamformer(
                d.all_receiver_pos(),
                signal_matrix,
                carrier_frequency,
                offset=-d.orientation)
        beamformer_outputs.append(beam_former_outputs_at_t)
        #plt.figure()
        #plt.plot(thetas_at_t,beamformer_outputs[0])
        #plt.plot(thetas_at_t,beamformer_outputs[-1])
        #plt.show()
        #print(beamformer_outputs[0],beamformer_outputs[-1])
        assert(np.isclose(beamformer_outputs[0],beamformer_outputs[-1]).all())
    #plt.plot(thetas_at_t,beamformer_outputs[0])
    #plt.axvline(x=0,color='red')
    #plt.show()
print("PASS!")

