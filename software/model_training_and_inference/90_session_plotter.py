import argparse
import pickle

from utils.plot import filenames_to_gif, plot_full_session

if __name__=='__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--session', type=str, required=True)
	parser.add_argument('--step-size', type=int, required=False, default=None)
	parser.add_argument('--output_prefix', type=str, required=False, default='session_output')
	args = parser.parse_args()
	
	session=pickle.load(open(args.session,'rb'))
	if args.step_size is None:
		steps,_,_=session['source_positions_at_t'].shape
		args.step_size=max(1,steps//32)
	filenames=plot_full_session(session,args.step_size,args.output_prefix)

	filenames_to_gif(filenames,"%s.gif" % args.output_prefix)


