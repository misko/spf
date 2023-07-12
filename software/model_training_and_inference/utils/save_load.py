import pickle
import io
import torch

#borrowed from online
class CPU_Unpickler(pickle.Unpickler):
	def find_class(self, module, name):
		if module == 'torch.storage' and name == '_load_from_bytes':
			return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
		else:
			return super().find_class(module, name)
