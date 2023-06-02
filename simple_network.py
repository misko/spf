from rf_torch import SessionsDataset,SessionsDatasetSimple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
	def __init__(self,ndim):
		super().__init__()
		self.bn1 = nn.BatchNorm1d(120)
		self.bn2 = nn.BatchNorm1d(84)
		self.bn3 = nn.BatchNorm1d(2)
		self.fc1 = nn.Linear(ndim, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 2)

	def forward(self, x):
		x = x.reshape(x.shape[0],-1)
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = F.relu(self.bn3(self.fc3(x)))
		return x.reshape(x.shape[0],1,2)


if __name__=='__main__': 
	print("init dataset")
	snapshots_per_sample=8
	ds=SessionsDatasetSimple('./sessions',snapshots_in_sample=snapshots_per_sample)
	print("init dataloader")
	trainloader = torch.utils.data.DataLoader(
			ds, 
			batch_size=32,
			shuffle=True, 
			num_workers=8)
	print("get one sample for dims")
	data, label = ds[0]
	ndim_per_snapshot=data.shape[1]

	print("init network")

	nets=(
		{'name':'one snapshot','net':Net(ndim_per_snapshot), 'snapshots_per_sample':1},
		{'name':'%d snapshots' % (snapshots_per_sample//2), 'net':Net(ndim_per_snapshot*snapshots_per_sample//2), 'snapshots_per_sample':snapshots_per_sample//2},
		{'name':'%d snapshots' % snapshots_per_sample, 'net':Net(ndim_per_snapshot*snapshots_per_sample), 'snapshots_per_sample':snapshots_per_sample},
	)

	for d_net in nets:
		d_net['optimizer']=optim.Adam(d_net['net'].parameters())
	criterion = nn.MSELoss()

	print("training loop")
	print_every=200
	losses_to_plot={ d_net['name']:[] for d_net in nets }
	losses_to_plot['baseline']=[]
	for epoch in range(200):  # loop over the dataset multiple times
		running_losses={ d['name']:0.0 for d in nets }
		baseline_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			for d_net in nets:
				d_net['optimizer'].zero_grad()
				outputs = d_net['net'](inputs[:,:d_net['snapshots_per_sample']])
				loss = criterion(outputs, labels)
				loss.backward()
				running_losses[d_net['name']] += ds.args.width*np.sqrt(loss.item())
				d_net['optimizer'].step()
			baseline_loss += ds.args.width*np.sqrt(criterion(labels*0+labels.mean(), labels).item())

			if i % print_every == print_every-1:
				for d_net in nets:
					losses_to_plot[d_net['name']].append(running_losses[d_net['name']]/print_every)
				losses_to_plot['baseline'].append(baseline_loss / print_every)
				loss_str=",".join([ "%s: %0.3f" % (d_net['name'],running_losses[d_net['name']]/print_every) for d_net in nets ])
				print(f'[{epoch + 1}, {i + 1:5d}] err_in_meters {loss_str} baseline: {baseline_loss / print_every:.3f}')
				running_losses={ d['name']:0.0 for d in nets }
				baseline_loss = 0.0
				plt.clf()
				for d_net in nets:
					plt.plot(losses_to_plot[d_net['name']],label=d_net['name'])
				plt.plot(losses_to_plot['baseline'],label='baseline')
				plt.xlabel("time")
				plt.ylabel("error in m")
				plt.legend()
				plt.pause(0.001)
		
	print('Finished Training')
