from rf_torch import SessionsDataset,SessionsDatasetSimple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
	ds=SessionsDatasetSimple('./sessions')

	trainloader = torch.utils.data.DataLoader(
			ds, 
			batch_size=32,
			shuffle=True, 
			num_workers=8)

	data, labels = next(iter(trainloader))
	data=data.reshape(data.shape[0],-1)

	net = Net(data.shape[1])

	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters())

	for epoch in range(200):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 200 == 199:	# print every 2000 mini-batches
				print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
				running_loss = 0.0

	print('Finished Training')
