import math
import os
from collections import OrderedDict
from functools import cache
from tempfile import TemporaryDirectory
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class TransformerModel(nn.Module):
	def __init__(self,
			d_radio_feature, 
			d_model,
			n_heads,
			d_hid,
			n_layers,
			dropout, 
			n_outputs):
		super().__init__()
		self.model_type = 'Transformer'

		encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
		
		self.linear_out = nn.Linear(d_model, n_outputs)
		assert( d_model>d_radio_feature)
		
		self.linear_in = nn.Linear(d_radio_feature, d_model-d_radio_feature)
		
		self.d_model=d_model

	def forward(self, src: Tensor) -> Tensor:
		output = self.transformer_encoder(
			torch.cat(
				[src,self.linear_in(src)],axis=2)) #/np.sqrt(self.d_radio_feature))
		output = self.linear_out(output)/np.sqrt(self.d_model)
		return output

class SnapshotNet(nn.Module):
	def __init__(self,
			snapshots_per_sample=1,
			d_radio_feature=257+4+4,
			d_model=512,
			n_heads=8,
			d_hid=256,
			n_layers=4,
			n_outputs=8,
			dropout=0.0,
			ssn_d_hid=128,
			ssn_n_layers=4,
			ssn_n_outputs=8,
			ssn_d_embed=64,
			ssn_dropout=0.0):
		super().__init__()
		self.d_radio_feature=d_radio_feature
		self.d_model=d_model
		self.n_heads=n_heads
		self.d_hid=d_hid
		self.n_outputs=n_outputs
		self.dropout=dropout

		self.snap_shot_net=SingleSnapshotNet(
			d_radio_feature=d_radio_feature,
			d_hid=ssn_d_hid,
			d_embed=ssn_d_embed,
			n_layers=ssn_n_layers,
			n_outputs=ssn_n_outputs,
			dropout=ssn_dropout)
		#self.snap_shot_net=Task1Net(d_radio_feature*snapshots_per_sample)

		self.tformer=TransformerModel(
			d_radio_feature=d_radio_feature+ssn_d_embed+n_outputs,
			d_model=d_model,
			n_heads=n_heads,
			d_hid=d_hid,
			n_layers=n_layers,
			dropout=dropout,
			n_outputs=n_outputs)

	def forward(self,x):
		single_snapshot_output,embed=self.snap_shot_net(x)
		d=self.snap_shot_net(x)
		#return single_snapshot_output,single_snapshot_output
		tformer_output=self.tformer(
			torch.cat([
				x,
				d['embedding'],
				d['single_snapshot_pred']
				],axis=2))
		return {'transformer_pred':tformer_output,'single_snapshot_pred':d['single_snapshot_pred']}
		

class SingleSnapshotNet(nn.Module):
	def __init__(self,
			d_radio_feature,
			d_hid,
			d_embed,
			n_layers,
			n_outputs,
			dropout,
			snapshots_per_sample=0):
		super(SingleSnapshotNet,self).__init__()
		self.snapshots_per_sample=snapshots_per_sample
		self.d_radio_feature=d_radio_feature
		if self.snapshots_per_sample>0:
			self.d_radio_feature*=snapshots_per_sample
		self.d_hid=d_hid
		self.d_embed=d_embed
		self.n_layers=n_layers
		self.n_outputs=n_outputs
		self.dropout=dropout
		
		self.embed_net=nn.Sequential(
			nn.Linear(self.d_radio_feature,d_hid),
			*[nn.Sequential(
				nn.LayerNorm(d_hid),
				nn.Linear(d_hid,d_hid),
				nn.ReLU()
				)
			for _ in range(n_layers) ],
			nn.LayerNorm(d_hid),
			nn.Linear(d_hid,d_embed),
			nn.LayerNorm(d_embed))
		self.lin_output=nn.Linear(d_embed,self.n_outputs)

	def forward(self, x):
		if self.snapshots_per_sample>0:
			x=x.reshape(x.shape[0],-1)
		embed=self.embed_net(x)
		output=self.lin_output(embed)
		if self.snapshots_per_sample>0:
			output=output.reshape(-1,1,self.n_outputs)
			return {'fc_pred':output}
		return {'single_snapshot_pred':output,'embedding':embed}

class Task1Net(nn.Module):
	def __init__(self,ndim,n_outputs=8):
		super().__init__()
		self.bn1 = nn.BatchNorm1d(120)
		self.bn2 = nn.BatchNorm1d(84)
		self.bn3 = nn.BatchNorm1d(n_outputs)
		self.fc1 = nn.Linear(ndim, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, n_outputs)
		self.n_outputs=n_outputs
		self.ndim=ndim

	def forward(self, x):
		x = x.reshape(x.shape[0],-1)
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = F.relu(self.bn3(self.fc3(x)))
		x = x.reshape(-1,1,self.n_outputs)
		return {'fc_pred':x} #.reshape(x.shape[0],1,2)


class UNet(nn.Module):

	def __init__(self, in_channels=3, out_channels=1, width=128, init_features=32):
		super(UNet, self).__init__()

		features = init_features
		self.encoder1 = UNet._block(in_channels, features, name="enc1")
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.encoder2 = UNet._block(features, features * 2, name="enc2")
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.encoder5 = UNet._block(features * 8, features * 16, name="enc4")
		self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.bottleneck = UNet._block(features * 16, features * 32, name="bottleneck")
		#self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

		self.upconv5 = nn.ConvTranspose2d(
			features * 32, features * 16, kernel_size=2, stride=2
		)
		self.decoder5 = UNet._block((features * 16) * 2, features * 16, name="dec5")
		self.upconv4 = nn.ConvTranspose2d(
			features * 16, features * 8, kernel_size=2, stride=2
		)
		self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
		self.upconv3 = nn.ConvTranspose2d(
			features * 8, features * 4, kernel_size=2, stride=2
		)
		self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
		self.upconv2 = nn.ConvTranspose2d(
			features * 4, features * 2, kernel_size=2, stride=2
		)
		self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
		self.upconv1 = nn.ConvTranspose2d(
			features * 2, features, kernel_size=2, stride=2
		)
		self.decoder1 = UNet._block(features * 2, features, name="dec1")

		self.conv = nn.Conv2d(
			in_channels=features, out_channels=out_channels, kernel_size=1
		)

	def forward(self, x):
		enc1 = self.encoder1(x)
		enc2 = self.encoder2(self.pool1(enc1))
		enc3 = self.encoder3(self.pool2(enc2))
		enc4 = self.encoder4(self.pool3(enc3))
		enc5 = self.encoder5(self.pool4(enc4))

		bottleneck = self.bottleneck(self.pool5(enc5))
		#bottleneck = self.bottleneck(self.pool4(enc4))

		dec5 = self.upconv5(bottleneck)
		dec5 = torch.cat((dec5, enc5), dim=1)
		dec5 = self.decoder5(dec5)
		dec4 = self.upconv4(dec5)
		dec4 = torch.cat((dec4, enc4), dim=1)
		dec4 = self.decoder4(dec4)
		dec3 = self.upconv3(dec4)
		dec3 = torch.cat((dec3, enc3), dim=1)
		dec3 = self.decoder3(dec3)
		dec2 = self.upconv2(dec3)
		dec2 = torch.cat((dec2, enc2), dim=1)
		dec2 = self.decoder2(dec2)
		dec1 = self.upconv1(dec2)
		dec1 = torch.cat((dec1, enc1), dim=1)
		dec1 = self.decoder1(dec1)
		out=torch.sigmoid(self.conv(dec1))
		return {'image_preds':out/out.sum(axis=[2,3],keepdims=True)}

	@staticmethod
	def _block(in_channels, features, name):
		return nn.Sequential(
			OrderedDict(
				[
					(
						name + "conv1",
						nn.Conv2d(
							in_channels=in_channels,
							out_channels=features,
							kernel_size=3,
							padding=1,
							bias=False,
						),
					),
					(name + "norm1", nn.BatchNorm2d(num_features=features)),
					(name + "relu1", nn.ReLU(inplace=True)),
					(
						name + "conv2",
						nn.Conv2d(
							in_channels=features,
							out_channels=features,
							kernel_size=3,
							padding=1,
							bias=False,
						),
					),
					(name + "norm2", nn.BatchNorm2d(num_features=features)),
					(name + "relu2", nn.ReLU(inplace=True)),
				]
			)
		)