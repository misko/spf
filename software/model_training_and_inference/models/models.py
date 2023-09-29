import math
import os
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Tuple
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import math

class PositionalEncoding(nn.Module):

  def __init__(self, d, l):
    super().__init__()

    position = torch.arange(l).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(1,l, d)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    """
    Arguments:
      x: Tensor, shape ``[batch, time_steps, features]``
    """
    _pe=self.pe.expand((x.shape[0],self.pe.shape[1],self.pe.shape[2]))
    return torch.cat([x,_pe],axis=2)

if False:
  from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexReLU, ComplexBatchNorm1d #, ComplexSigmoid
  from complexPyTorch.complexFunctions import complex_relu
  class HybridFFNN(nn.Module):
    def __init__(self,
      d_inputs,
      d_outputs,
      n_complex_layers,
      n_real_layers,
      d_hidden,
      norm=False):
      super().__init__()
      
      self.complex_net=ComplexFFNN(
        d_inputs,
        d_hidden,
        n_layers=n_complex_layers,
        d_hidden=d_hidden,
        norm=norm
      )
      self.real_net=nn.Sequential(
        nn.Linear(d_hidden,d_hidden),
        *[nn.Sequential(
          nn.Linear(d_hidden,d_hidden),
          nn.LayerNorm(d_hidden),# if norm else nn.Identity(),
          nn.ReLU()
          )
        for _ in range(n_real_layers) ],
        nn.LayerNorm(d_hidden),# if norm else nn.Identity(),
        nn.Linear(d_hidden,d_outputs)
      )
      
    def forward(self,x):
      complex_out=self.complex_net(x)
      real_out=self.real_net(complex_out.abs())
      return F.softmax(real_out,dim=1)
      #return real_out/(real_out.sum(axis=1,keepdims=True)+1e-9)
    

  class ComplexFFNN(nn.Module):
    def __init__(self,
      d_inputs,
      d_outputs,
      n_layers,
      d_hidden,
      norm=False):
      super().__init__()
      
      self.output_net=nn.Sequential(
        ComplexBatchNorm1d(d_inputs) if norm else nn.Identity(),
        ComplexLinear(d_inputs,d_hidden),
        *[nn.Sequential(
          ComplexLinear(d_hidden,d_hidden),
          ComplexReLU(),
          ComplexBatchNorm1d(d_hidden) if norm else nn.Identity(),
          )
        for _ in range(n_layers) ],
        ComplexLinear(d_hidden,d_outputs),
      )
    def forward(self,x):
      out=self.output_net(x).abs()
      if out.sum().isnan():
        breakpoint()
        a=1
      #breakpoint()
      return F.softmax(self.output_net(x).abs(),dim=1)
      #return out/(out.sum(axis=1,keepdims=True)+1e-9)
      
      

class TransformerEncOnlyModel(nn.Module):
  def __init__(self,
      d_in, 
      d_model,
      n_heads,
      d_hid,
      n_layers,
      dropout, 
      n_outputs,
      n_layers_output=4):
    super().__init__()
    self.model_type = 'Transformer'

    encoder_layers = TransformerEncoderLayer(
      d_model, 
      n_heads, 
      d_hid, 
      dropout,
      activation='gelu',
      batch_first=True)
    self.transformer_encoder = TransformerEncoder(
      encoder_layers, 
      n_layers,
      nn.LayerNorm(d_model),
      )

    assert( d_model>=d_in)
    
    self.linear_in = nn.Linear(d_in, d_model) if d_model>d_in else nn.Identity()
    self.d_model=d_model

    self.output_net=nn.Sequential(
      nn.Linear(self.d_model,d_hid),
      nn.SELU(),
      *[SkipConnection(nn.Sequential(
        #nn.LayerNorm(d_hid),
        nn.Linear(d_hid,d_hid),
        nn.SELU()
        ))
      for _ in range(n_layers_output) ],
      #nn.LayerNorm(d_hid),
      nn.Linear(d_hid,n_outputs),
      #nn.LayerNorm(n_outputs)
      )

  def forward(self, src: Tensor, src_key_padding_mask=None) -> Tensor:
    #src_enc = self.transformer_encoder(
    #  torch.cat(
    #    [src,self.linear_in(src)],axis=2)) #/np.sqrt(self.d_radio_feature))
    src_enc = self.transformer_encoder( self.linear_in(src), src_key_padding_mask=src_key_padding_mask)
    #output = self.transformer_encoder(src) #,self.linear_in(src)],axis=2)) #/np.sqrt(self.d_radio_feature))
    output = self.output_net(src_enc) #/np.sqrt(self.d_model)
    if output.isnan().any():
      breakpoint()
    return output


class SkipConnection(nn.Module):
  def __init__(self,module):
    super().__init__()
    self.module=module

  def forward(self,x):
    return self.module(x)+x

class FilterNet(nn.Module):
  def __init__(self,
      d_drone_state=4+4,
      d_emitter_state=4+4+2, # 2 means , 2 variances, 2 angles
      d_radio_feature=258,
      d_model=512,
      n_heads=8,
      d_hid=256,
      d_embed=64,
      n_layers=1,
      n_outputs=4+4+2+1, # 2 means , 2 variances, 2 angles, responsibility
      dropout=0.0,
      ssn_d_hid=64,
      ssn_n_layers=8,
      ssn_n_outputs=8,
      ssn_dropout=0.0,
      tformer_input=['drone_state','embedding']):
    super().__init__()
    self.d_radio_feature=d_radio_feature
    self.d_drone_state=d_drone_state
    self.d_model=d_model
    self.n_heads=n_heads
    self.d_embed=d_embed
    self.d_hid=d_hid
    self.dropout=dropout
    self.tformer_input=tformer_input
    self.tformer_input_dim=0
    self.n_layers=n_layers
    self.n_outputs=n_outputs


    for k,v in [
      ('drone_state',d_drone_state),
      ('radio_feature',d_radio_feature),
      ('embedding',self.d_embed)]:
      if k in self.tformer_input:
        self.tformer_input_dim+=v
    
    self.snap_shot_net=SingleSnapshotNet(
      d_input_feature=d_radio_feature+d_drone_state,
      d_hid=ssn_d_hid,
      d_embed=self.d_embed-1,
      n_layers=ssn_n_layers,
      n_outputs=n_outputs,
      dropout=ssn_dropout)

    self.transformer_enc=TransformerEncOnlyModel(
      d_in=self.d_model, 
      d_model=self.d_model,
      n_heads=self.n_heads,
      d_hid=self.d_hid,
      n_layers=self.n_layers,
      dropout=self.dropout, 
      n_outputs=self.n_outputs,
      n_layers_output=4)

  def forward(self,x):
    single_snapshot_output=self.snap_shot_net(x)
    snap_shot_embeddings=single_snapshot_output['embedding'] #torch.Size([batch, time, embedding dim])
    emitter_state_embeddings=self.emitter_embedding_net(x)['emitter_state_embedding']

    batch_size,time_steps,n_sources,_=emitter_state_embeddings.shape

    snap_shot_embeddings=torch.cat([
      snap_shot_embeddings, #torch.Size([batch, time, embedding dim])
      torch.zeros(batch_size,time_steps,1).to(snap_shot_embeddings.device)],dim=2).reshape(batch_size,time_steps,1,self.d_embed)
    emitter_state_embeddings=torch.cat([
      emitter_state_embeddings,
      torch.ones(batch_size,time_steps,n_sources,1).to(emitter_state_embeddings.device)],dim=3)


    self_attention_input=torch.cat([emitter_state_embeddings,snap_shot_embeddings],dim=2)
    #input shape = (batch*time, nsources+1snapshot, embedding_dim)
    #output shape = (batch*time,N,embedding_dim)

    self_attention_output=self.transformer_enc(
      self_attention_input.reshape(
        batch_size*time_steps,
        n_sources+1,
        self.d_model)).reshape(batch_size,time_steps,n_sources+1,self.n_outputs)
    #make sure assignments are probabilities?
    self_attention_output=torch.cat([self_attention_output[...,:-1],nn.functional.softmax(self_attention_output[...,[-1]],dim=2)],dim=3)
    return {'transformer_pred':self_attention_output,
      'single_snapshot_pred':single_snapshot_output['single_snapshot_pred']}

class TrajectoryNet(nn.Module):
  def __init__(self,
      d_drone_state=4+4,
      d_radio_feature=258,
      d_detector_observation_embedding=128,
      d_trajectory_embedding=256,
      trajectory_prediction_n_layers=8,
      d_trajectory_prediction_output=(2+2+1)+(2+2+1),
      d_model=512,
      n_heads=8,
      d_hid=256,
      n_layers=4,
      n_outputs=8,
      ssn_d_hid=64,
      ssn_n_layers=8,
      ssn_d_output=5,
      ):
    super().__init__()
    self.d_detector_observation_embedding=d_detector_observation_embedding
    self.d_trajectory_embedding=d_trajectory_embedding
    self.d_trajectory_prediction_output=d_trajectory_prediction_output

    self.snap_shot_net=EmbeddingNet(
        d_in=d_radio_feature+d_drone_state, # time is inside drone state
        d_hid=ssn_d_hid,
        d_out=ssn_d_output,
        d_embed=d_detector_observation_embedding,
        n_layers=ssn_n_layers
      )

    self.trajectory_prediction_net=EmbeddingNet(
        d_in=d_trajectory_embedding+1, 
        d_hid=d_trajectory_embedding,
        d_out=d_trajectory_prediction_output,
        d_embed=d_trajectory_embedding,
        n_layers=trajectory_prediction_n_layers
      )
    
    self.tformer=TransformerEncOnlyModel(
      d_in=d_detector_observation_embedding,
      d_model=d_model,
      n_heads=n_heads,
      d_hid=d_hid,
      n_layers=n_layers,
      dropout=0.0,
      n_outputs=d_trajectory_embedding)


  def forward(self,x):
    # Lets get the detector observation embeddings
    batch_size,time_steps,n_emitters,_=x['emitters_n_broadcasts'].shape
    device=x['emitters_n_broadcasts'].device

    ###############
    # SINGLE SNAPSHOT PREDICTIONS + EMBEDDINGS
    ###############
    d=self.snap_shot_net(
      torch.cat([ # drone state + radio feature as input to SSN
        x['drone_state'],
        x['radio_feature']
        ],dim=2)
    )
    drone_state_and_observation_embeddings=d['embedding']
    single_snapshot_predictions=d['output']


    ################
    # TRAJECTORY EMBEDDINGS 
    #pick random time for each batch item, crop session, and compute tracjectory emebedding for each tracked object
    ################

    #generate random times to grab
    rt=torch.randint(low=2, high=time_steps-1, size=(batch_size,)) # keep on CPU?, low=2 here gaurantees at least one thing was being tracked for each example
    #now lets grab the (nsources,1) vector for each batch example that tells us how many times the object has transmitted previously
    tracking=x['emitters_n_broadcasts'][torch.arange(batch_size),rt-1].cpu() #positive values for things already being tracked

    #find out how many objects are being tracked in each batch example
    # so that we can pull out their embeddings to calculate trajectory embeddings
    nested_tensors=[]
    idxs=[] # List of batch idx and idx of object being tracked
    max_snapshots=0
    for b in torch.arange(batch_size):
      t=rt[b]
      for tracked in torch.where(tracking[b]>0)[0]: 
        # get the mask where this emitter is broadcasting in the first t steps
        times_where_this_tracked_is_broadcasting=x['emitters_broadcasting'][b,:t,tracked,0].to(bool)
        # pull the drone state and observations for these time steps
        nested_tensors.append(
          drone_state_and_observation_embeddings[b,:t][times_where_this_tracked_is_broadcasting]
        )
        assert(nested_tensors[-1].shape[0]!=0)
        max_snapshots=max(max_snapshots,nested_tensors[-1].shape[0]) # efficiency, what is the biggest we need to compute
        idxs.append((b.item(),tracked))

    #move all the drone+observation sequences into a common tensor with padding
    tformer_input=torch.zeros((len(idxs),max_snapshots,self.d_detector_observation_embedding),device=device)
    src_key_padding_mask=torch.zeros((len(idxs),max_snapshots),dtype=bool) # TODO ones and mask is faster?
    for idx,emebeddings_per_batch_and_tracked in enumerate(nested_tensors):
      tracked_time_steps,_=emebeddings_per_batch_and_tracked.shape
      tformer_input[idx,:tracked_time_steps]=emebeddings_per_batch_and_tracked  
      src_key_padding_mask[idx,tracked_time_steps:]=True
    src_key_padding_mask=src_key_padding_mask.to(device) #TODO initialize on device?

    #run the self attention layer
    self_attention_output=self.tformer(tformer_input,src_key_padding_mask=src_key_padding_mask)

    #select or aggregate the emebedding to use from the historical options
    trajectory_embeddings=torch.zeros((batch_size,n_emitters,self.d_trajectory_embedding),device=device)
    for idx,(b,emitter_idx) in enumerate(idxs):
      #trajectory_embeddings[b,emitter_idx]=self_attention_output[idx,~src_key_padding_mask[idx]].mean(axis=0)
      trajectory_embeddings[b,emitter_idx]=self_attention_output[idx,~src_key_padding_mask[idx]][-1]
 

    #######
    # TRAJECTORY PREDICTIONS
    #######
    time_fractions=torch.linspace(0,1,time_steps,device=device)[...,None] # TODO add an assert to make sure this does not go out of sync with other format
    trajectory_predictions=torch.zeros((batch_size,n_emitters,time_steps,self.d_trajectory_prediction_output),device=device)
    for b in torch.arange(batch_size):
      for emitter_idx in torch.arange(n_emitters):
        trajectory_input=torch.cat([
          trajectory_embeddings[b,emitter_idx][None].expand((time_steps,self.d_trajectory_embedding)),
          time_fractions,
          #TODO ADD DRONE EMBEDDING!
        ],axis=1)
        trajectory_predictions[b,emitter_idx]=self.trajectory_prediction_net(trajectory_input)['output']

    return {
      'single_snapshot_predictions':single_snapshot_predictions,
      #'trajectory_embeddings':trajectory_embeddings,
      'trajectory_predictions':trajectory_predictions.transpose(2,1),
      }


class EmbeddingNet(nn.Module):
  def __init__(self,
      d_in,
      d_out,
      d_hid,
      d_embed,
      n_layers):
    super(EmbeddingNet,self).__init__()
    
    self.embed_net=nn.Sequential(
      nn.Linear(d_in,d_hid),
      nn.SELU(),
      *[SkipConnection(
        nn.Sequential(
        nn.LayerNorm(d_hid),
        nn.Linear(d_hid,d_hid),
        #nn.ReLU()
        nn.SELU()
        ))
      for _ in range(n_layers) ],
      #nn.LayerNorm(d_hid),
      nn.Linear(d_hid,d_embed),
      nn.LayerNorm(d_embed),
      )
    self.lin_output=nn.Linear(d_embed,d_out)

  def forward(self, x):
    embed=self.embed_net(x)
    output=self.lin_output(embed)

    if output.isnan().any():
      breakpoint()
    return {'embedding':embed,
      'output':output}

class SnapshotNet(nn.Module):
  def __init__(self,
      snapshots_per_sample=1,
      d_drone_state=4+4,
      d_radio_feature=258,
      d_model=512,
      n_heads=8,
      d_hid=256,
      n_layers=1,
      n_outputs=8,
      dropout=0.0,
      ssn_d_hid=64,
      ssn_n_layers=8,
      ssn_n_outputs=8,
      ssn_d_embed=64,
      ssn_dropout=0.0,
      tformer_input=['drone_state','radio_feature','embedding','single_snapshot_pred'],
      positional_encoding_len=0):
    super().__init__()
    self.d_radio_feature=d_radio_feature
    self.d_drone_state=d_drone_state
    self.d_model=d_model
    self.n_heads=n_heads
    self.d_hid=d_hid
    self.n_outputs=n_outputs
    self.dropout=dropout
    self.tformer_input=tformer_input
    self.tformer_input_dim=0
    for k,v in [
      ('drone_state',d_drone_state),
      ('radio_feature',d_radio_feature),
      ('embedding',ssn_d_embed),
      ('single_snapshot_pred',n_outputs)]:
      if k in self.tformer_input:
        self.tformer_input_dim+=v
    self.tformer_input_dim+=positional_encoding_len
    
    
    self.snap_shot_net=SingleSnapshotNet(
      d_input_feature=d_radio_feature+d_drone_state,
      d_hid=ssn_d_hid,
      d_embed=ssn_d_embed,
      n_layers=ssn_n_layers,
      n_outputs=ssn_n_outputs,
      dropout=ssn_dropout)
    #self.snap_shot_net=Task1Net(d_radio_feature*snapshots_per_sample)

    self.tformer=TransformerEncOnlyModel(
      d_in=self.tformer_input_dim,
      #d_radio_feature=ssn_d_embed, #+n_outputs,
      d_model=d_model,
      n_heads=n_heads,
      d_hid=d_hid,
      n_layers=n_layers,
      dropout=dropout,
      n_outputs=n_outputs)

    self.positional_encoding=nn.Identity()
    if positional_encoding_len>0:
      self.positional_encoding=PositionalEncoding(positional_encoding_len,snapshots_per_sample)

  def forward(self,x):
    d=self.snap_shot_net(x)
    #return single_snapshot_output,single_snapshot_output
    tformer_input=self.positional_encoding(torch.cat([d[t] for t in self.tformer_input ],axis=2))
    tformer_output=self.tformer(tformer_input)
    return {'transformer_pred':tformer_output,
      'single_snapshot_pred':d['single_snapshot_pred']}

class SingleSnapshotNet(nn.Module):
  def __init__(self,
      d_input_feature,
      d_hid,
      d_embed,
      n_layers,
      n_outputs,
      dropout,
      snapshots_per_sample=0):
    super(SingleSnapshotNet,self).__init__()
    self.snapshots_per_sample=snapshots_per_sample
    self.d_input_feature=d_input_feature
    if self.snapshots_per_sample>0:
      self.d_input_feature*=snapshots_per_sample
    self.d_hid=d_hid
    self.d_embed=d_embed
    self.n_layers=n_layers
    self.n_outputs=n_outputs
    self.dropout=dropout
    
    self.embed_net=nn.Sequential(
      #nn.LayerNorm(self.d_radio_feature),
      nn.Linear(self.d_input_feature,d_hid),
      nn.SELU(),
      *[SkipConnection(
        nn.Sequential(
        nn.LayerNorm(d_hid),
        nn.Linear(d_hid,d_hid),
        #nn.ReLU()
        nn.SELU()
        ))
      for _ in range(n_layers) ],
      #nn.LayerNorm(d_hid),
      nn.Linear(d_hid,d_embed),
      nn.LayerNorm(d_embed)
      )
    self.lin_output=nn.Linear(d_embed,self.n_outputs)

  def forward(self, x_dict):
    x=torch.cat([x_dict['drone_state'],x_dict['radio_feature']],axis=2)
    if self.snapshots_per_sample>0:
      x=x.reshape(x.shape[0],-1)
    embed=self.embed_net(x)
    output=self.lin_output(embed)
    if output.isnan().any():
      breakpoint()
    if self.snapshots_per_sample>0:
      output=output.reshape(-1,1,self.n_outputs)
      return {'fc_pred':output}
    return {'drone_state':x_dict['drone_state'],
      'radio_feature':x_dict['radio_feature'],
      'single_snapshot_pred':output,'embedding':embed}

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
