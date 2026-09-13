import json

import torchaudio

import IPython

from torchaudio import transforms

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import math
import torch.nn as nn
import torch

from model_scripts.pann14 import Cnn14

class Cnn14Frontend(nn.Module):
    def __init__(self, pretrained_model, cutoff_block=4):
        super().__init__()
        self.pre_trained_model = pretrained_model
        self.spec_augmenter = self.pre_trained_model.spec_augmenter
        self.spectrogram_extractor = self.pre_trained_model.spectrogram_extractor
        self.logmel_extractor = self.pre_trained_model.logmel_extractor
        self.bn0 = self.pre_trained_model.bn0
        self.blocks = nn.ModuleList([
            getattr(self.pre_trained_model, f"conv_block{i}") for i in range(1, cutoff_block + 1)
        ])

    def forward(self, waveform, is_training=True):
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if is_training:
            x = self.spec_augmenter(x)
        for block in self.blocks:
            x = block(x, pool_size=(2, 2), pool_type="avg")
        return x  # shape: (B, C, F', T') — feed this into CBAM


class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max_val = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max_val,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max_pool = F.adaptive_max_pool1d(x, output_size=1)
        # print(max_pool.shape)
        avg = F.adaptive_avg_pool1d(x, output_size=1)
        # print(avg.shape)
        b, c, _ = x.size()
        linear_max = self.linear(max_pool.view(b,c)).view(b, c, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x


class EncoderBlock(nn.Module):

    def __init__(self, cbam_channels, r, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads)

        # CBAM layer

        self.cbam = CBAM(cbam_channels, r)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        #CBAM part

        x = x.transpose(1, 2)
        x = self.cbam(x)
        x = x.transpose(1, 2)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class PANN_Transformer_CBAM_Gate(nn.Module):
    def __init__(self, pre_trained_model, cbam_channels, r, num_layers, input_dim, num_heads, dim_feedforward, out_dim, num_classes, dropout=0.0):
        super().__init__()
        self.pann = Cnn14Frontend(pre_trained_model)
        self.encoder = TransformerEncoder(num_layers=num_layers, cbam_channels=cbam_channels, r = r,
                                          input_dim=input_dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, waveform, is_training=True):
        waveform = torch.flatten(waveform, start_dim=1)
        x = self.pann(waveform, is_training)
        x = x.mean(dim=3)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def load_pann_transformer_cbam_gate(pre_trained_model_path, configuration):
    PANN = Cnn14(sample_rate=16000, window_size=1024, hop_size=320,
              mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    checkpoint = torch.load("pre_trained_models/Cnn14_mAP=0.431.pth", map_location="cpu")
    PANN.load_state_dict(checkpoint["model"])
    for block_name in ["conv_block1", "conv_block2", "conv_block3"]:
        block = getattr(PANN, block_name)
        for param in block.parameters():
            param.requires_grad = False

    with open(configuration) as f:
        config = json.load(f)


    model = PANN_Transformer_CBAM_Gate(pre_trained_model=PANN, **config)

    checkpoint = torch.load(pre_trained_model_path, map_location="cpu")
    model.load_state_dict(checkpoint)

    return model