"""Adapted from https://github.com/locuslab/TCN"""
from typing import Tuple

import torch


class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(torch.nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            torch.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(torch.nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_channels,
        kernel_size=2,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super(TCN, self).__init__()
        self.encoder = torch.nn.Embedding(output_size, input_size)
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.decoder = torch.nn.Linear(input_size, output_size)
        self.decoder.weight = self.encoder.weight
        self.drop = torch.nn.Dropout(emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y = self.tcn(emb.transpose(1, 2))
        o = self.decoder(y.transpose(1, 2))
        return o.contiguous()


class TCNWrapper(TemporalConvNet):
    """Drop-in replacement for LSTM/GRU modules"""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_first=False,
        dropout=0.0,
        kernel_size=2,
    ):
        super().__init__(
            num_inputs=input_size,
            num_channels=[hidden_size] * num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.batch_first = batch_first

    def forward(
        self, x: torch.Tensor, dummy_input=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.batch_first:
            x = x.transpose(0, 1)
        # Convolution convention is channels first so swap
        x = super().forward(x.transpose(1, 2)).transpose(1, 2)
        return x, dummy_input


def main():
    num_hidden = 128
    num_layers = 3
    batch_size = 4
    sequence_length = 12

    layer = TCNWrapper(
        input_size=num_hidden,
        hidden_size=num_hidden,
        num_layers=num_layers,
        batch_first=True,
    )
    inputs = torch.zeros(batch_size, sequence_length, num_hidden)
    outputs = layer(inputs)
    assert outputs.shape == (batch_size, sequence_length, num_hidden)


if __name__ == "__main__":
    main()
