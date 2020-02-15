from torch import nn
import torch

class ConvLSTM(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1, config=None):
        super().__init__()
        self.config = config
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self._batch_norm = nn.BatchNorm2d(num_filter*4)
        self._batch_size, self._state_height, self._state_width = b_h_w
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(self.config['DEVICE'])
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(self.config['DEVICE'])
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(self.config['DEVICE'])
        self._input_channel = input_channel
        self._num_filter = num_filter

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None, seq_len=None):

        if seq_len is None:
            seq_len = self.config['IN_LEN']

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(self.config['DEVICE'])
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(self.config['DEVICE'])
        else:
            h, c = states

        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float).to(self.config['DEVICE'])
            else:
                x = inputs[index, ...]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)
            conv_x = self._batch_norm(conv_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

