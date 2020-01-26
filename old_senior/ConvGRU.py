from torch import nn
import torch

class ConvGRU(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1, device='cuda'):
        super().__init__()
#         self._conv_x = nn.Conv2d(in_channels=input_channel,
#                                out_channels=num_filter*3,
#                                kernel_size=kernel_size,
#                                stride=stride,
#                                padding=padding)
#         self._conv_h = nn.Conv2d(in_channels=num_filter,
#                                out_channels=num_filter*3,
#                                kernel_size=kernel_size,
#                                stride=stride,
#                                padding=padding)
        self.conv_gates = nn.Conv2d(in_channels=input_channel + num_filter,
                                    out_channels=2 * num_filter,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    bias=True)
        self.conv_can   = nn.Conv2d(in_channels=input_channel + num_filter,
                                    out_channels=num_filter,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    bias=True)
        self._leakyRelu = nn.LeakyReLU(0.2)
        self._batch_size, self._state_height, self._state_width = b_h_w
        self._input_channel = input_channel
        self._num_filter = num_filter
        self._device = device
        

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None, seq_len=6):

        if states is None:
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(self._device)
        else:
            h = states

        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float).to(self._device)
            else:
                x = inputs[index, ...]
            
#             conv_x = self._conv_x(x)
#             conv_h = self._conv_h(h)

#             xz, xr, xh = torch.chunk(conv_x, 3, dim=1)
#             hz, hr, hh = torch.chunk(conv_h, 3, dim=1)

#             z = torch.sigmoid(xz+hz)
#             r = torch.sigmoid(xr+hr)
#             nh = torch.tanh(xh+r*hh)
#             h = (1-z)*nh+z*h 
            combined = torch.cat([x, h], dim=1)
            combined_conv = self.conv_gates(combined)
            
            gamma, beta = torch.split(combined_conv, self._num_filter, dim=1)
            reset_gate = torch.sigmoid(gamma)
            update_gate = torch.sigmoid(beta)
            
            combined = torch.cat([x, reset_gate * h], dim=1)
            cc_cnm = self.conv_can(combined)
            cnm = torch.tanh(cc_cnm)
                
            h = (1 - update_gate) * h + update_gate * cnm
    
            outputs.append(h)
        
        return torch.stack(outputs), h

