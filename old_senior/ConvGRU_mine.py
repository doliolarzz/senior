import torch

class ConvGRUCell(torch.nn.Module):

  def __init__(self, input_size, hidden_size, kernel_size, t_b_c_h_w):

    super().__init__()

    self.length, self.batch_size, self.channel, self.height, self.width = t_b_c_h_w
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.kernel_size = kernel_size

    self.conv2d_X = torch.nn.Conv2d(
      in_channels=input_size,
      out_channels=hidden_size*3,
      kernel_size=kernel_size,
      padding=(kernel_size // 2, kernel_size // 2),
    )

    self.conv2d_H_prev = torch.nn.Conv2d(
      in_channels=hidden_size,
      out_channels=hidden_size*3,
      kernel_size=kernel_size,
      padding=(kernel_size // 2, kernel_size // 2),
    )

  def forward(self, X, H_prev):
    conv_inputs_X = self.conv2d_X(X)
    conv_inputs_H_prev = self.conv2d_H_prev(H_prev)

    # Z = update, R = reset, H = memory
    Z_X, R_X, _H_X = torch.split(
      conv_inputs_X, self.hidden_size, dim=1)
    Z_H_prev, R_H_prev, _H_H_prev = torch.split(
      conv_inputs_H_prev, self.hidden_size, dim=1)
    Z = torch.sigmoid(Z_X + Z_H_prev)
    R = torch.sigmoid(R_X + R_H_prev)
    _H = torch.nn.functional.relu(_H_X + R * _H_H_prev)
    H = (1 - Z) * _H + Z * H_prev

    return H

  def init_hidden(self, batch_size, cuda=True):
    state = torch.autograd.Variable(torch.zeros(batch_size, self.hidden_size, self.height, self.width))
    if cuda:
      state = state.cuda()
    return state

class ConvGRU(torch.nn.Module):

  def __init__(self, input_size, hidden_size, kernel_size, t_b_c_h_w, return_sequence=True):
    
    super().__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.kernel_size = kernel_size
    self.t_b_c_h_w = t_b_c_h_w
    self.return_sequence = return_sequence
    
    cell = ConvGRUCell(self.input_size, self.hidden_size, self.kernel_size, self.t_b_c_h_w)
    self.cells = torch.nn.ModuleList([cell])
    
  def forward(self, X, hidden_state=None):

    if hidden_state is None:
      hidden_state = self.get_init_states(batch_size=X.size(1), n_cells=X.size(0))
    else:
      hidden_state = hidden_state[-1]
    
    seq_len = self.t_b_c_h_w[0]
    layer_input = X
    
    outputs = []
    h = hidden_state
    for t in range(seq_len):
      if X is None:
        h = self.cells[0](X=torch.zeros((
          self.t_b_c_h_w[1], self.input_size, self.t_b_c_h_w[3], self.t_b_c_h_w[4])).cuda(), H_prev=h)
      else:
        h = self.cells[0](X=layer_input[t], H_prev=h)
      outputs.append(h)
    
    if not self.return_sequence:
      outputs = outputs[-1]

    return outputs, outputs

  def get_init_states(self, batch_size, n_cells, cuda=True):
#     init_states = []
#     for i in range(n_cells):
#       init_states.append(self.cells[i].init_hidden(batch_size, cuda))
#     return init_states
    return self.cells[0].init_hidden(batch_size, cuda)
    