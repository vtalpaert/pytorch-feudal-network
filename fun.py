from collections import namedtuple

import torch
import torch.nn as nn


def d_cos(alpha, beta, batch_size=None):
    """Cosine similarity between two vectors
    Inputs:
        vectors are size batch x d
    """
    if batch_size is None:
        batch_size = alpha.size(0)
        assert(batch_size == beta.size(0))
    out = torch.zeros(batch_size, 1)
    norms = alpha.norm(dim=1) * beta.norm(dim=1)
    for batch in range(batch_size):
        norm = norms[batch]
        if norm:
            out[batch, 0] = alpha[batch].dot(beta[batch]) / norm
        else:
            out[batch, 0] = 0
    return out


class View(nn.Module):
    """Layer changing the tensor shape
    Assumes batch first tensors
    Args:
        the output shape without providing the batch size
    """
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class LSTMCell(nn.LSTMCell):
    """Regular LSTMCell class
    If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    (Behaviour not working contrary to nn.LSTMCell documentation's promise)
    """
    def forward(self, inputs, hidden):
        if hidden is None:
            batch_size = inputs.size(0)
            hidden = (
                torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size)
            )
        return super(LSTMCell, self).forward(inputs, hidden)


class dLSTM(nn.Module):
    """Implements the dilated LSTM
    Uses a cyclic buffer of size r to keep r independent hidden states,
    but the final output is the sum on the r intermediary outputs
    """
    def __init__(self, r, input_size, hidden_size):
        super(dLSTM, self).__init__()
        self.lstm = LSTMCell(input_size, hidden_size)
        self.r = r
        self.my_filter = lambda tensor: tensor is not None

    def init_state(self):
        # TODO use one tensor instead of list, tbd when already working fine
        # TODO or use a mask over one big tensor
        hidden = [None for _ in range(self.r)]
        out = [None for _ in range(self.r)]
        tick = 0
        return tick, out, hidden

    def forward(self, inputs, states):
        tick, out, hidden = states
        out[tick], c_x = self.lstm(inputs, hidden[tick])
        hidden[tick] = out[tick], c_x
        tick = (tick + 1) % self.r
        new_states = (tick, out, hidden)
        g_t = sum(filter(self.my_filter, out))
        return g_t, new_states


'''
class dLSTMrI(dLSTM):
    """dLSTM with intrinsic reward"""
    def __init__(self, r, input_size, hidden_size):
        super(dLSTMrI, self).__init__(r, input_size, hidden_size)

    def init_state(self):
        tick, out, hidden = super(dLSTMrI, self).init_state()
        inputs = [None for _ in range(self.r)]
        return tick, out, hidden, stacked_inputs

    def forward(self, inputs):
        self.inputs[self.tick] = inputs
        return super(dLSTMrI, self).forward(inputs)

    def intrinsic_reward(self):
        t = (self.tick - 1) % self.r
        s_t = self.inputs[t]
        if s_t is None:
            raise ValueError("No recorded input")
        rI = torch.zeros(s_t.size(0), 1)
        for i in range(1, self.r):
            s_t_i = self.inputs[(t - i) % self.r]
            g_t_i = self.out[(t - i) % self.r]
            if s_t_i is not None and g_t_i is not None:
                rI += d_cos(s_t - s_t_i, g_t_i)
        return rI / self.r
'''


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class FuN(nn.Module):
    def __init__(self, observation_space, action_space, d=256, k=16, c=10):
        super(FuN, self).__init__()

        self.saved_actions = []
        self.rewards = []

        # Note that we expect input in Pytorch images' style : batch x C x H x W
        # but in gym a Box.shape for an image is (H, W, C)
        height, width, channels = observation_space.shape

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
        elif action_space.__class__.__name__ == "Box":
            raise NotImplementedError
            # we first test this code with a softmax at the end
            num_outputs = action_space.shape[0]
        elif isinstance(action_space, int):
            num_outputs = action_space
        else:
            raise NotImplementedError

        percept_linear_in = 32 * int((int((height - 4) / 4) - 2) / 2) * int((int((width - 4) / 4) - 2) / 2)
        self.f_percept = nn.Sequential(
            nn.Conv2d(channels, 16, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=2),
            nn.ReLU(),
            View((percept_linear_in,)),
            nn.Linear(percept_linear_in, d),
            nn.ReLU()
        )

        self.f_Mspace = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU()
        )

        self.f_Mrnn = dLSTM(c, d, d)

        self.f_Wrnn = LSTMCell(d, num_outputs * k)

        self.view_as_actions = View((k, num_outputs))

        self.phi = nn.Sequential(
            nn.Linear(d, k, bias=False),
            View((1, k))
        )

        self.value = nn.Linear(d, 1)

    def forward(self, x, states):
        W_hidden, M_states = states

        # perception
        z = self.f_percept(x)  # shared intermediate representation [batch x d]

        # Manager
        s = self.f_Mspace(z)  # latent state representation [batch x d]
        g, M_states = self.f_Mrnn(s, M_states)  # goal [batch x d]

        # Reset the gradient for the Worker's goal
        g_W = g.detach()
        g_W.requires_grad = g.requires_grad
        # g_W is a copy of g without the computation history

        w = self.phi(g_W)  # projection [ batch x 1 x k]

        # Worker
        U_flat, c_x = self.f_Wrnn(z, W_hidden)
        W_hidden = U_flat, c_x
        U = self.view_as_actions(U_flat)  # [batch x k x a]

        a = (w @ U).squeeze()  # [batch x a)]

        return self.value(z), a, g, (W_hidden, M_states)

    def init_state(self):
        return None, self.f_Mrnn.init_state()


def test_forward():
    from gym.spaces import Box

    batch = 4
    action_space = 6
    height = 128
    width = 128
    observation_space = Box(0, 255, [height, width, 3])
    fun = FuN(observation_space, action_space)
    states = fun.init_state()

    for i in range(10):
        image_batch = torch.randn(batch, 3, height, width)
        action, goal, states = fun(image_batch, states)
        #print(i, "th rewards are ", fun._intrinsic_reward())


if __name__ == "__main__":
    test_forward()
