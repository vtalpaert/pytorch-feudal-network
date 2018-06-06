import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        print("Flatten", x.size())
        return x.view(x.size()[0], **self.shape)


class dLSTM(nn.Module):
    def __init__(self, r, input_size, hidden_size, num_layers=1):
        super(dLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.hidden = [None for _ in range(r)]
        self.out = [None for _ in range(r)]
        self.tick = 0
        self.r = r
        self.my_filter = lambda tensor: tensor is not None

    def forward(self, inputs):
        self.out[self.tick], self.hidden[self.tick] = self.lstm(inputs, self.hidden[self.tick])
        self.tick = (self.tick + 1) % self.r
        return sum(filter(self.my_filter, self.out))


class FuN(nn.Module):
    def __init__(self, in_channels, action_space, d=256, k=16, c=10):
        super(FuN, self).__init__()

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

        self.f_percept = nn.Sequential(
            nn.Conv2d(in_channels, 16, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=2),
            nn.ReLU(),
            View((-1,)),
            nn.Linear(32 * 4 * 4, d),  # TODO: input layer
            nn.ReLU()
        )

        self.f_Mspace = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU()
        )

        self.f_Mrnn = dLSTM(c, d, d)
        self.f_Wrnn = nn.LSTM(d, num_outputs * k)
        self.W_hidden = None  # worker's hidden state

        self.stack_to_action = View((num_outputs, k))

        self.phi = nn.Linear(d, k, bias=False)

    def forward(self, x):
        ## perception
        # shared intermediate representation [d]
        z = self.f_percept(x)

        ## Manager
        # latent state representation [d]
        s = self.f_Mspace(z)
        # goal [d]
        g = self.f_Mrnn(s)

        # Reset the gradient for the Worker's goal
        g_W = g.detach()
        g_W.requires_grad = g.requires_grad
        # g_W is a copy of g without the computation history

        # projection [k x 1]
        w = self.phi(g_W)

        # Worker
        U_flat, self.W_hidden = self.f_Wrnn(z, self.W_hidden)
        U = self.stack_to_action(U_flat)  # [n x a x k)]

        a = (U @ w)
        a = a.view(a.size()[0])  # action [a]

        return F.softmax(a)
