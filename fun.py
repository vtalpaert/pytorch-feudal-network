import torch
import torch.nn as nn
import torch.nn.functional as F


def d_cos(alpha, beta, batch_size=None):
    """inputs are size batch x 1 x d"""
    if batch_size is None:
        batch_size = alpha.size(0)
        assert(batch_size == beta.size(0))
    out = torch.zeros(batch_size, 1)
    norms = alpha.norm(dim=2) * beta.norm(dim=2)
    for batch in range(batch_size):
        norm = norms[batch]
        if norm:
            out[batch, 0] = alpha[batch, 0].dot(beta[batch, 0]) / norm
        else:
            out[batch, 0] = 0
    return out


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size()[0], *self.shape)


class dLSTM(nn.Module):
    def __init__(self, r, input_size, hidden_size):
        super(dLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.r = r
        self.my_filter = lambda tensor: tensor is not None
        self.reset()

    def reset(self):
        # TODO use one tensor instead of list, tbd when already working fine
        self.hidden = [None for _ in range(self.r)]
        self.out = [None for _ in range(self.r)]
        self.tick = 0

    def forward(self, inputs):
        self.out[self.tick], self.hidden[self.tick] = self.lstm(inputs, self.hidden[self.tick])
        self.tick = (self.tick + 1) % self.r
        return sum(filter(self.my_filter, self.out))


class dLSTMrI(dLSTM):
    """dLSTM with intrinsic reward"""
    def __init__(self, r, input_size, hidden_size):
        super(dLSTMrI, self).__init__(r, input_size, hidden_size)

    def reset(self):
        super(dLSTMrI, self).reset()
        self.input = [None for _ in range(self.r)]

    def forward(self, inputs):
        self.input[self.tick] = inputs
        return super(dLSTMrI, self).forward(inputs)

    def intrinsic_reward(self):
        t = (self.tick - 1) % self.r
        rI = 0
        s_t = self.input[t]
        if s_t is None:
            print("No recorded input")
            return 0
        for i in range(1, self.r):
            s_t_i = self.input[(t - i) % self.r]
            g_t_i = self.out[(t - i) % self.r]
            if s_t_i is not None and g_t_i is not None:
                rI += d_cos(s_t - s_t_i, g_t_i)
        return rI / self.r


class FuN(nn.Module):
    def __init__(self, observation_space, action_space, d=256, k=16, c=10):
        super(FuN, self).__init__()

        # Note that in Pytorch images are : batch x C x H x W
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
            View((1, percept_linear_in,)),
            nn.Linear(percept_linear_in, d),
            nn.ReLU()
        )

        self.f_Mspace = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU()
        )

        self.f_Mrnn = dLSTMrI(c, d, d)

        self.f_Wrnn = nn.LSTM(d, num_outputs * k, batch_first=True)
        self.W_hidden = None  # worker's hidden state

        self.view_as_actions = View((k, num_outputs))

        self.phi = nn.Linear(d, k, bias=False)

    def forward(self, x):
        # perception
        z = self.f_percept(x)  # shared intermediate representation [batch x d]

        # Manager
        s = self.f_Mspace(z)  # latent state representation [batch x 1 x d]
        g = self.f_Mrnn(s)  # goal [batch x 1 x d]

        # Reset the gradient for the Worker's goal
        g_W = g.detach()
        g_W.requires_grad = g.requires_grad
        # g_W is a copy of g without the computation history

        w = self.phi(g_W)  # projection [ batch x 1 x k]

        # Worker
        U_flat, self.W_hidden = self.f_Wrnn(z, self.W_hidden)
        U = self.view_as_actions(U_flat)  # [batch x k x a]

        a = (w @ U)  # [batch x 1 x a)]

        return F.softmax(a, dim=2), g

    def _intrinsic_reward(self):
        return self.f_Mrnn.intrinsic_reward()


def test_forward():
    from gym.spaces import Box

    batch = 4
    action_space = 6
    height = 128
    width = 128
    observation_space = Box(0, 255, [height, width, 3])
    fun = FuN(observation_space, action_space)

    for _ in range(10):
        image_batch = torch.randn(batch, 3, height, width)
        action, goal = fun(image_batch)
        print(fun._intrinsic_reward())


if __name__ == "__main__":
    test_forward()
