import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def reset_grad2(t, requires_grad=None):
    tr = t.detach()
    tr.requires_grad = t.requires_grad if requires_grad is None else requires_grad
    return tr


class View(nn.Module):
    """Layer changing the tensor shape
    Assumes batch first tensors by default
    Args:
        the output shape without providing the batch size
    """
    def __init__(self, shape, batched=True):
        super(View, self).__init__()
        self.shape = shape
        self.batched = batched

    def forward(self, x):
        if self.batched:
            return x.view(x.size(0), *self.shape)
        else:
            return x.view(*self.shape)


class dLSTM(nn.Module):
    """Implements the dilated LSTM
    Uses a cyclic list of size r to keep r independent hidden states
    """
    def __init__(self, r, input_size, hidden_size):
        super(dLSTM, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.r = r

    def init_state(self, batch_size):
        # note that we cannot keep the state in only one tensor as updating one place of the tensor counts
        # as an inplace operation and breaks the gradient history
        h0 = [torch.zeros(batch_size, self.lstm.hidden_size, requires_grad=self.training) for _ in range(self.r)]
        c0 = [torch.zeros(batch_size, self.lstm.hidden_size, requires_grad=self.training) for _ in range(self.r)]
        tick = 0
        return tick, h0, c0

    def forward(self, inputs, states):
        """Returns g_t, (tick, hidden)
        """
        tick, hx, cx = states
        hx[tick], cx[tick] = self.lstm(inputs, (hx[tick], cx[tick]))
        tick = (tick + 1) % self.r
        out = sum(hx) / self.r  # TODO verify that network output is mean of hidden states
        return out, (tick, hx, cx)


class Perception(nn.Module):
    """Returns z, the shared intermediate representation [batch x d]
    """

    def __init__(self, observation_shape, d, channel_first):
        super(Perception, self).__init__()
        self.channel_first = channel_first

        # Note that we expect input in Pytorch images' style : batch x C x H x W (this is arg channel_first)
        # but in gym a Box.shape for an image is (H, W, C) (use channel_first=False)
        if channel_first:
            channels, height, width = observation_shape
        else:
            height, width, channels = observation_shape
            self.view = View((channels, height, width))

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

    def forward(self, x):
        if not self.channel_first:
            x = self.view(x)
        return self.f_percept(x)


class Worker(nn.Module):
    def __init__(self, num_outputs, d, k):
        super(Worker, self).__init__()

        self.f_Wrnn = nn.LSTMCell(d, num_outputs * k)

        self.view_as_actions = View((k, num_outputs))

        self.phi = nn.Sequential(
            nn.Linear(d, k, bias=False),
            View((1, k))
        )

        self.value_function = nn.Linear(num_outputs * k, 1)

    def reset_states_grad(self, states):
        h, c = states
        return reset_grad2(h), reset_grad2(c)

    def init_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.f_Wrnn.hidden_size, requires_grad=self.training),
            torch.zeros(batch_size, self.f_Wrnn.hidden_size, requires_grad=self.training)
        )

    def forward(self, z, sum_g_W, states_W, reset_value_grad):
        """
        :param z:
        :param sum_g_W: should not have computation history
        :param worker_states:
        :return:
        """
        w = self.phi(sum_g_W)  # projection [ batch x 1 x k]

        # Worker
        U_flat, c_x = states_W = self.f_Wrnn(z, states_W)
        U = self.view_as_actions(U_flat)  # [batch x k x a]

        a = (w @ U).squeeze(1)  # [batch x a]

        probs = F.softmax(a, dim=1)

        if reset_value_grad:
            value = self.value_function(reset_grad2(U_flat))
        else:
            value = self.value_function(U_flat)

        return value, probs, states_W


class Manager(nn.Module):
    def __init__(self, d, c):
        super(Manager, self).__init__()
        self.c = c

        self.f_Mspace = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU()
        )

        self.f_Mrnn = dLSTM(c, d, d)

        self.value_function = nn.Linear(d, 1)

    def forward(self, z, states_M, reset_value_grad):
        s = self.f_Mspace(z)  # latent state representation [batch x d]
        g_hat, states_M = self.f_Mrnn(s, states_M)

        g = F.normalize(g_hat)  # goal [batch x d]

        if reset_value_grad:
            value = self.value_function(reset_grad2(g_hat))
        else:
            value = self.value_function(g_hat)

        return value, g, s, states_M

    def init_state(self, batch_size):
        return self.f_Mrnn.init_state(batch_size)

    def reset_states_grad(self, states):
        tick, hx, cx = states
        return tick, list(map(reset_grad2, hx)), list(map(reset_grad2, cx))


class FeudalNet(nn.Module):
    def __init__(self, observation_space, action_space, d=256, k=16, c=10, channel_first=True):
        super(FeudalNet, self).__init__()
        self.d, self.k, self.c = d, k, c

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
        elif action_space.__class__.__name__ == "Box":
            raise NotImplementedError
            # we first test this code with a softmax at the end
            #num_outputs = action_space.shape[0]
        elif isinstance(action_space, int):
            num_outputs = action_space
        else:
            raise NotImplementedError

        self.perception = Perception(observation_space.shape, d, channel_first)
        self.worker = Worker(num_outputs, d, k)
        self.manager = Manager(d, c)
        self.manager_partial_loss = nn.CosineEmbeddingLoss()

    def init_weights(self):
        """all submodules are already initialized like this"""
        def default_init(m):
            """Default is a uniform distribution"""
            for module_type in [nn.Linear, nn.Conv2d, nn.LSTMCell]:
                if isinstance(m, module_type):
                    m.reset_parameters()
        self.apply(default_init)

    def forward(self, x, states, reset_value_grad=False):
        states_W, states_M, ss = states
        tick_dlstm, hx_M, cx_M = states_M

        z = self.perception(x)

        s_prev = ss[tick_dlstm]
        g_prev = F.normalize(hx_M[tick_dlstm])

        value_manager, g, s, states_M = self.manager(z, states_M, reset_value_grad)
        ss[tick_dlstm] = s.detach()
        nabla_dcos_t_minus_c = self.manager_partial_loss((s - s_prev), g_prev, - torch.ones(g_prev.size(0)))

        # TODO randomly sample g_t from a univariate Gaussian

        # sum on c different gt values, note that gt = normalize(hx)
        sum_goal = sum(map(F.normalize, states_M[1]))
        sum_goal_W = reset_grad2(sum_goal, requires_grad=self.training)

        value_worker, action_probs, states_W = self.worker(z, sum_goal_W, states_W, reset_value_grad)

        return value_worker, value_manager, action_probs, g, nabla_dcos_t_minus_c, (states_W, states_M, ss)

    def init_state(self, batch_size):
        ss = [torch.zeros(batch_size, self.d, requires_grad=False) for _ in range(self.c)]
        return self.worker.init_state(batch_size), self.manager.init_state(batch_size), ss

    def reset_states_grad(self, states):
        states_W, states_M, ss = states
        return self.worker.reset_states_grad(states_W), self.manager.reset_states_grad(states_M), ss

    def _intrinsic_reward(self, states):
        states_W, states_M, ss = states
        tick, hx_M, cx_M = states_M
        t = (tick - 1) % self.c  # tick is always ahead
        s_t = ss[t]
        rI = torch.zeros(s_t.size(0), 1)
        for i in range(1, self.c):
            t_minus_i = (t - i) % self.c
            s_t_i = ss[t_minus_i]
            g_t_i = F.normalize(hx_M[t_minus_i].data)
            rI += F.cosine_similarity(s_t - s_t_i, g_t_i)
        return rI / self.c


def test_forward():
    from gym.spaces import Box
    import numpy as np

    batch = 4
    action_space = 6
    height = 128
    width = 128
    observation_space = Box(0, 255, [3, height, width], dtype=np.uint8)
    fun = FeudalNet(observation_space, action_space, channel_first=True)
    states = fun.init_state(batch)

    for i in range(10):
        image_batch = torch.randn(batch, 3, height, width, requires_grad=True)
        value_worker, value_manager, action_probs, goal, nabla_dcos, states = fun(image_batch, states)
        print("value worker", value_worker, "value manager", value_manager)
        print("intrinsic reward", fun._intrinsic_reward(states))


if __name__ == "__main__":
    test_forward()
