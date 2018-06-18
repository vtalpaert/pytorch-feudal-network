from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


def reset_grad1(t):
    return Variable(t.data)


def reset_grad2(t):
    tr = t.detach()
    tr.requires_grad = t.requires_grad
    return tr


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


class LSTMCell(nn.LSTMCell):
    """Regular LSTMCell class
    If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    (Behaviour not working contrary to nn.LSTMCell documentation's promise)
    """
    def forward(self, inputs, hidden):
        if hidden is None:  # TODO cuda
            batch_size = inputs.size(0)
            hidden = (
                torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size)
            )
        return super(LSTMCell, self).forward(inputs, hidden)


class dLSTM(nn.Module):
    """Implements the dilated LSTM
    Uses a cyclic buffer of size r to keep r independent hidden states,
    the buffer is a tensor of size [batch x r x 2 x hidden_size]
    """
    def __init__(self, r, input_size, hidden_size):
        super(dLSTM, self).__init__()
        self.lstm = LSTMCell(input_size, hidden_size)
        self.r = r

    def init_state(self, batch_size):
        hidden = torch.zeros(batch_size, self.r, 2, self.lstm.hidden_size)
        tick = 0
        return tick, hidden

    def forward(self, inputs, states):
        """Returns g_t, (tick, hidden)
        hidden is [batch x r x 2 x hidden_size], we use 2 for hx, cx
        """
        tick, hidden = states
        hidden[:, tick, 0, :], hidden[:, tick, 1, :] = self.lstm(inputs, (hidden[:, tick, 0, :], hidden[:, tick, 1, :]))
        tick = (tick + 1) % self.r
        return hidden[:, tick, 0, :], (tick, hidden)


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


class Perception(nn.Module):
    """Returns z, the shared intermediate representation [batch x d]
    """

    def __init__(self, observation_shape, d, channel_first):
        super(Perception, self).__init__()
        self.channel_first = channel_first

        # Note that we expect input in Pytorch images' style : batch x C x H x W (this is arg channel_first)
        # but in gym a Box.shape for an image is (H, W, C) (use channel_first=False)
        height, width, channels = observation_shape
        if not channel_first:
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

        self.f_Wrnn = LSTMCell(d, num_outputs * k)

        self.view_as_actions = View((k, num_outputs))

        self.phi = nn.Sequential(
            nn.Linear(d, k, bias=False),
            View((1, k))
        )

        self.value_function = nn.Linear(num_outputs * k, 1)

    def reset_states_grad(self, states):
        return reset_grad2(states)

    def init_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.f_Wrnn.hidden_size),
            torch.zeros(batch_size, self.f_Wrnn.hidden_size)
        )

    def forward(self, z, sum_g_W, states_W):
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

        a = (w @ U).squeeze()  # [batch x a]

        probs = F.softmax(a, dim=1)

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

    def forward(self, z, states_M):
        s = self.f_Mspace(z)  # latent state representation [batch x d]
        g_hat, states_M = self.f_Mrnn(s, states_M)

        g = F.normalize(g_hat)  # goal [batch x d]

        value = self.value_function(g_hat)

        return value, g, s, states_M

    def init_state(self, batch_size):
        return self.f_Mrnn.init_state(batch_size)

    def reset_states_grad(self, states):
        tick, hidden = states
        return tick, reset_grad2(hidden)


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

    def forward(self, x, states):
        states_W, states_M, ss = states
        tick_dlstm, _ = states_M

        z = self.perception(x)

        value_manager, g, s, states_M = self.manager(z, states_M)
        ss[:, tick_dlstm, :] = s

        # sum on c different gt values, note that gt = normalize(hx)
        sum_goal = F.normalize(states_M[1][:, :, 0:1, :], dim=3).sum(dim=1)
        sum_goal_W = reset_grad2(sum_goal)

        value_worker, action_probs, states_W = self.worker(z, sum_goal_W, states_W)

        return value_worker, value_manager, action_probs, g, (states_W, states_M, ss)

    def init_state(self, batch_size):
        ss = torch.zeros(batch_size, self.c, self.d)
        return self.worker.init_state(batch_size), self.manager.init_state(batch_size), ss

    def reset_states_grad(self, states):
        states_W, states_M, ss = states
        return self.worker.reset_states_grad(states_W), self.manager.reset_states_grad(states_M), ss


def train(env, lr, num_steps, max_episode_length):
    model = FeudalNet(env.observation_space, env.action_space, channel_first=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    obs = env.reset()
    obs = torch.from_numpy(obs)
    done = True

    episode_length = 0
    while True:
        # Sync with the shared model
        #model.load_state_dict(shared_model.state_dict())

        if done:
            states = model.init_state()
        else:
            states = model.reset_states_grad(states)

        values = []
        log_probs = []
        rewards = []
        entropies = []  # regularisation

        for step in range(num_steps):
            episode_length += 1
            value, action_probs, goal, states = model(obs.unsqueeze(0), states)
            m = Categorical(probs=action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            #entropy = -(log_prob * prob).sum(1, keepdim=True)
            #entropies.append(entropy)

            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= max_episode_length
            reward = max(min(reward, 1), -1)

            #with lock:
            #    counter.value += 1

            if done:
                episode_length = 0
                obs = env.reset()

            state = torch.from_numpy(obs)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                                   values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                          log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)


    optimizer.step()


def test_forward():
    from gym.spaces import Box
    import numpy as np

    batch = 4
    action_space = 6
    height = 128
    width = 128
    observation_space = Box(0, 255, [height, width, 3], dtype=np.uint8)
    fun = FeudalNet(observation_space, action_space, channel_first=True)
    states = fun.init_state(batch)

    for i in range(10):
        image_batch = torch.randn(batch, 3, height, width)
        value_worker, value_manager, action_probs, goal, states = fun(image_batch, states)
        print(value_worker, value_manager)
        #print(i, "th rewards are ", fun._intrinsic_reward())


if __name__ == "__main__":
    test_forward()
