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


# TODO propagate train()
class dLSTM(nn.Module):
    """Implements the dilated LSTM
    Uses a cyclic list of size r to keep r independent hidden states
    """
    def __init__(self, r, input_size, hidden_size):
        super(dLSTM, self).__init__()
        self.lstm = LSTMCell(input_size, hidden_size)
        self.r = r

    def init_state(self, batch_size):
        # note that we cannot keep the state in only one tensor as updating one place of the tensor counts
        # as an inplace operation and breaks the gradient history
        h0 = [torch.zeros(batch_size, self.lstm.hidden_size, requires_grad=True) for _ in range(self.r)]
        c0 = [torch.zeros(batch_size, self.lstm.hidden_size, requires_grad=True) for _ in range(self.r)]
        tick = 0
        return tick, h0, c0

    def forward(self, inputs, states):
        """Returns g_t, (tick, hidden)
        hidden is [batch x r x 2 x hidden_size], we use 2 for hx, cx
        """
        tick, hx, cx = states
        out, _ = hx[tick], cx[tick] = self.lstm(inputs, (hx[tick], cx[tick]))
        tick = (tick + 1) % self.r
        return out, (tick, hx, cx)


class Perception(nn.Module):
    """Returns z, the shared intermediate representation [batch x d]
    """

    def __init__(self, observation_shape, d, channel_first):
        super(Perception, self).__init__()
        self.channel_first = channel_first

        # Note that we expect input in Pytorch images' style : batch x C x H x W (this is arg channel_first)
        # but in gym a Box.shape for an image is (H, W, C) (use channel_first=False)
        height, width, channels = observation_shape
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

        self.f_Wrnn = LSTMCell(d, num_outputs * k)

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

    def forward(self, x, states, reset_value_grad=False):
        states_W, states_M, ss = states
        tick_dlstm, hx_M, cx_M = states_M

        z = self.perception(x)

        s_prev = ss[:, tick_dlstm, :]
        g_prev = F.normalize(hx_M[tick_dlstm])

        value_manager, g, s, states_M = self.manager(z, states_M, reset_value_grad)
        ss[:, tick_dlstm, :] = s
        nabla_dcos_t_minus_c = d_cos((s - s_prev).data, g_prev)

        # sum on c different gt values, note that gt = normalize(hx)
        sum_goal = sum(map(F.normalize, states_M[1]))
        sum_goal_W = reset_grad2(sum_goal)

        value_worker, action_probs, states_W = self.worker(z, sum_goal_W, states_W, reset_value_grad)

        return value_worker, value_manager, action_probs, g, nabla_dcos_t_minus_c, (states_W, states_M, ss)

    def init_state(self, batch_size):
        ss = torch.zeros(batch_size, self.c, self.d)
        return self.worker.init_state(batch_size), self.manager.init_state(batch_size), ss

    def reset_states_grad(self, states):
        states_W, states_M, ss = states
        return self.worker.reset_states_grad(states_W), self.manager.reset_states_grad(states_M), ss

    def _intrinsic_reward(self, states):
        states_W, states_M, ss = states
        tick, hx_M, cx_M = states_M
        t = (tick - 1) % self.c  # tick is always ahead
        s_t = ss[:, t, :]
        rI = torch.zeros(s_t.size(0), 1)
        for i in range(1, self.c):
            t_minus_i = (t - i) % self.c
            s_t_i = ss[:, t_minus_i, :]
            g_t_i = F.normalize(hx_M[t_minus_i])
            rI += d_cos(s_t - s_t_i, g_t_i)
        return rI / self.c


def train(
        env,
        model,
        lr,
        alpha,  # intrinsic reward multiplier
        entropy_coef,  # beta
        tau_worker,
        gamma_worker,
        gamma_manager,
        num_steps,
        max_episode_length,
        max_grad_norm,
        value_worker_loss_coef=0.5,
        value_manager_loss_coef=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    obs = env.reset()
    obs = torch.from_numpy(obs)
    done = True

    episode_length = 0
    while True:
        print("start")
        # Sync with the shared model
        #model.load_state_dict(shared_model.state_dict())

        if done:
            states = model.init_state(1)
        else:
            states = model.reset_states_grad(states)

        values_worker, values_manager = [], []
        log_probs = []
        rewards, intrinsic_rewards = [], []
        entropies = []  # regularisation
        manager_partial_loss = []

        for step in range(num_steps):
            episode_length += 1
            value_worker, value_manager, action_probs, goal, nabla_dcos, states = model(obs.unsqueeze(0), states)
            m = Categorical(probs=action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            entropy = -(log_prob * action_probs).sum(1, keepdim=True)
            entropies.append(entropy)
            manager_partial_loss.append(nabla_dcos)

            obs, reward, done, _ = env.step(action.numpy())
            env.render()
            done = done or episode_length >= max_episode_length
            reward = max(min(reward, 1), -1)
            intrinsic_reward = model._intrinsic_reward(states)
            intrinsic_reward = float(intrinsic_reward)  # TODO batch

            #with lock:
            #    counter.value += 1

            if done:
                episode_length = 0
                obs = env.reset()

            obs = torch.from_numpy(obs)
            values_manager.append(value_manager)
            values_worker.append(value_worker)
            log_probs.append(log_prob)
            rewards.append(reward)
            intrinsic_rewards.append(intrinsic_reward)

            if done:
                break

        R_worker = torch.zeros(1, 1)
        R_manager = torch.zeros(1, 1)
        if not done:
            value_worker, value_manager, _, _, _, _ = model(obs.unsqueeze(0), states)
            R_worker = value_worker.data
            R_manager = value_manager.data

        values_worker.append(Variable(R_worker))
        values_manager.append(Variable(R_manager))
        policy_loss = 0
        manager_loss = 0
        value_manager_loss = 0
        value_worker_loss = 0
        gae_worker = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R_worker = gamma_worker * R_worker + rewards[i] + alpha * intrinsic_rewards[i]
            R_manager = gamma_manager * R_manager + rewards[i]
            advantage_worker = R_worker - values_worker[i]
            advantage_manager = R_manager - values_manager[i]
            value_worker_loss = value_worker_loss + 0.5 * advantage_worker.pow(2)
            value_manager_loss = value_manager_loss + 0.5 * advantage_manager.pow(2)

            # Generalized Advantage Estimation
            delta_t_worker = \
                rewards[i] \
                + alpha * intrinsic_rewards[i]\
                + gamma_worker * values_worker[i + 1].data \
                - values_worker[i].data
            gae_worker = gae_worker * gamma_worker * tau_worker + delta_t_worker

            policy_loss = policy_loss \
                - log_probs[i] * gae_worker - entropy_coef * entropies[i]

            if (i + model.c) < len(rewards):
                manager_loss = manager_loss \
                    - advantage_manager * manager_partial_loss[i + model.c]

        optimizer.zero_grad()

        total_loss = policy_loss \
            + manager_loss \
            + value_manager_loss_coef * value_manager_loss \
            + value_worker_loss_coef * value_worker_loss

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        #ensure_shared_grads(model, shared_model)
        optimizer.step()


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
