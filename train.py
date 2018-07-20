from itertools import count

import torch
from torch.distributions import Categorical
import torch.optim as optim
from torch.autograd import Variable

from fun import FeudalNet
from envs import create_atari_env

from tensorboardX import SummaryWriter


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(
        rank,
        shared_model,
        counter,
        log_dir,
        lock,
        optimizer,
        args):
    seed = args.seed + rank
    torch.manual_seed(seed)

    env = create_atari_env(args.env_name)
    env.seed(seed)
    model = FeudalNet(env.observation_space, env.action_space, channel_first=True)

    if optimizer is None:
        print("no shared optimizer")
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=log_dir)

    model.train()

    obs = env.reset()
    obs = torch.from_numpy(obs)
    done = True

    episode_length = 0
    for epoch in count():
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        if done:
            states = model.init_state(1)
        else:
            states = model.reset_states_grad(states)

        values_worker, values_manager = [], []
        log_probs = []
        rewards, intrinsic_rewards = [], []
        entropies = []  # regularisation
        manager_partial_loss = []

        for step in range(args.num_steps):
            episode_length += 1
            value_worker, value_manager, action_probs, goal, nabla_dcos, states = model(obs.unsqueeze(0), states)
            m = Categorical(probs=action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            entropy = -(log_prob * action_probs).sum(1, keepdim=True)
            entropies.append(entropy)
            manager_partial_loss.append(nabla_dcos)

            obs, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)
            intrinsic_reward = model._intrinsic_reward(states)
            intrinsic_reward = float(intrinsic_reward)  # TODO batch

            #plt_reward.add_value(None, intrinsic_reward, "Intrinsic reward")
            #plt_reward.add_value(None, reward, "Reward")
            #plt_reward.draw()

            with lock:
                counter.value += 1

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
            R_worker = args.gamma_worker * R_worker + rewards[i] + args.alpha * intrinsic_rewards[i]
            R_manager = args.gamma_manager * R_manager + rewards[i]
            advantage_worker = R_worker - values_worker[i]
            advantage_manager = R_manager - values_manager[i]
            value_worker_loss = value_worker_loss + 0.5 * advantage_worker.pow(2)
            value_manager_loss = value_manager_loss + 0.5 * advantage_manager.pow(2)

            # Generalized Advantage Estimation
            delta_t_worker = \
                rewards[i] \
                + args.alpha * intrinsic_rewards[i]\
                + args.gamma_worker * values_worker[i + 1].data \
                - values_worker[i].data
            gae_worker = gae_worker * args.gamma_worker * args.tau_worker + delta_t_worker

            policy_loss = policy_loss \
                - log_probs[i] * gae_worker - args.entropy_coef * entropies[i]

            if (i + model.c) < len(rewards):
                # TODO try padding the manager_partial_loss with end values (or zeros)
                manager_loss = manager_loss \
                    - advantage_manager * manager_partial_loss[i + model.c]

        optimizer.zero_grad()

        total_loss = policy_loss \
            + manager_loss \
            + args.value_manager_loss_coef * value_manager_loss \
            + args.value_worker_loss_coef * value_worker_loss

        total_loss.backward()

        with lock:
            writer.add_scalars(
                'data/loss' + str(rank),
                {
                    'manager': float(manager_loss),
                    'worker': float(policy_loss),
                    'total': float(total_loss),
                },
                epoch
            )
            writer.add_scalars(
                'data/value_loss' + str(rank),
                {
                    'value_manager': float(value_manager_loss),
                    'value_worker': float(value_worker_loss),
                },
                epoch
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
