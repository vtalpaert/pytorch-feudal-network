import time
from collections import deque
from itertools import count

import torch

from plot import Plotter
from fun import FeudalNet
from envs import create_atari_env


def test(
        rank,
        shared_model,
        counter,
        args):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    model = FeudalNet(env.observation_space, env.action_space, channel_first=True)

    model.eval()

    obs = env.reset()
    obs = torch.from_numpy(obs)
    done = True
    reward_sum = 0

    plt_loss = Plotter("Loss", ylim_max=1000)
    plt_reward = Plotter("Reward")

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    for epoch in count():
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            states = model.init_state(1)
        else:
            states = model.reset_states_grad(states)

        value_worker, value_manager, action_probs, goal, _, states = model(obs.unsqueeze(0), states)
        action = action_probs.max(1, keepdim=True)[1].data.numpy()

        obs, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            obs = env.reset()
            time.sleep(60)

        obs = torch.from_numpy(obs)
