import cv2
import gym
import numpy as np
from gym.spaces.box import Box

from fun import FeudalNet
from train import train


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    return env


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42], dtype=np.float32)

    def observation(self, observation):
        return _process_frame42(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


if __name__ == "__main__":
    env = create_atari_env("PongDeterministic-v4")
    model = FeudalNet(env.observation_space, env.action_space, channel_first=True)

    lr = 0.001  # try LogUniform(1e-4.5, 1e-3.5)
    alpha = 0.8
    entropy_coef = 0.01
    tau_worker = 1.00
    gamma_worker = 0.95
    gamma_manager = 0.99
    num_steps = 400
    max_episode_length = 1000000
    max_grad_norm = 40
    value_worker_loss_coef = 0.5
    value_manager_loss_coef = 0.5

    train(env, model, lr, alpha, entropy_coef, tau_worker, gamma_worker, gamma_manager, num_steps, max_episode_length, max_grad_norm, value_worker_loss_coef=0.5, value_manager_loss_coef=0.5)
