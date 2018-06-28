from collections import namedtuple
import os

import torch
import torch.multiprocessing as mp

import my_optim
from fun import FeudalNet
from envs import create_atari_env
from train import train
from test import test


Args = namedtuple("Args", [
    "lr",
    "alpha",  # intrinsic reward multiplier
    "entropy_coef",  # beta
    "tau_worker",
    "gamma_worker",
    "gamma_manager",
    "num_steps",
    "max_episode_length",
    "max_grad_norm",
    "value_worker_loss_coef",
    "value_manager_loss_coef",
    "seed",
    "num_processes",
    "no_shared",
    "env_name",
])


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    mp.set_start_method('spawn')

    args = Args(
        lr=0.001,  # try LogUniform(1e-4.5, 1e-3.5)
        alpha=0.8,
        entropy_coef=0,  # 0.01,
        tau_worker=1.00,
        gamma_worker=0.95,
        gamma_manager=0.99,
        num_steps=400,
        max_episode_length=1000000,
        max_grad_norm=40,
        value_worker_loss_coef=0.5,
        value_manager_loss_coef=0.5,
        seed=123,
        num_processes=4,
        no_shared=False,
        env_name='PongDeterministic-v4',
    )

    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    shared_model = FeudalNet(env.observation_space, env.action_space, channel_first=True)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, shared_model, counter, args))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, shared_model, counter, lock, optimizer, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
