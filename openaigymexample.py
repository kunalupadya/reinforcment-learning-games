# adapted from https://github.com/DerwenAI/gym_example/blob/master/train.py

import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import gym
from time import sleep
import matplotlib.pyplot as plt
from gym.utils.play import play, PlayPlot
from ray import tune
from ray.rllib.models.preprocessors import get_preprocessor
import numpy as np
import sys
import os
from BlackjackEnvRender import BlackjackEnvRender
import gym_snake #this line is important to import all snake envs, do not delete

# ONLY CHANGE IF TRAINING MODELS LOCALLY
TRAIN_LOCAL = True

SELECT_ENV = "Snake-v0"
ALGORITHM = "PPO"


def get_trainer(algorithm):
    if algorithm == 'PPO':
        return ppo, ppo.PPOTrainer
    elif algorithm == 'DQN':
        return dqn, dqn.DQNTrainer


def train_gym_game(agent, n_iter):
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f}"
    for n in range(n_iter):
        result = agent.train()
        # chkpt_file = agent.save(chkpt_root)

        print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"]
        ))

    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())
    return agent

def init_gym_game(select_env, n_iter = 5, algorithm = 'PPO', config = None):
    ray.init(ignore_reinit_error=True)
    env = make_env(select_env)
    register_env(select_env, lambda config: env)

    algo, trainer = get_trainer(algorithm)

    if config == None:
        config = algo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = train_gym_game(trainer(config, env=select_env), n_iter)
    # apply the trained policy in a rollout
    env = make_env(select_env)
    return env, agent

def gen_saved_agents(select_env, checkpoint_path, algorithm):
    env, agent = init_gym_game(select_env, 0, algorithm)
    for i in range(3):
        agent = train_gym_game(agent, 100)
        agent.save(checkpoint_path)

def restore_saved_agent(select_env, checkpoint_path, algorithm):
    ray.init(ignore_reinit_error=True)
    env = make_env(select_env)
    register_env(select_env, lambda config: env)

    algo, trainer = get_trainer(algorithm)
    config = algo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 0
    agent = trainer(config, env=select_env)
    agent.restore(checkpoint_path)
    # apply the trained policy in a rollout
    env = make_env(select_env)
    return env, agent


def make_env(select_env):
    if select_env == "Blackjack-v0":
        env = BlackjackEnvRender()
    elif select_env == "Snake-v0":
        env = gym.make("Snake-42x42-v0")
    else:
        env = gym.make(select_env)
    return env


if __name__ == "__main__":
    checkpoint_path = os.getcwd() + '/' + SELECT_ENV + '/' + ALGORITHM
    print(checkpoint_path)
    gen_saved_agents(SELECT_ENV, checkpoint_path, ALGORITHM)