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

def init_gym_game(select_env, openai_env, n_iter = 5, algorithm = 'PPO', atari = False):
    ray.init(ignore_reinit_error=True)
    register_env(select_env, lambda config: gym.make(select_env))

    algo, trainer = get_trainer(algorithm)

    config = algo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = train_gym_game(trainer(config, env=select_env), n_iter)
    # apply the trained policy in a rollout
    env = gym.make(select_env)
    return env, agent

def run_gym_game(select_env, openai_env, n_iter = 5, atari = False):
    env, agent = init_gym_game(select_env, openai_env, n_iter)
    state = env.reset()
    sum_reward = 0
    n_step = 1000

    rgbs = []
    prep = get_preprocessor(env.observation_space)(env.observation_space)

    for step in range(n_step):
        if atari:
            action = agent.compute_action(np.mean(prep.transform(state), 2))
        else:
            action = agent.compute_action(prep.transform(state))
        state, reward, done, info = env.step(action)
        sum_reward += reward
        env.render()

        if done == 1:
            # print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0
    return rgbs, agent

def gen_saved_agents(select_env, openai_env, checkpoint_path, algorithm):
    env, agent = init_gym_game(select_env, openai_env, 0, algorithm)
    for i in range(3):
        agent = train_gym_game(agent, 100)
        agent.save(checkpoint_path)

def restore_saved_agent(select_env, openai_env, checkpoint_path, algorithm, atari=False):
    ray.init(ignore_reinit_error=True)
    register_env(select_env, lambda config: gym.make(select_env))

    algo, trainer = get_trainer(algorithm)
    config = algo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 0
    agent = trainer(config, env=select_env)
    agent.restore(checkpoint_path)
    # apply the trained policy in a rollout
    env = gym.make(select_env)
    return env, agent, atari

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'CartPole':
        from gym.envs.classic_control import CartPoleEnv
        algorithm = 'PPO' if len(sys.argv) > 1 else sys.argv[2]
        checkpoint_path = os.getcwd() + '/' + sys.argv[1] + '/' + algorithm if len(sys.argv) > 2 else sys.argv[3]
        gen_saved_agents("CartPole-v1", CartPoleEnv(), checkpoint_path, algorithm)
    if len(sys.argv) > 1 and sys.argv[1] == 'LunarLander':
        algorithm = 'PPO' if len(sys.argv) == 1 else sys.argv[2]
        checkpoint_path = os.getcwd() + '/' + sys.argv[1] + '/' + algorithm if len(sys.argv) > 2 else sys.argv[3]
        print(checkpoint_path)
        gen_saved_agents("LunarLander-v2", None, checkpoint_path, algorithm)
    else:
        from gym.envs.classic_control import CartPoleEnv
        select_env = "CartPole-v1"
        run_gym_game(select_env, CartPoleEnv())
        run_gym_game("BreakoutNoFrameskip-v4", None, 1, True)