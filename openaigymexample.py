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
from ray.rllib.models.preprocessors import get_preprocessor, GenericPixelPreprocessor
import numpy as np



def run_gym_game(select_env, openai_env, n_iter = 5, atari=False):
    ray.init(ignore_reinit_error=True)


    register_env(select_env, lambda config: gym.make(select_env))

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=select_env)

    # config = dqn.DEFAULT_CONFIG.copy()
    # config["log_level"] = "WARN"
    # agent = dqn.DQNTrainer(config, env=select_env)

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
            # chkpt_file
        ))

    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

    # apply the trained policy in a rollout
    env = gym.make(select_env)
    print("45")
    state = env.reset()
    sum_reward = 0
    n_step = 1000

    rgbs = []
    prep = get_preprocessor(env.observation_space)(env.observation_space)

    globals().update(locals())
    for step in range(n_step):
        # time.time
        if atari:
            action = agent.compute_action(np.mean(prep.transform(state), 2))
        else:
            action = agent.compute_action(prep.transform(state))
        state, reward, done, info = env.step(action)
        sum_reward += reward

        env.render()
        # rgb = env.render(mode="rgb_array")
        #plt.imshow(rgb)
        #plt.show()
        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0
    return rgbs, agent

if __name__ == "__main__":
    # from gym.envs.classic_control import CartPoleEnv
    # select_env = "CartPole-v1"
    # run_gym_game(select_env, CartPoleEnv())
    run_gym_game("BreakoutNoFrameskip-v4", None, 1, True)