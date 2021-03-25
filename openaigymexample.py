# adapted from https://github.com/DerwenAI/gym_example/blob/master/train.py

import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import gym
from time import sleep
import matplotlib.pyplot as plt
import sys
import os

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
            # chkpt_file
        ))

    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())
    return agent

def init_gym_game(select_env, openai_env, n_iter = 5):
    ray.init(ignore_reinit_error=True)
    register_env(select_env, lambda config: openai_env)

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = train_gym_game(ppo.PPOTrainer(config, env=select_env), n_iter)
    # apply the trained policy in a rollout
    env = gym.make(select_env)
    return env, agent

def run_gym_game(select_env, openai_env, n_iter = 5):
    env, agent = init_gym_game(select_env, openai_env, n_iter)
    print("45")
    state = env.reset()
    sum_reward = 0
    n_step = 1000

    rgbs = []
    for step in range(n_step):
        # time.time
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        #sleep(0.250)
        rgb = env.render(mode="rgb_array")
        #plt.imshow(rgb)
        #plt.show()
        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0
    return rgbs, agent

def gen_saved_agents(select_env, openai_env, checkpoint_path):
    env, agent = init_gym_game(select_env, openai_env, 0)
    for i in range(3):
        agent = train_gym_game(agent, 1)
        agent.save(checkpoint_path)

def restore_saved_agent(select_env, openai_env, checkpoint_path):
    ray.init(ignore_reinit_error=True)
    register_env(select_env, lambda config: openai_env)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 0
    agent = ppo.PPOTrainer(config, env=select_env)
    agent.restore(checkpoint_path)
    # apply the trained policy in a rollout
    env = gym.make(select_env)
    return env, agent

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'CartPole':
        from gym.envs.classic_control import CartPoleEnv
        checkpoint_path = os.getcwd() + '/' + sys.argv[1] if len(sys.argv) == 2 else sys.argv[2]
        gen_saved_agents("CartPole-v1", CartPoleEnv(), checkpoint_path)
    else:
        from gym.envs.classic_control import CartPoleEnv
        select_env = "CartPole-v1"
        run_gym_game(select_env, CartPoleEnv())