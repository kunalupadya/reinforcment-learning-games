#!/usr/bin/env python
import sys, gym, time

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.
ACTIONS = None

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, ACTIONS
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action, ACTIONS
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause, SKIP_CONTROL
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

def humanTrainGame(environment):
    env = gym.make(environment)
    env.reset()

    if not hasattr(env.action_space, 'n'):
        raise Exception('Keyboard agent only supports discrete action spaces')
    global ACTIONS
    ACTIONS = env.action_space.n

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    while 1:
        window_still_open = rollout(env)
        if window_still_open==False: break

if __name__ == "__main__":
    environment = 'MountainCar-v0' if len(sys.argv) == 1 else sys.argv[1]
    humanTrainGame(environment)