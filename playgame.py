import PySimpleGUI as sg
#from game_instantiator import GameInstantiator
import gym
from gym.utils.play import play, PlayPlot
import ray
from ray import tune
from ray.rllib.agents.impala import ImpalaTrainer

## Code from https://www.codeproject.com/Articles/5271948/Learning-Breakout-More-Quickly
## Playing the game manually

def callback(ob_t, obs_tp1, action, rew, done, info):
    return [rew]

def runCartPole():
    cartPole = dict()
    cartPole[(ord('a'),)] = 0
    cartPole[(ord('d'),)] = 1
    env = gym.make("CartPole-v1")
    # play(env, callback=plotter.callback, zoom=1)
    play(env, keys_to_action=cartPole)
#plotter = PlayPlot(callback, 30*5, ["reward"])



def makePlayWindow():
    layout = [[sg.Text("Play cartpole!")],
              [sg.Button("Play"), sg.Button("Quit")]]
    window = sg.Window("Cartpole window", layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Quit":
            break
        if event == 'Play':
            runCartPole()
    window.close()

if __name__ == "__main__":
    makePlayWindow()


## Ignore the code below
#ray.shutdown()
#ray.init(include_webui=False, ignore_reinit_error=True)
#ray.init()
#
#ENV = "Breakout-ramNoFrameskip-v4"
#TARGET_REWARD = 200
#TRAINER = ImpalaTrainer
#
#tune.run(TRAINER, stop={"episode_reward_mean": TARGET_REWARD},
#         config={"env": ENV,
#                 "monitor": True,
#                 "evaluation_num_episodes": 25,
#                 "rollout_fragment_length": 50,
#                 "train_batch_size": 500,
#                 "num_workers": 1,
#                 "num_envs_per_worker": 2,
#                 "clip_rewards": True,
#                 "lr_schedule": [
#                     [0, 0.0005],
#                     [20_000_000, 0.000000000001]
#                 ]})
#print("Finished training")



# sg.theme('SystemDefaultForReal')
#
# layout = [[sg.Text(
#     'Welcome to the RFFL, where we want to make reinforcement learning\naccessible and understandable.\n\nWhat game would you like to play?')],
#     [sg.Radio('PixelCopter', "GAMES", key="PixelCopter")],
#     [sg.Radio('Catcher', "GAMES", key="Catcher")],
#     [sg.Radio('CartPole', "GAMES", key="CartPole")],
#     [sg.Radio('MountainCar', "GAMES", key="MountainCar")],
#     [sg.Radio('SpaceInvaders', "GAMES", key="SpaceInvaders")],
#     [sg.Radio('Pong', "GAMES", key="Pong")],
#     [sg.Button('Display'), sg.Button('Play'), sg.Button('Exit')]]
#
# window = sg.Window('Welcome', layout)
#
# def open_pixelCopter(iterations):
#     # TODO
#     print('Not implimented')
#
#
# def open_catcher(iterations):
#     # layout = [[sg.Image(background_color='black', key='animation')]]
#
#     # catcher_window = sg.Window('Catcher', layout, modal = True, finalize=True)
#     # animation = catcher_window['animation']
#     # while True:
#     #   event, values = catcher_window.Read(timeout = 100)
#     #   if event is None:
#     #     break
#     #   animation.update_animation('anim.gif', 100)
#
#     # catcher_window.close()
#
#     #image = sg.Image(data=animation, background_color='white', key='anim')
#     #layout = [[image], [sg.Text('Catcher trained on ' + str(iterations) + ' iterations')]]
#     layout = [[sg.Text('Catcher trained on ' + str(iterations) + ' iterations')]]
#
#     catcher_window = sg.Window('Catcher', layout, finalize=True)
#
#     while True:
#         event, values = catcher_window.read(timeout=100)
#         if event in (None, 'Exit'):
#             break
#         #image.update_animation_no_buffering(animation, time_between_frames=60)
#
#     # for i in range(18000):
#     #   sg.popup_animated(animation, time_between_frames=60)
#     # sg.popup_animated(None)
#
# def open_game_play(values):
#     return
#
# def open_game(values, iterations):
#     GameInst = GameInstantiator()
#     if values['PixelCopter']:
#         open_pixelCopter(iterations)
#     elif values['Catcher']:
#         open_catcher(iterations)
#     elif values['CartPole']:
#         GameInst.run_cartpole(iterations)
#     elif values['MountainCar']:
#         GameInst.run_mountain_car(iterations)
#     elif values['SpaceInvaders']:
#         GameInst.run_space_invaders(iterations)
#     elif values['Pong']:
#         GameInst.run_pong(iterations)
#
# def open_game_menu(prior_values):
#     show_iters = False
#     layout = [[sg.Text('How many iterations would you like to view?')],
#               [sg.Radio('100', "ITER", key='100'), sg.Radio('200', "ITER", key='200'),
#                sg.Radio('300', "ITER", key='300'), sg.Radio('Let me train my own model!', "ITER", key='train')],
#               [sg.pin(sg.Column([[sg.Text('How many iterations would you like to train?'),
#                                   sg.Spin([i * 5 for i in range(1, 11)], key='iterations')]], key='train_opt',
#                                 visible=show_iters))],
#               [sg.Button('Next'), sg.Button('Exit')]]
#     window2 = sg.Window('Game Options', layout, modal=True)
#
#     while True:
#         event, values = window2.read()
#         print(event, values)
#
#         if event is None or 'Exit' in event:
#             print(event)
#             break
#
#         if event == 'Next':
#             if values['train'] and not show_iters:
#                 show_iters = True
#                 window2['train_opt'].update(visible=show_iters)
#             else:
#                 n_iter = int(values['iterations']) if values['train'] else \
#                 [x for x in range(100, 400, 100) if values[str(x)]][0]  # Sorry for what I've done
#                 open_game(prior_values, n_iter)
#
#     window2.close()
#
# while True:
#     event, values = window.read()
#     print(event, values)
#
#     if event in (None, 'Exit'):
#         break
#
#     if event == 'Display':
#         open_game_menu(values)
#
#     if event == 'Play':
#         open_game_play(values)
#
#
#window.close()