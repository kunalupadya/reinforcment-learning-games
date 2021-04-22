import PySimpleGUI as sg
import webbrowser
from matplotlib import pyplot as plt
import glob
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.error import UnsupportedSpaceException
from game_instantiator import GameInstantiator, NON_ATARI
import numpy as np
import json

matplotlib.use('TkAgg')

sg.SetOptions(
                 button_color = sg.COLOR_SYSTEM_DEFAULT
               , text_color = sg.COLOR_SYSTEM_DEFAULT
             )

algorithms = ['PPO', 'DQN']

games = [x[:-1] for x in glob.glob('*-*/')]


sg.theme('SystemDefaultForReal')

layout = [[sg.Text('Welcome to the RFFL, where we want to make reinforcement learning\naccessible and understandable.\n\nWhat game would you like to play?')]]
layout.extend([[sg.Radio(x.split('-')[0], "GAMES", key=x)] for x in games])
layout.append([sg.Button('Display'), sg.Button('Exit')])

window = sg.Window('Welcome', layout)

def open_game(chosen_game, iterations, algorithm, params_file):

    env = None
    agent = None

    GameInst = GameInstantiator()

    algo, _ = GameInst.getTrainer(algorithm)
    config = algo.DEFAULT_CONFIG.copy()

    if params_file != '':
        file = open(params_file, )
        hyperparameters = json.load(file)
        for hyperparam in hyperparameters:
            config[hyperparam[0]] = hyperparam[1]

    try:
        env, agent = GameInst.getAgent(chosen_game, iterations, algorithm, config)
    except UnsupportedSpaceException:
        sg.Popup('Algorithm not supported')

    return env, agent

def animate_game(env, agent, window3, chosen_game):
    atari = False if chosen_game in NON_ATARI else True

    state = env.reset()
    sum_reward = 0
    n_step = 1000

    prep = get_preprocessor(env.observation_space)(env.observation_space)

    rgbs = []
    for step in range(n_step):
        event, values = window3.read(timeout = 0)
        print(event, values)
        if event is None or 'Exit' in event:
            break
        if chosen_game == "Blackjack-v0":
            action = agent.compute_action(state)
        elif chosen_game == "Snake-v0":
            action = agent.compute_action(state)
        elif atari:
            action = agent.compute_action(np.mean(prep.transform(state), 2))
        else:
            action = agent.compute_action(prep.transform(state))

        state, reward, done, info = env.step(action)
        sum_reward += reward
        env.render()

        if done == 1:
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0
    #TODO dhruv show rgbs

    window3.close()

def open_game_menu(chosen_game):
    env = None
    agent = None
    show_iters = False
    show_learn_more = False

    layout = [[sg.Text('Which algorithm would you like to train?')],
              [sg.Radio(algo, "ALGO", key=algo, enable_events=True) for algo in algorithms] ,
              [sg.Text('How many iterations would you like to view?')],
              [sg.Radio('100', "ITER", key='100'), sg.Radio('200', "ITER", key='200'), sg.Radio('300', "ITER", key='300'), sg.Radio('Let me train my own model!', "ITER", key='train')] ,
              [sg.pin(sg.Column([[sg.Text('How many iterations would you like to train?'), sg.Spin([i for i in range(1,11)], key='iterations')]], key='train_opt', visible=show_iters))],
              [sg.pin(sg.Column([[sg.Text('Upload algorithm hyperparameters? (Optional): '), sg.FileBrowse(key='params_file')]], key='upload_params', visible=show_iters))],
              [sg.Button('Next'), sg.Button('Exit'), sg.Button('Play Game', key='teach'), sg.Button('Learn More', key = 'learn_more', visible=show_learn_more)]]
    window2 = sg.Window('Game Options', layout, modal=True)

    while True:
        event, values = window2.read()
        print(event, values)

        if event in algorithms and not show_learn_more:
            show_learn_more = True
            window2['learn_more'].update(visible=show_learn_more)
        if event == 'learn_more':
            algorithm = [x for x in algorithms if values[str(x)]][0]
            if algorithm == 'PPO':
                webbrowser.open('https://arxiv.org/abs/1707.06347')
            elif algorithm == 'DQN':
                webbrowser.open('https://arxiv.org/abs/1312.5602')

        if event == 'teach':
            from keyboard_agent import humanTrainGame
            humanTrainGame(chosen_game)


        if event == 'Next':
            if values['train'] and not show_iters:
                show_iters = True
                window2['train_opt'].update(visible=show_iters)
                window2['upload_params'].update(visible=show_iters)
            else:
                n_iter = int(values['iterations']) if values['train'] else [x for x in range(100, 400, 100) if values[str(x)]][0] # Sorry for what I've done
                algorithm = [x for x in algorithms if values[str(x)]][0]

                env, agent = open_game(chosen_game, n_iter, algorithm, values['params_file'])

        if env != agent:
            animate_game(env, agent, window2, chosen_game)

        if event is None or 'Exit' in event:
            break
    window2.close()

while True:
    event, values = window.read()
    print(event, values)

    if event in  (None, 'Exit'):
        break

    if event == 'Display':
        chosen_game = [x for x in games if values[x]][0]
        open_game_menu(chosen_game)

window.close() 