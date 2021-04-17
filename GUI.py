import PySimpleGUI as sg
import webbrowser
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from ray.rllib.models.preprocessors import get_preprocessor

matplotlib.use('TkAgg')

sg.SetOptions(
                 button_color = sg.COLOR_SYSTEM_DEFAULT
               , text_color = sg.COLOR_SYSTEM_DEFAULT
             )

algorithms = ['PPO', 'DQN']

sg.theme('SystemDefaultForReal')

layout = [[sg.Text('Welcome to RFFL, where we want to make reinforcement learning\naccessible and understandable.\n\nWhat game would you like to play?')],
          #[sg.Radio('PixelCopter', "GAMES", key="PixelCopter")],
          #[sg.Radio('Catcher', "GAMES", key="Catcher")],
          [sg.Radio('CartPole', "GAMES", key="CartPole")],
          [sg.Radio('MountainCar', "GAMES", key="MountainCar")],
          #[sg.Radio('SpaceInvaders', "GAMES", key="SpaceInvaders")],
          #[sg.Radio('Pong', "GAMES", key="Pong")],
          [sg.Radio('LunarLander', "GAMES", key="LunarLander")],
          [sg.Button('Display'), sg.Button('Exit')]]

window = sg.Window('Welcome', layout)


def open_game(values, iterations, algorithm):
    from game_instantiator import GameInstantiator

    env = None
    agent = None

    GameInst = GameInstantiator()
    # if values['PixelCopter']:
    #   open_pixelCopter(iterations)
    # elif values['Catcher']:
    #   open_catcher(iterations)
    if values['CartPole']:
        getAgent = GameInst.restore_cartpole if iterations % 100 == 0 else GameInst.run_cartpole
        env, agent = getAgent(iterations, algorithm)
    elif values['MountainCar']:
        GameInst.run_mountain_car(iterations, algorithm)
    # elif values['SpaceInvaders']:
    #   GameInst.run_space_invaders(iterations, algorithm)
    # elif values['Pong']:
    #   GameInst.run_pong(iterations, algorithm)
    elif values['LunarLander']:
        getAgent = GameInst.restore_lunar_lander if iterations % 100 == 0 else GameInst.run_lunar_lander
        env, agent = getAgent(iterations, algorithm)

    # a = agent.DEFAULT_CONFIG.copy()
    # z = [(k, a[k]) for k in a if (type(a[k]) in {int, str, bool, dict})]

    return env, agent

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def animate_game(env, agent, window3, atari=False):
    # fig = plt.figure()
    # plt.clf()

    # layout = [[sg.Text('Plot test')],
    #         [sg.Canvas(key='-CANVAS-')],
    #         [sg.Button('Ok')]]

    # window3 = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout, finalize=True, element_justification='center', font='Helvetica 18')
    # fig_canvas_agg = draw_figure(window3['-CANVAS-'].TKCanvas, fig)

    state = env.reset()
    sum_reward = 0
    n_step = 1000

    prep = get_preprocessor(env.observation_space)(env.observation_space)

    for step in range(n_step):
        event, values = window3.read(timeout = 0)
        print(event, values)
        if event is None or 'Exit' in event:
            break
        if atari:
            action = agent.compute_action(np.mean(prep.transform(state), 2))
        else:
            action = agent.compute_action(prep.transform(state))

        state, reward, done, info = env.step(action)
        sum_reward += reward

        plt.imshow(env.render(mode="rgb_array"))
        if done == 1:
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0

    window3.close()

def open_game_menu(prior_values):
    env = None
    agent = None
    show_iters = False
    show_learn_more = False
    show_teach = prior_values['MountainCar']
    print(show_teach)
    layout = [[sg.Text('Which algorithm would you like to train?')],
              [sg.Radio(algo, "ALGO", key=algo, enable_events=True) for algo in algorithms] ,
              [sg.Text('How many iterations would you like to view?')],
              [sg.Radio('100', "ITER", key='100'), sg.Radio('200', "ITER", key='200'), sg.Radio('300', "ITER", key='300'), sg.Radio('Let me train my own model!', "ITER", key='train')] ,
              [sg.pin(sg.Column([[sg.Text('How many iterations would you like to train?'), sg.Spin([i for i in range(1,11)], key='iterations')]], key='train_opt', visible=show_iters))],
              [sg.Button('Next'), sg.Button('Exit'), sg.Button('Teach the Algorithm', key='teach', visible=show_teach), sg.Button('Learn About This Algorithm', key = 'learn_more', visible=show_learn_more)]]
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
            humanTrainGame('MountainCar-v0')


        if event == 'Next':
            if values['train'] and not show_iters:
                show_iters = True
                window2['train_opt'].update(visible=show_iters)
            else:
                n_iter = int(values['iterations']) if values['train'] else [x for x in range(100, 400, 100) if values[str(x)]][0] # Sorry for what I've done
                algorithm = [x for x in algorithms if values[str(x)]][0]

                # agent = get_agent(algorithm)
                # if train:
                #     agent = train(agent)

                env, agent = open_game(prior_values, n_iter, algorithm)

        if env != agent:
            animate_game(env, agent, window2)

        if event is None or 'Exit' in event:
            break
    window2.close()

while True:
    event, values = window.read()
    print(event, values)

    if event in  (None, 'Exit'):
        break

    if event == 'Display':
        open_game_menu(values)

window.close() 