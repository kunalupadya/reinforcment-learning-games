import PySimpleGUI as sg
from game_instantiator import GameInstantiator

sg.theme('SystemDefaultForReal')

layout = [[sg.Text(
    'Welcome to the RFFL, where we want to make reinforcement learning\naccessible and understandable.\n\nWhat game would you like to play?')],
          [sg.Radio('PixelCopter', "GAMES", key="PixelCopter")],
          [sg.Radio('Catcher', "GAMES", key="Catcher")],
          [sg.Radio('CartPole', "GAMES", key="CartPole")],
          [sg.Radio('MountainCar', "GAMES", key="MountainCar")],
          [sg.Radio('Pong', "GAMES", key="Pong")],
          [sg.Button('Display'), sg.Button('Exit')]]

window = sg.Window('Welcome', layout)


def open_pixelCopter(iterations):
    # TODO
    print('Not implimented')


def open_catcher(iterations):
    # layout = [[sg.Image(background_color='black', key='animation')]]

    # catcher_window = sg.Window('Catcher', layout, modal = True, finalize=True)
    # animation = catcher_window['animation']
    # while True:
    #   event, values = catcher_window.Read(timeout = 100)
    #   if event is None:
    #     break
    #   animation.update_animation('anim.gif', 100)

    # catcher_window.close()

    #image = sg.Image(data=animation, background_color='white', key='anim')
    layout = [[image], [sg.Text('Catcher trained on ' + str(iterations) + ' iterations')]]

    catcher_window = sg.Window('Catcher', layout, finalize=True)

    while True:
        event, values = catcher_window.read(timeout=100)
        if event in (None, 'Exit'):
            break
        #image.update_animation_no_buffering(animation, time_between_frames=60)

    # for i in range(18000):
    #   sg.popup_animated(animation, time_between_frames=60)
    # sg.popup_animated(None)


def open_game(values, iterations):
    GameInst = GameInstantiator()
    if values['PixelCopter']:
        open_pixelCopter(iterations)
    elif values['Catcher']:
        open_catcher(iterations)
    elif values['CartPole']:
        GameInst.run_cartpole(iterations)
    elif values['MountainCar']:
        GameInst.run_mountain_car(iterations)
    elif values['Pong']:
        GameInst.run_pong(iterations)

def open_game_menu(prior_values):
    show_iters = False
    layout = [[sg.Text('How many iterations would you like to view?')],
              [sg.Radio('100', "ITER", key='100'), sg.Radio('200', "ITER", key='200'), sg.Radio('300', "ITER", key='300'), sg.Radio('Let me train my own model!', "ITER", key='train')] ,
              [sg.pin(sg.Column([[sg.Text('How many iterations would you like to train?'), sg.Spin([i * 5 for i in range(1 ,11)], key='iterations')]], key='train_opt', visible=show_iters))],
              [sg.Button('Next'), sg.Button('Exit')]]
    window2 = sg.Window('Game Options', layout, modal=True)

    while True:
        event, values = window2.read()
        print(event, values)

        if event is None or 'Exit' in event:
            print(event)
            break

        if event == 'Next':
            if values['train'] and not show_iters:
                show_iters = True
                window2['train_opt'].update(visible=show_iters)
            else:
                n_iter = int(values['iterations']) if values['train'] else [x for x in range(100, 400, 100) if values[str(x)]][0] # Sorry for what I've done
                open_game(prior_values, n_iter)

    window2.close()

while True:
    event, values = window.read()
    print(event, values)

    if event in  (None, 'Exit'):
        break

    if event == 'Display':
        open_game_menu(values)

window.close()