import PySimpleGUI as sg

# layout = [[sg.Text('My one-shot window.')],
#                  [sg.InputText()],
#                  [sg.Submit(), sg.Cancel()]]

layout1 = [[sg.Text('Welcome to RFFL, where we want to make reinforcement learning\naccessible and understandable.\n\nWhat game would you like to play?')],
          [sg.Combo(['choice 1', 'choice 2', 'choice 3'], enable_events=True, key='combo')],
          [sg.Radio('CartPole', "GAMES", key="CartPole")],
          [sg.Radio('MountainCar', "GAMES", key="MountainCar")],
          #[sg.Radio('SpaceInvaders', "GAMES", key="SpaceInvaders")],
          #[sg.Radio('Pong', "GAMES", key="Pong")],
          [sg.Radio('LunarLander', "GAMES", key="LunarLander")],
          [sg.Button('Display'), sg.Button('Exit')]]


layout1 = [[sg.Text('This is layout 1 - It is all Checkboxes')],
           *[[sg.CB(f'Checkbox {i}')] for i in range(5)]]

layout2 = [[sg.Button('Back', key='b2')],
           [sg.Text('This is layout 2')],
           [sg.Input(key='-IN-')],
           [sg.Input(key='-IN2-')]]

layout3 = [[sg.Button('Back', key='b3')],
           [sg.Text('This is layout 3 - It is all Radio Buttons')],
           *[[sg.R(f'Radio {i}', 1)] for i in range(8)]]

# ----------- Create actual layout using Columns and a row of Buttons
layout = [[sg.Column(layout1, key='-COL1-'), sg.Column(layout2, visible=False, key='-COL2-'), sg.Column(layout3, visible=False, key='-COL3-')],
          [sg.Button('Cycle Layout'), sg.Button('1'), sg.Button('2', key="yeet"), sg.Button('3'), sg.Button('Exit')]]
window = sg.Window('Swapping the contents of a window', layout)

layout = 1  # The currently visible layout
while True:
    event, values = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
    if event == 'Cycle Layout':
        window[f'-COL{layout}-'].update(visible=False)
        layout = layout + 1 if layout < 3 else 1
        window[f'-COL{layout}-'].update(visible=True)
    elif event in '123':
        window[f'-COL{layout}-'].update(visible=False)
        layout = int(event)
        window[f'-COL{layout}-'].update(visible=True)
window.close()
