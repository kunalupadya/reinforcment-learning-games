from openaigymexample import run_gym_game, init_gym_game, restore_saved_agent
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv
from gym.envs.atari import AtariEnv
import os


class GameInstantiator():
    def __init__(self, testing = False):
        self.game_call = run_gym_game if testing else init_gym_game

    def run_cartpole(self, n_iter = 5):
        return self.game_call("CartPole-v1", CartPoleEnv(), n_iter)

    def restore_cartpole(self, n_iter):
        return restore_saved_agent("CartPole-v1", CartPoleEnv(), os.getcwd() + '/CartPole/checkpoint_' + str(n_iter) + '/checkpoint-' + str(n_iter))

    def run_pong(self, n_iter = 5):
        return self.game_call("Pong-v0", AtariEnv(game="pong"), n_iter)

    def run_space_invaders(self, n_iter = 5):
        return self.game_call("SpaceInvaders-v0", AtariEnv(game="space_invaders"), n_iter)

    def run_mountain_car(self, n_iter = 5):
        return self.game_call("MountainCar-v0", MountainCarEnv(), n_iter)
if __name__ == "__main__":
    g = GameInstantiator(True)
    g.run_space_invaders(5)
    # g.run_mountain_car()