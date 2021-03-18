from openaigymexample import run_gym_game
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv
from gym.envs.atari import AtariEnv


class GameInstantiator():
    def run_cartpole(self):
        return run_gym_game("CartPole-v1", CartPoleEnv())

    def run_space_invaders(self):
        return run_gym_game("SpaceInvaders-v0", AtariEnv())

    def run_mountain_car(self):
        return run_gym_game("MountainCar-v0", MountainCarEnv())
if __name__ == "__main__":
    g = GameInstantiator()
    g.run_space_invaders()
    # g.run_mountain_car()