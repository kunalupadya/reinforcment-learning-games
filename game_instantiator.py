from openaigymexample import run_gym_game
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv
from gym.envs.atari import AtariEnv


class GameInstantiator():
    def run_cartpole(self, n_iter = 5):
        return run_gym_game("CartPole-v1", CartPoleEnv(), n_iter)

    def run_pong(self, n_iter = 5):
        return run_gym_game("Pong-v0", AtariEnv(game="pong"), n_iter)

    def run_space_invaders(self, n_iter = 5):
        return run_gym_game("SpaceInvaders-v0", AtariEnv(game="space_invaders"), n_iter)

    def run_mountain_car(self, n_iter = 5):
        return run_gym_game("MountainCar-v0", MountainCarEnv(), n_iter)
if __name__ == "__main__":
    g = GameInstantiator()
    g.run_space_invaders(5)
    # g.run_mountain_car()