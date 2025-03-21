#!/usr/bin/env python
"""
basic_agent.py

This script loads a trained PPO model for the Basic level environment
and runs the VizDoom environment in evaluation mode.
"""

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from vizdoom import DoomGame
from gymnasium.spaces import Discrete, Box


class VizDoomGym:
    def __init__(self, render=False):
        self.game = DoomGame()
        # Load the basic level configuration file
        self.game.load_config('github/ViZDoom/scenarios/basic.cfg')
        self.game.set_window_visible(render)
        self.game.init()

        # Define observation and action spaces
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)  # Actions: MOVE_LEFT, MOVE_RIGHT, ATTACK

    def step(self, action):
        # Define the action mapping using identity matrix
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = {"ammo": ammo}
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = {}

        done = self.game.is_episode_finished()
        return state, reward, done, False, info  # False is for 'truncated' as per new Gymnasium API

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state), {}

    def grayscale(self, observation):
        # Convert to grayscale and resize to (100, 160, 1)
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resize, (100, 160, 1))

    def close(self):
        self.game.close()


def main():
    # Load the trained model (adjust the model path if necessary)
    model_path = './models/PPO1/model_800000.zip'
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    # Create the environment with rendering enabled
    env = VizDoomGym(render=True)
    state, _ = env.reset()

    try:
        while True:
            # Use the model to predict the action in a deterministic way
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Done: {done}")

            # Delay to control frame rate
            time.sleep(0.02)

            if done:
                state, _ = env.reset()

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")

    finally:
        env.close()
        print("Environment closed.")


if __name__ == '__main__':
    main()
