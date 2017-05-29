import cv2
from A3C_config import *
import numpy as np
import gym

class smart_env():
    def __init__(self, frames=args.frames):
        self._frames = frames
        self._env = gym.make(args.game)
        self._state = None
        self.states = []
    def reset(self):
        self._state = rgb2gray(resize(self._env.reset()))
        self.state = np.stack([self._state for _ in range(self._frames)], axis=2)
        return self.state
    def next(self, action):
        s_next, reward, end_flag, _ = self._env.step(action)
        self._state = rgb2gray(resize(s_next))
        self.state = np.append(self._state[:, :, np.newaxis], self.state[:, :, 0:3], axis=2)
        return self.state, reward, end_flag

def rgb2gray(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    return res

def resize(image):
    return cv2.resize(image, (80, 80))










