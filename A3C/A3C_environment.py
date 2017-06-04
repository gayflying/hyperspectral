#environment class for finantial RL by @jialiang.cui
import gym
import numpy as np

class futuresGame:
    def __init(self,futuresNum = 3,inforFieldsNum = 2):
        self.mFuturesNum = futuresNum
        self.inforFieldsNum = inforFieldsNum
        self.ction_space = gym.spaces.box.Box(np.linspace(-1,-1,futuresNum+1),np.linspace(1,1,futuresNum+1))
        self.observation_space = gym.spaces.box.Box(np.zeros(inforFieldsNum * futuresNum),np.linspace(100000000,100000000,inforFieldsNum * futuresNum))
        

    def reset(self):
        self.mProperty = 10000
        self.mAssetAllocation = np.zeros(self.mFuturesNum + 1)
        self.mAssetAllocation[-1] = 1
        return self.observation_space.sample()


    def step(self, action):
        #update property and assert allocation

        #update observation
        observation = self.observation_space.sample()
        #update assert allocation num
        reward = 100
        done = false
        info = {}
        return [observation, assetAllocation, reward, done, info]
