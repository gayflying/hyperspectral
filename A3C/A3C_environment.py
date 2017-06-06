#environment class for finantial RL by @jialiang.cui
import gym.spaces
import numpy as np

from A3C_data import futuresData

class futuresGame:
    def __init__(self,data):
        self.mSupportEmpty = 1
        self.mData = data
        self.mFuturesNum = self.mData.mFuturesNum
        self.mInforFieldsNum = self.mData.mInforFieldsNum
        self.ction_space = gym.spaces.box.Box(np.linspace(0,0,self.mFuturesNum+self.mSupportEmpty),np.linspace(1,1,self.mFuturesNum+1))
        self.observation_space = gym.spaces.box.Box(np.zeros(self.mInforFieldsNum * self.mFuturesNum),np.linspace(100000000,100000000,self.mInforFieldsNum * self.mFuturesNum))

    def reset(self, initProperty = 100000.0):
        self.totalReward = 0
        self.mProperty = initProperty
        self.mAssetAllocation = np.zeros(self.mFuturesNum + self.mSupportEmpty)
        if self.mSupportEmpty:
            self.mAssetAllocation[-1] = 1
        observation = self.mData.getObservation(0)
        self.mPrice = self.mData.getPrice(0)
        self.time = 1
        return observation


    def step(self, action):
        assert self.time <= self.mData.mLength - 1
        assert len(action) == self.mFuturesNum + self.mSupportEmpty
        #update property and assert allocation
        
        newPrice = self.mData.getPrice(self.time)
        newProperty = 0.0
        reward = 0.0
        newContrib = np.zeros(self.mFuturesNum + self.mSupportEmpty)
        for i in range(0,self.mFuturesNum):
            reward -= self.mData.mPoundage * abs(self.mAssetAllocation[i] - action[i]) * self.mProperty
            oldContribi = self.mProperty  * action[i]
            newContrib[i] = oldContribi / self.mPrice[i] * newPrice[i]
            newProperty += newContrib[i]
        if self.mSupportEmpty == 1:
            newProperty += action[-1] * self.mProperty
        reward += newProperty - self.mProperty
        percentageReward = reward / self.mProperty

        #update
        for i in range(0,self.mFuturesNum):
            self.mAssetAllocation[i] = newContrib[i] / newProperty
        if self.mSupportEmpty == 1:
            self.mAssetAllocation[-1] = action[-1] / self.mProperty * newProperty
        self.mPrice = newPrice
        self.mProperty = newProperty

        #update observation
        observation = self.mData.getObservation(self.time)

        self.time += 1
        if self.time >= self.mData.mLength:
            done = True
        else:
            done = False
        info = {}
        self.totalReward += reward
        return [observation, self.mAssetAllocation, percentageReward, done, info]
