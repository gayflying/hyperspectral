from A3C_environment import *
from A3C_data import *
import numpy as np

data = futuresData()
data.loadData_moreday0607()

env = futuresGame(data)
obs = env.reset()
print('init observation:' + str(obs))

for i in range(0,10):
    action = np.zeros(7)
    action[0] = 0.1
    action[1] = 0.2
    action[-1] = 0.7
    obs, allo, reward, done, info = env.step(action)
    #print(obs)
    #print(allo)
    print(reward)
    #print(done)
    #print(info)
