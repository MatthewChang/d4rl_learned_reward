import torch
import gym
import d4rl  # Import required to register environments
import numpy as np
import mujoco_py
import numpy as np
import os
import re
from matplotlib import pyplot as plt
import random
import cv2
# env = gym.make('kitchen-end-effector-v0')
# print("Obs Space:", env.observation_space)
# print("Obs Shape:", env.reset().shape)
# exit()

# klass = 'hinge'
klass = 'microwave'

env = gym.make('kitchen-partial-v0')
dataset = env.get_dataset()
dataset = d4rl.qlearning_dataset(env)
dataset['observations'].shape
observations = []
maxes = dataset['observations'][:,:30].max(axis=0)
mins = dataset['observations'][:,:30].min(axis=0)
for i in range(30):
    obs = dataset['observations'][0].copy()
    obs[i] = maxes[i]
    env.robot.reset(env,obs[:30],np.zeros((30,)))
    env.sim.forward()
    img = np.array(env.render_pixels(1280,720))
    plt.imsave(f'vis/dof%02d-max.png'%(i),img)

    obs[i] = mins[i]
    env.robot.reset(env,obs[:30],np.zeros((30,)))
    env.sim.forward()
    img = np.array(env.render_pixels(1280,720))
    plt.imsave(f'vis/dof%02d-min.png'%(i),img)


def convert_obs(ob):
    env.robot.reset(env,ob[:30],np.zeros((30,)))
    env.sim.forward()
    efsid = env.model.site_name2id('end_effector')
    ef_pos = env.data.site_xpos[efsid]
    return np.concatenate((ef_pos,ob[9:30]))
    

new_obs = []
new_next_obs = []
rewards = []
from tqdm import tqdm
for i in tqdm(range(len(dataset['observations']))):
    obs = dataset['observations'][i]
    robs = obs[:30]
    # hinge_cabinet
    if klass == 'hinge':
        # reward = robs[-9] > 0.42
        reward = robs[-9] > 0.52
    elif klass == 'microwave':
        reward = abs(robs[-8]) >= 0.7

    # if reward:
        # env.robot.reset(env,obs[:30],np.zeros((30,)))
        # env.sim.forward()
        # img = np.array(env.render_pixels(1280,720))
        # cv2.putText(img, "%0.2f"%(robs[-9]), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=2)
        # plt.imsave(f'vis/test.png',img)
        # import pdb; pdb.set_trace()

    new_obs.append(convert_obs(obs))
    new_next_obs.append(convert_obs(dataset['next_observations'][i]))
    rewards.append(reward)

new_obs = np.stack(new_obs)
new_next_obs = np.stack(new_next_obs)
rewards = np.stack(rewards)
rewards = rewards.astype(np.float32)

import h5py
fil = h5py.File(f"end_effector_kitchen_partial_{klass}_updated.hdf5", "w")
fil.create_dataset('observations',data=new_obs)
fil.create_dataset('next_observations',data=new_next_obs)
fil.create_dataset('rewards',data=rewards)
fil.close()

# fil = h5py.File(f"end_effector_kitchen_partial_{klass}.hdf5",'r')
# fil.keys()
# fil['observations'].shape
# dataset['observations'].shape

