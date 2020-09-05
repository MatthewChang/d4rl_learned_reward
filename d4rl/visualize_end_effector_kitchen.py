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
import h5py

import d4rl
import gym
import torch
from stable_baselines3 import DDPG
# env = gym.make('kitchen-end-effector-v0')
# model1 = DDPG("MlpPolicy", env, verbose=1)
# model1 = model1.load('/data02/arjung2/rl_unplugged/kitchen/runs/kitchen-end-effector-v0_ddpg_init_gt/checkpoints/63600.zip')
# def rew(x):
    # with torch.no_grad():
        # xt = torch.tensor(x).cuda().unsqueeze(0)
        # acts = model1.actor(xt)
        # res = model1.critic(xt,acts)[0].item()
    # return res

# latent kitchen
obs = np.load('/data02/arjung2/rl_unplugged/kitchen_obs.npy')
mean = torch.tensor(obs.mean(axis=0)).cuda().float()
stds = torch.tensor(obs.std(axis=0)).cuda().float()
network = FCQNetwork(24,8,means=mean,stds=stds,positional_encoding=False,arch_dim=256).cuda()
network.load_state_dict(torch.load('/data01/arjung2/habitat_test/configs_kitchen_archdim256_latent_top8/experiments/real_data/models/sample30000.torch')['model_state_dict'])
network.cuda()
def rew(x):
    with torch.no_grad():
        res = network(torch.tensor([x]).float().cuda())
        return res.max().item()

fil = h5py.File("end_effector_kitchen_partial_microwave.hdf5", "r")
fil.keys()
env = gym.make('kitchen-partial-v0')
def convert_obs(ob):
    env.robot.reset(env,ob[:30],np.zeros((30,)))
    env.sim.forward()
    efsid = env.model.site_name2id('end_effector')
    ef_pos = env.data.site_xpos[efsid]
    return np.concatenate((ef_pos,ob[9:30]))

dataset = env.get_dataset()
dataset = d4rl.qlearning_dataset(env)
inds = np.random.randint(0,dataset['observations'].shape[0],(25,))
samples = dataset['observations'][inds]
# get the end effector version of the sample
for imid,ind in enumerate(inds):
    obs = dataset['observations'][ind]
    ef_data = convert_obs(obs)
    # verify that the ef version matches whats in the ef dataset
    assert all(ef_data == fil['observations'][ind])
    # get the value here
    value = rew(ef_data)
    env.robot.reset(env,obs[:30],np.zeros((30,)))
    env.sim.forward()
    img = np.array(env.render_pixels(1280,720))
    cv2.putText(img, "%0.2f"%(value), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=2)
    plt.imsave(f'vis/%03d.png'%(imid),img)

