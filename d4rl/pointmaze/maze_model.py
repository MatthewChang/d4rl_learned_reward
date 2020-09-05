""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel
import numpy as np
import random
from matplotlib import pyplot as plt 
import skfmm
import cv2
import torch
import mujoco_py
from sklearn.neighbors import KDTree

WALL = 10
EMPTY = 11
GOAL = 12
START = 13


def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            elif tile == 'S':
                maze_arr[w][h] = START
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr


def point_maze(maze_str,massScale = 1.0):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel('point_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density=str(1000*massScale), margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(type="2d",name="groundplane",builtin="checker",rgb1="0.2 0.3 0.4",rgb2="0.1 0.2 0.3",width=100,height=100)
    asset.texture(name="skybox",type="skybox",builtin="gradient",rgb1=".4 .6 .8",rgb2="0 0 0",
               width="800",height="800",mark="random",markrgb="1 1 1")
    asset.material(name="groundplane",texture="groundplane",texrepeat="20 20")
    asset.material(name="wall",rgba=".7 .5 .3 1")
    asset.material(name="target",rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4",diffuse=".8 .8 .8",specular="0.1 0.1 0.1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name='ground',size="40 40 0.25",pos="0 0 -0.1",type="plane",contype=1,conaffinity=0,material="groundplane")

    particle = worldbody.body(name='particle', pos=[1.2,1.2,0])
    particle.geom(name='particle_geom', type='sphere', size=0.1, rgba='0.0 0.0 1.0 0.0', contype=1)
    particle.site(name='particle_site', pos=[0.0,0.0,0], size=0.2, rgba='0.3 0.6 0.3 1')
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    worldbody.site(name='target_site', pos=[0.0,0.0,0], size=0.2, material='target')

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w,h] == WALL:
                worldbody.geom(conaffinity=1,
                               type='box',
                               name='wall_%d_%d'%(w,h),
                               material='wall',
                               pos=[w+1.0,h+1.0,0],
                               size=[0.5,0.5,0.2])

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


# Render orientation is the transpose of what the ascii looks like
LARGE_MAZE_25 = '\\'.join(
        ['#########################',
 '#GOOO#OOO#O#OO#OO#OOO#OO#',
 '##O#OO#O#OOOO#O#O#O#O#O##',
 '#O#OO#OOO###OOOOOOO##OO##',
 '#OO#O###O#OO#O#O##OOO#OO#',
 '##OOOOO#OO#OO#OO#OO##O#O#',
 '#OO#O#O##OOO#O#OO#O#OO#O#',
 '#O###OOOOO##OO#O#O#OO#OO#',
 '##O#O#O##OOO#O#O#OOO#O#O#',
 '#OO#OO#OOO#OOOOO#O#O#O#O#',
 '##O#O###O#OO###O#OO##OOO#',
 '#OO#OO#O##O#OOSOOO#OO##O#',
 '##OOO#OO#O#O#O###O#O##OO#',
 '#O##O#O#OOOOO##OOOOO#O#O#',
 '#OOOOOOO####O#O###O#OOOO#',
 '#O#O#O#OOOO#OOOOOOOOO#O##',
 '#O#O##OO##OOO##O#O#O##OO#',
 '#OO#OOO#OOO#OO#OO#OOOO#O#',
 '#O#OO#O##O#OO#OO#OO##O#O#',
 '#OO#OO#OOO###OO###O#OO#O#',
 '#O#OO#O#O#OOOO#O#OOO#O#O#',
 '#O#O#OOO#OO##O#O#O#O##OO#',
 '##O#O#O#O##OOO#OOO##OOO##',
 '#OOOOOOOOOOO#OO#O#OOO#OO#',
 '#########################']
 )

EXTRA_LARGE_MAZE = '\\'.join(
        ['###############',
 '##G#OOO#OO#O#O#',
 '#OOO#O#O#O#OOO#',
 '#O#O#O#OOO#O#O#',
 '##OOOOO#O#O#OO#',
 '#O##O#O#O#O#O##',
 '#O#O#OOOSOOOOO#',
 '#O#OOO##O##O#O#',
 '#OO#O#OO##O#OO#',
 '##OOO#O#OOO#O##',
 '#OO##OOOO#OOOO#',
 '##OOO#O#O##O#O#',
 '#OO#O#OO#OOO#O#',
 '##O#OO#OO#O#OO#',
 '###############']
 )

LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOS#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####0###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#GO#OOO#OOO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"


MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#SOO#OG#\\'+\
        "########"

MEDIUM_MAZE_FLIPPED = \
        '########\\'+\
        '#GO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#SOO#O0#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"


class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(self,
                 maze_spec=U_MAZE,
                 reward_type='dense',
                 bonus=True,
                 reset_target=False,
                 fixed_start=False,
                 learned_reward_model = None,
                 render_in_info = False,
                 # currently unused, just consuming the argument
                 frame_skip = 5,
                 massScale=1.0,
                 normalize=lambda x:x,
                 state_only=False,
                 **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)

        self.normalize = normalize
        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.fixed_start = fixed_start
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.start_locations = list(zip(*np.where(self.maze_arr == START)))
        self.reset_locations.sort()
        self.success = False
        self.bonus = bonus
        if  reward_type == 'exploration':
            self.scale = 4
        else: 
            self.scale = 200
        self.render_in_info = render_in_info

        self._target = np.array([0.0,0.0])
        self.learned_reward_model = learned_reward_model
        self.num_steps = 0
        self.state_only=state_only

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0])
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            # If no goal, use the first empty tile
            self.set_target(np.array(self.reset_locations[0]).astype(self.observation_space.dtype))
        self.empty_and_goal_locations = self.reset_locations + self.goal_locations + self.start_locations

        # FMM Map setup
        ma = np.ones_like(self.maze_arr).astype(np.float)
        mask = (self.maze_arr==WALL).astype(np.int)
        scaled_map = cv2.resize(ma,None,fx=self.scale,fy=self.scale,interpolation=cv2.INTER_NEAREST)
        scaled_mask = cv2.resize(mask,None,fx=self.scale,fy=self.scale,interpolation=cv2.INTER_NEAREST).astype(np.bool)
        # erode because the ball can clip into the wall some
        # eroded = cv2.erode(scaled_mask.astype(np.uint16),np.ones((25,25))).astype(np.bool)
        # eroded = cv2.erode(scaled_mask.astype(np.uint16),np.ones((1,1)))
        masked_map = np.ma.MaskedArray(scaled_map, mask=scaled_mask)
        masked_map[self._target[0]*self.scale+self.scale//2,self._target[1]*self.scale+self.scale//2] =-1 
        self.dists = skfmm.distance(masked_map)
        self.max_dist = self.dists.max()
        self.kdt_dists = np.stack(np.where(self.dists),axis=1)
        self.dist_kdt = KDTree(self.kdt_dists)

        self.explore_map_base = scaled_mask.astype(int)
        self.explore_map = scaled_mask.astype(int)
        self.total_to_explore = (self.explore_map == 0).sum()

        model = point_maze(maze_spec,massScale=massScale)
        self.old_value = 0

        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1)
        utils.EzPickle.__init__(self)

        self.set_marker()

    def render(self,*args,width=420,height=380):
        return self.sim.render(width,height)

    def pos_to_map(self,pos):
        return pos*self.scale + self.scale//2

    def percent_explored(self):
        return (self.explore_map == 2).sum()/self.total_to_explore

    def step(self, action):

        # self.get_obs() -> s_{t-1}
        goal_dist = 0.4

        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()

        # self.get_obs() -> s_{t}
        ob = self._get_obs()
        dist = np.linalg.norm(ob[0:2] - self._target)

        dpos = self.pos_to_map(ob[:2])
        geo_dist = self.dists[int(dpos[0]),int(dpos[1])]/self.max_dist
        if np.ma.is_masked(geo_dist):
            geo_dist = 1
            # nearest_ind = self.dist_kdt.query([dpos])[1][0,0]
            # nearest_loc = self.kdt_dists[nearest_ind,:]
            # geo_dist = self.dists[nearest_loc[0],nearest_loc[1]]/self.max_dist
        done = False
        if self.reward_type == 'sparse':
            # avoid gym error if done on first step
            if dist <= goal_dist and self.num_steps > 0:
                reward = 1.0 
                done = True
                # print("Done\n\n\n")
            else:
                reward = 0
        elif self.reward_type == 'dense':
            # reward = np.exp(-dist)
            # val = 1-geo_dist
            val = np.exp(-dist)
            if dist <= goal_dist and self.num_steps > 0:
                # reward = 1-self.old_value
                reward = val-self.old_value
                if self.bonus:
                    reward += 5
                done = True
            else:
                reward = val-self.old_value
                self.old_value = val
        elif self.reward_type == 'exploration':
            before = self.percent_explored()
            self.explore_map[int(dpos[0]),int(dpos[1])] = 2
            reward = self.percent_explored()-before
        elif self.reward_type == 'learned':
            reward=self.learned_reward_model(torch.tensor(ob).float().cuda())
            reward += 5.0 if dist <= goal_dist else 0.0
        elif self.reward_type == 'valdif':
            value = self.learned_reward_model(torch.tensor(ob).float().cuda())
            reward= value - self.old_value
            self.old_value = value
            if dist <= goal_dist and self.num_steps > 0:
                done = True
                reward += 5.0
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)
        info = {}
        info['raw_obs'] = ob
        info['sparse_reward'] = 1.0 if dist <= goal_dist else 0.0
        info['is_success'] = dist <= goal_dist
        info['dist_to_goal'] = dist
        info['geodesic_distance'] = geo_dist
        if self.render_in_info:
            info['render'] = self.render()
        self.num_steps += 1
        info['map'] = self.explore_map
        return self.normalize(ob), reward, done, info

    def _get_obs(self):
        if self.state_only:
            return self.sim.data.qpos.ravel()
        else:
            return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
            target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        self._target = target_location

    def set_marker(self):
        self.data.site_xpos[self.model.site_name2id('target_site')] = np.array([self._target[0]+1, self._target[1]+1, 0.0])

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        if self.fixed_start:
            idx = self.np_random.choice(len(self.start_locations))
            reset_location = np.array(self.start_locations[idx]).astype(self.observation_space.dtype)
        else:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.num_steps = 0
        if self.reset_target:
            self.set_target()
        self.set_marker()
        self.explore_map = self.explore_map_base.copy()

        if self.reward_type == 'valdif':
            ob = self._get_obs()
            self.old_value = self.learned_reward_model(torch.tensor(ob).float().cuda())
            # v(s_t)-v(s_{t-1})
        else:
            self.old_value = 0
        return self.normalize(self._get_obs())

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.explore_map = self.explore_map_base.copy()

        if self.reward_type == 'valdif':
            ob = self._get_obs()
            self.old_value = self.learned_reward_model(torch.tensor(ob).float().cuda())
        else:
            self.old_value = 0
        return self.normalize(self._get_obs())
