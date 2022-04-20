import numpy as np
import torch
import mujoco_py


def disk_goal_sampler(np_random, goal_region_radius=10.):
    th = 2 * np.pi * np_random.uniform()
    radius = goal_region_radius * np_random.uniform()
    return radius * np.array([np.cos(th), np.sin(th)])


def constant_goal_sampler(np_random, location=10.0 * np.ones([2])):
    return location


class GoalReachingEnv(object):
    """General goal-reaching environment."""
    BASE_ENV = None  # Must be specified by child class.

    def __init__(self, goal_sampler, eval=False, reward_type=None,bonus=True):
        # must be set by inherited components or something
        # self._goal_sampler = goal_sampler
        # self._goal = np.ones([2])
        # self.target_goal = self._goal

        # This flag is used to make sure that when using this environment
        # for evaluation, that is no goals are appended to the state
        self.eval = eval

        # This is the reward type fed as input to the goal confitioned policy
        self.reward_type = reward_type

        self.scale = 100
        np_map = np.array(self._maze_map)
        target = np.where(np_map == 'g')
        occupancy = np.array(self._maze_map)
        occupancy[occupancy == 'r'] = 0
        occupancy[occupancy == 'g'] = 0
        self.start_dist = 0
        self.last_val = 0
        self.old_value = 0
        self.bonus = bonus
        print("BONUS: ",self.bonus)

    def pos_to_arr(self, pos):
        small_map_pos = (pos + [self._init_torso_x, self._init_torso_y]) / self._maze_size_scaling
        return np.flip(small_map_pos)

    def pos_to_map(self, pos):
        small_map_pos = (pos + [self._init_torso_x, self._init_torso_y]) / self._maze_size_scaling
        return np.flip((small_map_pos * self.scale + self.scale / 2).astype(np.int))

    def _get_obs(self):
        base_obs = self.BASE_ENV._get_obs(self)
        goal_direction = self._goal - self.get_xy()
        if not self.eval:
            obs = np.concatenate([base_obs, goal_direction])
            return obs
        else:
            return base_obs

    @property
    def img_viewer(self):
        if not hasattr(self,'_viewer') or not self._viewer:
            self._viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)
            self.sim.add_render_context(self._viewer)
        return self._viewer

    # x -> cols, y = rows
    def step(self, a):
        self.BASE_ENV.step(self, a)
        obs = self._get_obs()
        dist = np.linalg.norm(self.get_xy() - self.target_goal)
        goal_dist = 0.5*self._maze_size_scaling

        map_pos = self.pos_to_map(self.get_xy())
        if self.reward_type == 'dense':
            scaled_val = (self.start_dist - dist)/self.start_dist
            reward = scaled_val - self.last_val
            self.last_val = scaled_val
            if self.bonus:
                reward += 5.0 if dist <= goal_dist else 0.0
        elif self.reward_type == 'sparse':
            reward = 1.0 if dist <= goal_dist else 0.0
        elif self.reward_type == 'learned':
            reward = self.learned_reward_model(torch.tensor(self.point_val_pos()+(0,0)).float().cuda(),[self._init_torso_x,self._init_torso_y])
            reward += 5.0 if dist <= goal_dist else 0.0
        elif self.reward_type == 'valdif':
            value = self.learned_reward_model(torch.tensor(self.point_val_pos()+(0,0)).float().cuda(),[self._init_torso_x,self._init_torso_y])
            reward = value - self.old_value
            self.old_value = value
            if self.bonus and dist <= goal_dist:
                reward += 5.0
        else:
            raise Exception(f'invalid reward')
        
        done = False
        # Terminate episode when we reach a goal
        if self.eval and dist <= goal_dist and self.reward_type != 'learned':
            done = True

        info = {}
        info['sparse_reward'] = 1.0 if dist <= goal_dist else 0.0
        info['is_success'] = dist <= goal_dist
        info['dist_to_goal'] = dist
        if self.render_in_info:
            info['render'] = self.render()

        # print(info['dist_to_goal'])

        return obs, reward, done, info

    def reset_model(self):
        if self.target_goal is not None or self.eval:
            self._goal = self.target_goal
        else:
            self._goal = self._goal_sampler(self.np_random)

        res = self.BASE_ENV.reset_model(self)

        obs = self._get_obs()
        self.start_dist = np.linalg.norm(self.get_xy() - self.target_goal)
        self.last_val = 0
        if self.reward_type == 'valdif':
            self.old_value = self.learned_reward_model(torch.tensor(self.point_val_pos()+(0,0)).float().cuda())
        return res
