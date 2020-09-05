""" Kitchen environment for long horizon manipulation """
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from d4rl.kitchen.adept_envs import robot_env
from d4rl.kitchen.adept_envs.utils.configurable import configurable
from gym import spaces
from dm_control.mujoco import engine

agent_bodies = ['panda0_link1','panda0_link2','panda0_link3','panda0_link4','panda0_link5','panda0_link6','panda0_link7','panda0_leftfinger','panda0_rightfinger','hook']
knobs = ['knob 1', 'knob 2', 'knob 3', 'knob 4']
lights = ['lightswitchbaseroot', 'lightswitchroot', 'lightblock_hinge']
slides = ['slide', 'slidelink']
hinge_doors = ['hingeleftdoor', 'hingerightdoor']
microwave = ['microwave', 'microroot', 'microdoorroot']
kettle = ['kettle', 'kettleroot']
bodies = {'Knobs': knobs, 'Lights': lights, "Slides": slides, 'HingeDoors': hinge_doors,'Microwave': microwave,'Kettle': kettle}


OBJECT_INDS = { # 'bottom burner': np.array([-19, -18]),
                # 'top burner': np.array([-15, -14]),
  # 'burners': np.array([-19, -18,-15,-14]),
  'burners': np.array([-19, -18,-17,-16,-15,-14]),
 'light switch': np.array([-13, -12]),
 'slide cabinet': np.array([-11]),
 # 'hinge cabinet': np.array([-10,  -9]),
 'hinge cabinet': np.array([-9]),
 'microwave': np.array([-8]),
 'kettle': np.array([-7, -6, -5, -4, -3, -2, -1])}

OBJECT_THRESHOLDS = { # 'bottom burner': np.array([-19, -18]),
                # 'top burner': np.array([-15, -14]),
  'burners': 0,
 'light switch': 0,
 # half open, full open at 0.443
 'slide cabinet': 0.248,

 # one open with fully inserted
 'hinge cabinet': 0.42,
 'microwave': 0.7,
 'kettle': 0}



@configurable(pickleable=True)
class KitchenV0(robot_env.RobotEnv):

    CALIBRATION_PATHS = {
        'default':
        os.path.join(os.path.dirname(__file__), 'robot/franka_config.xml')
    }
    # Converted to velocity actuation
    ROBOTS = {'robot': 'd4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_VelAct'}
    MODEL_ARM = os.path.join(
        os.path.dirname(__file__),
        '../franka/assets/franka_kitchen_jntpos_act_ab.xml')
    MODEL_ARM_NO_KETTLE = os.path.join(
        os.path.dirname(__file__),
        '../franka/assets/franka_kitchen_jntpos_act_ab_no_kettle.xml')
    MODEL_HOOK = os.path.join(
        os.path.dirname(__file__),
        '../franka/assets/franka_kitchen_jntpos_act_ab_hook.xml')
    MODEL_HOOK_NO_KETTLE = os.path.join(
        os.path.dirname(__file__),
        '../franka/assets/franka_kitchen_jntpos_act_ab_hook_no_kettle.xml')
    MODEL_HOOK_MINIMAL = os.path.join(
        os.path.dirname(__file__),
        '../franka/assets/franka_kitchen_jntpos_act_ab_hook_minimal.xml')
    # N_DOF_ROBOT = 9
    N_DOF_ROBOT = 9
    N_DOF_HOOK = 3
    N_DOF_OBJECT = 21

    def __init__(self, robot_params={}, frame_skip=40, use_velocity=False, further=False, hook=False,minimal=False,no_kettle=False,displacement_reward=False,object_reward=None,object_reward_sparse=False,end_effector_pos=False,init_qpos=None,value_augment=None,gamma=0.99,sparse_term=False):
        self.end_effector_pos = end_effector_pos
        self.goal_concat = not end_effector_pos
        self.use_velocity = use_velocity
        self.obs_dict = {}
        # self.robot_noise_ratio = 0.1  # 10% as per robot_config specs
        self.robot_noise_ratio = 0.0  # 10% as per robot_config specs
        self.goal = np.zeros((30,))
        self.episode_collisions = {k:0 for k,v in bodies.items()}
        self.hook=hook
        self.no_kettle = no_kettle
        self.displacement_reward = displacement_reward
        self.N_DOF = self.N_DOF_HOOK if hook else self.N_DOF_ROBOT
        self.object_reward=object_reward
        self.object_reward_sparse=object_reward_sparse
        self.value_augment = value_augment
        self.gamma = gamma
        self.last_value = 0
        self.sparse_term=sparse_term
        self.ep_success = False
        if hook:
            if minimal:
                model_file = self.MODEL_HOOK_MINIMAL
            elif no_kettle:
                model_file = self.MODEL_HOOK_NO_KETTLE
            else:
                model_file = self.MODEL_HOOK
        else:
            if no_kettle:
                model_file = self.MODEL_ARM_NO_KETTLE
            else:
                model_file = self.MODEL_ARM
        print("loading: ",model_file)


        super().__init__(
            model_file,
            robot=self.make_robot(
                n_jnt=self.N_DOF,  #root+robot_jnts
                n_obj=self.N_DOF_OBJECT,
                **robot_params),
            frame_skip=frame_skip,
            camera_settings=dict(
                distance=4.5,
                azimuth=-66,
                elevation=-65,
            ),
        )
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        # For the microwave kettle slide hinge
        self.init_qpos = np.array([ 1.48388023e-01, -1.76848573e+00,  1.84390296e+00, -2.47685760e+00,
                                    2.60252026e-01,  7.12533105e-01,  1.59515394e+00,  4.79267505e-02,
                                    3.71350919e-02, -2.66279850e-04, -5.18043486e-05,  3.12877220e-05,
                                   -4.51199853e-05, -3.90842156e-06, -4.22629655e-05,  6.28065475e-05,
                                    4.04984708e-05,  4.62730939e-04, -2.26906415e-04, -4.65501369e-04,
                                   -6.44129196e-03, -1.77048263e-03,  1.08009684e-03, -2.69397440e-01,
                                    3.50383255e-01,  1.61944683e+00,  1.00618764e+00,  4.06395120e-03,
                                   -6.62095997e-03, -2.68278933e-04])
        if self.hook:
            self.init_qpos = np.array([ 0,  0, 3.71350919e-02, 
                                   -2.66279850e-04, -5.18043486e-05,  3.12877220e-05,
                                   -4.51199853e-05, -3.90842156e-06, -4.22629655e-05,  6.28065475e-05,
                                    4.04984708e-05,  4.62730939e-04, -2.26906415e-04, -4.65501369e-04,
                                   -6.44129196e-03, -1.77048263e-03,  1.08009684e-03, -2.69397440e-01,
                                    3.50383255e-01,  1.61944683e+00,  1.00618764e+00,  4.06395120e-03,
                                   -6.62095997e-03, -2.68278933e-04])

        self.init_qvel = self.sim.model.key_qvel[0].copy()
        if further:
            self.init_qpos[-6] += 0.5
            self.init_qpos[-7] += 0.5

        if init_qpos is not None:
            self.init_qpos = init_qpos

        self.act_mid = np.zeros(self.N_DOF)
        self.act_amp = 2.0 * np.ones(self.N_DOF)

        act_lower = -1*np.ones((self.N_DOF,))
        act_upper =  1*np.ones((self.N_DOF,))
        self.action_space = spaces.Box(act_lower, act_upper)

        # if use_velocity:
            # self.obs_dim = self.init_qvel.shape[0]
            # import pdb; pdb.set_trace()

        obs_upper = 8. * np.ones(self.obs_dim)
        obs_lower = -obs_upper
        self.observation_space = spaces.Box(obs_lower, obs_upper)

    def _get_reward_n_score(self, obs_dict):
        raise NotImplementedError()

    # use 'end_effector' or "microhandle_site", 'hinge_site_2', 'hinge_site_1'
    def site_location(self,site):
        efsid = self.model.site_name2id(site)
        ef_pos = self.data.site_xpos[efsid]
        return ef_pos

    def step(self, a, b=None):
        a = np.clip(a, -1.0, 1.0)

        if not self.initializing:
            a = self.act_mid + a * self.act_amp  # mean center and scale
        else:
            self.goal = self._get_task_goal()  # update goal if init

        self.robot.step(
            self, a, step_duration=self.skip * self.model.opt.timestep)
        if self.no_kettle:
            self.sim.data.qpos[OBJECT_INDS['kettle']] = self.init_qpos[OBJECT_INDS['kettle']]
            self.sim.data.qvel[OBJECT_INDS['kettle']] = 0

        # observations
        obs = self._get_obs()

        #rewards
        reward_dict, score = self._get_reward_n_score(self.obs_dict)

        # termination
        done = False

        done_this_frame = []
        for contact in self.data.contact:
            g1 = self.data.contact[0].geom1
            g2 = self.data.contact[0].geom2
            b1 = self.model.id2name(self.model.geom_bodyid[g1],'body')
            b2 = self.model.id2name(self.model.geom_bodyid[g2],'body')
            non_agent = None
            if b1 in agent_bodies:
                non_agent = b2
            elif b2 in agent_bodies:
                non_agent = b1
            if non_agent is not None:
                for k,v in bodies.items():
                    if non_agent in v and k not in done_this_frame:
                        self.episode_collisions[k] += 1
                        done_this_frame.append(k)

        deltas = {}
        for k,inds in OBJECT_INDS.items():
            # disp = ((self.init_qpos[inds]-obs[inds])**2).sum()
            deltas[k] = self.init_qpos[inds]-obs[inds]

        efsid = self.model.site_name2id('end_effector')
        ef_pos = self.data.site_xpos[efsid]

        # finalize step
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
            'rewards': reward_dict,
            'score': score,
            # 'images': np.asarray(self.render(mode='rgb_array')),
            'collisions': self.episode_collisions,
            'displacements': deltas,
            'end_effector_pos': ef_pos,
        }
        if self.displacement_reward:
            reward = sum([(x**2).sum() for x in deltas.values()])
        elif self.object_reward is not None:
            sparse_success = sum([1 if abs(x) > OBJECT_THRESHOLDS[self.object_reward] else 0 for x in deltas[self.object_reward]])
            self.ep_success = self.ep_success or sparse_success >= 1
            if self.object_reward_sparse:
                reward =  5*sparse_success
                if self.sparse_term:
                    done = reward >= 1
            else:
                reward = sum(abs(deltas[self.object_reward]))
        else:
            reward = reward_dict['r_total'] 

        env_info['is_success'] = self.ep_success
        if self.value_augment is not None:
            next_value = self.value_augment(obs)
            # reward += self.gamma*next_value - self.last_value
            reward += next_value - self.last_value
            self.last_value = next_value

        return obs, reward, done, env_info

    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        self.obs_dict = {}
        self.obs_dict['t'] = t
        self.obs_dict['qp'] = qp
        self.obs_dict['qv'] = qv
        self.obs_dict['obj_qp'] = obj_qp
        self.obs_dict['obj_qv'] = obj_qv
        self.obs_dict['goal'] = self.goal

        if self.end_effector_pos:
            efsid = self.model.site_name2id('end_effector')
            ef_pos = self.data.site_xpos[efsid]
            # patch out burners and switch
            objp = obj_qp.copy()
            inds = list(OBJECT_INDS['burners']) + list(OBJECT_INDS['light switch'])
            for ind in inds:
                objp[ind] = self.init_qpos[ind]
            return np.concatenate((ef_pos,objp))

        if self.goal_concat:
            if self.use_velocity:
                return np.concatenate([self.obs_dict[x] for x in ['qp','obj_qp','qv','obj_qv']])
                # return np.concatenate([self.obs_dict[x] for x in ['qp','obj_qp','qv','obj_qv','goal']])
            else:
                return np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp']])
                # return np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp'], self.obs_dict['goal']])

    def reset_model(self):
        reset_pos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        self.goal = self._get_task_goal()  #sample a new goal on reset
        self.episode_collisions = {k:0 for k,v in bodies.items()}
        if self.value_augment is not None:
            self.last_value = self.value_augment(self._get_obs())
        self.ep_success = False
        return self._get_obs()

    def evaluate_success(self, paths):
        # score
        mean_score_per_rollout = np.zeros(shape=len(paths))
        for idx, path in enumerate(paths):
            mean_score_per_rollout[idx] = np.mean(path['env_infos']['score'])
        mean_score = np.mean(mean_score_per_rollout)

        # success percentage
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            num_success += bool(path['env_infos']['rewards']['bonus'][-1])
        success_percentage = num_success * 100.0 / num_paths

        # fuse results
        return np.sign(mean_score) * (
            1e6 * round(success_percentage, 2) + abs(mean_score))

    def close_env(self):
        self.robot.close()

    def set_goal(self, goal):
        self.goal = goal

    def _get_task_goal(self):
        return self.goal

    # Only include goal
    @property
    def goal_space(self):
        len_obs = self.observation_space.low.shape[0]
        env_lim = np.abs(self.observation_space.low[0])
        return spaces.Box(low=-env_lim, high=env_lim, shape=(len_obs//2,))

    def convert_to_active_observation(self, observation):
        return observation

class KitchenTaskRelaxV1(KitchenV0):
    """Kitchen environment with proper camera and goal setup"""

    def __init__(self,**kwargs):
        super(KitchenTaskRelaxV1, self).__init__(**kwargs)

    def _get_reward_n_score(self, obs_dict):
        reward_dict = {}
        reward_dict['true_reward'] = 0.
        reward_dict['bonus'] = 0.
        reward_dict['r_total'] = 0.
        score = 0.
        return reward_dict, score

    def render(self, mode='human'):
        if mode =='rgb_array':
            camera = engine.MovableCamera(self.sim, 1920, 2560)
            camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            img = camera.render()
            return img
        else:
            super(KitchenTaskRelaxV1, self).render()
