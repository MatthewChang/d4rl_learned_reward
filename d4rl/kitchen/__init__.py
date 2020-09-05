from .kitchen_envs import KitchenMicrowaveKettleLightSliderV0, KitchenMicrowaveKettleBottomBurnerLightV0
import numpy as np
from gym.envs.registration import register
max_episode_steps = 500

# Smaller dataset with only positive demonstrations.
register(
    id='kitchen-complete-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

register(
    id='kitchen-end-effector-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
        'end_effector_pos': True,
        'object_reward': 'microwave',
        'object_reward_sparse': True
    }
)

register(
    id='kitchen-end-effector-term-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
        'end_effector_pos': True,
        'object_reward': 'microwave',
        'object_reward_sparse': True,
        'sparse_term': True
    }
)

register(
    id='kitchen-end-effector-easy-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
        'end_effector_pos': True,
        'object_reward': 'microwave',
        'init_qpos': np.array([-9.7239339e-01, -1.7703724e+00,  1.8539937e+00, -1.6975234e+00,
       -4.5489213e-01,  1.2981732e+00,  2.3286870e+00,  4.7824331e-02,
        3.8079061e-02,  2.1415818e-04,  7.0790804e-05,  1.9465293e-05,
       -2.8676635e-05,  2.2976839e-05, -5.2659112e-07, -1.8751658e-05,
        4.4612356e-05,  1.5523852e-04, -3.2345843e-04,  4.9898797e-04,
       -3.1040160e-03,  5.9322068e-03,  6.5184929e-03, -2.6923415e-01,
        3.5010791e-01,  1.6194193e+00,  1.0047294e+00,  8.8227317e-03,
        7.2694495e-03, -4.0280484e-06], dtype=np.float32)

    }
)

register(
    id='kitchen-end-effector-hook-easy-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
        'end_effector_pos': True,
        'object_reward': 'microwave',
        'init_qpos': np.array([-9.7239339e-01, -1.7703724e+00,  1.8539937e+00, -1.6975234e+00,
       -4.5489213e-01,  1.2981732e+00,  2.3286870e+00,  4.7824331e-02,
        3.8079061e-02,  2.1415818e-04,  7.0790804e-05,  1.9465293e-05,
       -2.8676635e-05,  2.2976839e-05, -5.2659112e-07, -1.8751658e-05,
        4.4612356e-05,  1.5523852e-04, -3.2345843e-04,  4.9898797e-04,
       -3.1040160e-03,  5.9322068e-03,  6.5184929e-03, -2.6923415e-01,
        3.5010791e-01,  1.6194193e+00,  1.0047294e+00,  8.8227317e-03,
        7.2694495e-03, -4.0280484e-06], dtype=np.float32)
    }
)

register(
    id='kitchen-further-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
        'further': True
    }
)

register(
    id='kitchen-further-hook-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
        'further': True,
        'hook': True
    }
)

register(
    id='kitchen-complete-velocity-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
        'use_velocity': True
    }
)

# Whole dataset with undirected demonstrations. A subset of the demonstrations
# solve the task.
register(
    id='kitchen-partial-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

# Whole dataset with undirected demonstrations. No demonstration completely
# solves the task, but each demonstration partially solves different
# components of the task.
register(
    id='kitchen-mixed-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)

register(
    id='kitchen-hook-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True
    }
)

register(
    id='kitchen-no-kettle-hook-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'no_kettle': True
    }
)

register(
    id='kitchen-minimal-hook-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'minimal': True
    }
)

register(
    id='kitchen-no-kettle-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'no_kettle': True
    }
)

register(
    id='kitchen-no-kettle-hook-microwave-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'no_kettle': True,
        'object_reward': 'microwave'
    }
)

register(
    id='kitchen-no-kettle-hook-microwave-sparse-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'no_kettle': True,
        'object_reward': 'microwave',
        'object_reward_sparse': True
    }
)

register(
    id='kitchen-no-kettle-hook-microwave-sparse-term-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'no_kettle': True,
        'object_reward': 'microwave',
        'object_reward_sparse': True,
        'sparse_term': True
    }
)

register(
    id='kitchen-end-effector-hook-microwave-sparse-term-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'end_effector_pos': True,
        'object_reward': 'microwave',
        'object_reward_sparse': True,
        'sparse_term': True
    }
)

register(
    id='kitchen-no-kettle-end-effector-hook-microwave-sparse-term-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'end_effector_pos': True,
        'object_reward': 'microwave',
        'object_reward_sparse': True,
        'sparse_term': True,
        'no_kettle': True,
    }
)

register(
    id='kitchen-no-kettle-hook-hinge-doors-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'no_kettle': True,
        'object_reward': 'hinge cabinet'
    }
)

register(
    id='kitchen-no-kettle-hook-hinge-doors-sparse-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'no_kettle': True,
        'object_reward': 'hinge cabinet',
        'object_reward_sparse': True
    }
)


register(
    id='kitchen-no-kettle-hook-slide-door-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'no_kettle': True,
        'object_reward': 'slide cabinet'
    }
)

register(
    id='kitchen-no-kettle-hook-lights-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=max_episode_steps,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'hook': True,
        'no_kettle': True,
        'object_reward': 'light switch'
    }
)
