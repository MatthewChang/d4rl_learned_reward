from gym.envs.registration import register
from d4rl.locomotion import ant
from d4rl.locomotion import maze_env

"""
register(
    id='antmaze-umaze-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)
"""

register(
    id='antmaze-umaze-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-umaze-dense-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'dense',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-umaze-half-dense-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_HALF,
        'reward_type':'dense',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-umaze-three-fourths-dense-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_THREE_FOURTHS,
        'reward_type':'dense',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

# Custom
register(
    id='antmaze-umaze-learned-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'learned',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-umaze-diverse-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-play-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.MEDIUM_MAZE,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-diverse-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.MEDIUM_MAZE,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-close-diverse-dense-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.MEDIUM_MAZE_CLOSE,
        'reward_type':'dense',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-super-close-diverse-dense-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.MEDIUM_MAZE_SUPER_CLOSE,
        'reward_type':'dense',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-super-close-diverse-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.MEDIUM_MAZE_SUPER_CLOSE,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-super-close-valdif-diverse-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.MEDIUM_MAZE_SUPER_CLOSE,
        'reward_type':'valdif',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-super-close-learned-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.MEDIUM_MAZE_SUPER_CLOSE,
        'reward_type':'learned',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-diverse-dense-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.MEDIUM_MAZE,
        'reward_type':'dense',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-large-diverse-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-large-diverse-dense-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'dense',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-large-play-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-point_maze-transfer_learned-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.POINT_MAZE,
        'reward_type':'transfer_learned',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-point_maze_easy-transfer_learned-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.POINT_MAZE_EASY,
        'reward_type':'transfer_learned',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-point_maze-dense-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.POINT_MAZE,
        'reward_type':'dense',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-point_maze_easy-dense-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.POINT_MAZE_EASY,
        'reward_type':'dense',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-point_maze-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.POINT_MAZE,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-point_maze_easy-v0',
    entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.POINT_MAZE_EASY,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)
