import argparse
import json
import os
import gym
import gym_boxworld
from stable_baselines import A2C, ACKTR, ACER
from stable_baselines.common.policies import CnnPolicy, LstmPolicy
from relational_policies import RelationalPolicy, RelationalLstmPolicy  # custom Policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.common.atari_wrappers import FrameStack

from warehouse_env.warehouse_env import WarehouseEnv
import numpy as np

def saveInLearn(log_dir):
    # A unit of time saved
    unit_time = int(1e5)

    def callback(_locals, _globals):
        num_timesteps = _locals['self'].num_timesteps
        if num_timesteps >= 1 * unit_time and num_timesteps % unit_time == 0:
            _locals['self'].save(log_dir + 'model_{}.zip'.format(num_timesteps))
        return True
    return callback


def make_env(env_id, env_level, rank, log_dir, frame_stack=False, useMonitor=True, seed=0, map_file=None, render_as_observation=False,
            exponential_agent_training_curve=False):
    def _init():
        if env_id == "WarehouseEnv":
#             if map_file is "None" or map_file is None:
            simple_agent = np.zeros((11,11)) 
            simple_agent[5,5] = 1
#                      [[ 0, 1,  0,  0,  0,  0,  2, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  3,  0,  0, 0, 0]]
#             simple_agent = \
#                      [[ 0, 1,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0]]
            simple_world = np.zeros((11,11))
#                      [[ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  1,  0,  0, 0, 0],
#                       [ 0, 1,  0,  0,  0,  1,  0, 0, 0],
#                       [ 0, 0,  0,  0,  1,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0],
#                       [ 0, 0,  0,  0,  0,  0,  0, 0, 0]]
            env = WarehouseEnv(agent_map=simple_agent, obstacle_map=simple_world,
                               render_as_observation=render_as_observation, 
                               exponential_agent_training_curve=exponential_agent_training_curve)
        else:
            env = gym.make(env_id, level=env_level)
        if frame_stack:
            env = FrameStack(env, 4)
        if useMonitor:
            env = Monitor(env, log_dir + str(rank), allow_early_resets=True)
        return env

    set_global_seeds(seed)
    return _init


def set_logdir(config):
    log_dir = '{}/{}_{}_{}/log_0/'.format(config.log_dir, config.env_name, config.model_name, config.policy_name)
    # if log_dir exists,auto add new dir by order
    while os.path.exists(log_dir):
        lastdir_name = log_dir.split('/')[-2]
        order = int(lastdir_name.split('_')[-1])
        log_dir = log_dir.replace('_{}'.format(order), '_{}'.format(order + 1))
    os.makedirs(log_dir)
    with open(log_dir + 'config.txt', 'wt') as f:
        json.dump(config.__dict__, f, indent=2)
    print(("--------------------------Create dir:{} Successful!--------------------------\n").format(log_dir))
    return log_dir


def set_env(config, log_dir):
    env_id = "WarehouseEnv" if config.env_name == "WarehouseEnv" else config.env_name + 'NoFrameskip-v4'
    env = SubprocVecEnv([make_env(env_id, config.env_level, i, log_dir, frame_stack=config.frame_stack, map_file=config.map_file,
                                 render_as_observation=config.render_as_observation) for i in range(config.num_cpu)])
    return env


def set_model(config, env, log_dir):
    if config.timeline:
        from timeline_util import _train_step
        A2C.log_dir = log_dir
        A2C._train_step = _train_step
    policy = {'CnnPolicy': CnnPolicy, 'LstmPolicy': LstmPolicy, 'RelationalPolicy': RelationalPolicy, 'RelationalLstmPolicy': RelationalLstmPolicy}
    base_mode = {'A2C': A2C, "ACKTR": ACKTR, "ACER": ACER}
    # whether reduce oberservation
    policy[config.policy_name].reduce_obs = config.reduce_obs
    n_steps = config.env_steps
    policy_kwargs = dict(feature_extraction=(config.render_as_observation))
    model = base_mode[config.model_name](policy[config.policy_name], env, verbose=1, 
                                         tensorboard_log=log_dir, 
                                         n_steps=n_steps, policy_kwargs=policy_kwargs) 
#                                          priming_steps=config.priming_steps, 
#                                          coordinated_planner=config.coordinated_planner)
    print(("--------Algorithm:{} with {} num_cpu:{} total_timesteps:{} Start to train!--------\n")
          .format(config.model_name, config.policy_name, config.num_cpu, config.total_timesteps))
    return model


def run(config):
    log_dir = set_logdir(config)
    env = set_env(config, log_dir)
    model = set_model(config, env, log_dir)
    model.learn(total_timesteps=int(config.total_timesteps), callback=saveInLearn(log_dir) if config.save else None)
    if config.save:
        model.save(log_dir + 'model.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", choices=['BoxRandWorld', 'BoxWorld', "WarehouseEnv"], 
                        help="Name of environment")
    parser.add_argument("-env_level", choices=['easy', 'medium', 'hard'], default='easy', 
                        help="level of environment")
    
    parser.add_argument("-map_file", default='None', type=str, help="Map file")
    
    parser.add_argument("policy_name", choices=['RelationalPolicy', 'CnnPolicy', 'RelationalLstmPolicy', 'LstmPolicy'], 
                        help="Name of policy")
    
    parser.add_argument("-model_name", choices=['A2C', 'ACER', 'ACKTR'], 
                        default='A2C', help="Name of model")
    parser.add_argument("-reduce_obs", action='store_true')

    parser.add_argument("-timeline", action='store_true', help='performance analysis,default=False')
    parser.add_argument("-frame_stack", action='store_true', help='whether use frame_stack, default=False')
    parser.add_argument("-cuda_device", default='1', help='which cuda device to run, default="1"')
    parser.add_argument("-num_cpu", default=4, type=int, help='number of CPUs')
    parser.add_argument("-total_timesteps", default=2e6, type=float, help='total train timesteps, default=2e6')
    parser.add_argument("-log_dir", default='exp_result', help='log_dir path, default="exp_result"')
    parser.add_argument("-save", action='store_true', help='whether save model to log_dir, default=False')
    
    parser.add_argument("-env_steps", default=50, type=int, help='num steps, default="50"')
    parser.add_argument("-priming_steps", default=1000, type=int, help='priming steps, default="1000"')
    parser.add_argument("-coordinated_planner", action='store_true', help='whether to use a coordinated_planner, default false')
    parser.add_argument("-render_as_observation", action='store_true', help='whether to use a render_as_observation, default false')
    parser.add_argument("-delta_tolling", action='store_true', help='whether to use a delta_tolling, default false')
    parser.add_argument("-random_agent_reset_location", action='store_true', help='whether to use a random_agent_reset_location, default false')
    parser.add_argument("-exponential_agent_training_curve", action='store_true', help='whether to use a exponential_agent_training_curve, default false')

    config = parser.parse_args()
    # print(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_device
    run(config)
    print('Over!')
