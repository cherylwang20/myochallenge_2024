import os
import pickle
import time
import sys
sys.path.append('../')
sys.path.append('.')
sys.path.append("../utils")
sys.path.append('utils')

import copy
import numpy as np

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym
from stable_baselines3 import PPO

from utils import RemoteConnection

"""
Define your custom observation keys here
"""
custom_obs_keys = [
    "time", 
    'myohand_qpos',
    'myohand_qvel',
    'pros_hand_qpos',
    'pros_hand_qvel',
    'object_qpos',
    'object_qvel',
    'act',
    "touching_body",
]


def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)

class Policy:

    def __init__(self, env):
        self.action_space = env.action_space

    def __call__(self, env):
        return self.action_space.sample()

def get_custom_observation(rc, obs_keys):
    """
    Use this function to create an observation vector from the 
    environment provided observation dict for your own policy.
    By using the same keys as in your local training, you can ensure that 
    your observation still works.
    """

    obs_dict = rc.get_obsdict()
    # add new features here that can be computed from obs_dict
    # obs_dict['qpos_without_xy'] = np.array(obs_dict['internal_qpos'][2:35].copy())

    return rc.obsdict2obsvec(obs_dict, obs_keys)


time.sleep(10)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8085")
else:
    rc = RemoteConnection("localhost:8085")

#policy = Policy(rc)
path = '/'.join(os.path.realpath('baseline.zip').split('/')[:-1])

root_path = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
print(root_path)
model = PPO.load(os.path.join(root_path, 'baseline'))

print('Loading Manipulation Policy')

# compute correct observation space using the custom keys
shape = get_custom_observation(rc, custom_obs_keys).shape
rc.set_output_keys(custom_obs_keys)

flat_completed = None
trial = 0
while not flat_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    ret = 0

    print(f"MANI-MPL: Start Resetting the environment and get 1st obs of iter {trial}")
    
    obs = rc.reset()

    print(f"Trial: {trial}, flat_completed: {flat_completed}")
    counter = 0
    step = 0
    while not flag_trial:

        ################################################
        ## Replace with your trained policy.
        obs = get_custom_observation(rc, custom_obs_keys)
        action = model.predict(obs)
        action = action[0]
        action[30] = 1
        if step > 130:
            action[32:40] = 0
            action[40:49] = 1

        #hard coding the MPL to the desire position, since we know the actuation of the MPL is the last 17 index of action
        action[-17:] = np.array([-0.65001469 , 1.     ,    -0.23187843 , 0.59583695 , 0.92356688, -0.16,
                                -0.28 ,      -0.88   ,     0.25 ,      -0.846   ,   -0.24981132 ,-0.91823529,
                                -0.945  ,    -0.925   ,   -0.929   ,   -0.49    ,   -0.18      ])
        if step > 250:
            action[-17:] = np.array([-0.4199236 ,  1.      ,   -0.9840558 ,  0.35299219,  0.92356688,  0.02095238,
                                        -0.28    ,   -0.88  ,      0.25      , -0.846     , -0.24981132, -0.91823529,
                                        -0.945   ,   -0.925   ,   -0.929    ,  -0.49     ,  -0.918     ])
        if step > 350:
            action[-17:] = np.array([-0.4199236 ,  1.     ,    -0.9840558,   0.35299219 , 0.3910828 ,  0.02095238,
                                        -0.28    ,   -0.88     ,   0.25   ,    -0.846     , -0.24981132 ,-0.91823529,
                                        -0.945    ,  -0.925    ,  -0.929    ,  -0.49  ,     -0.918     ])
        ################################################

        base = rc.act_on_environment(action)

        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]
        step += 1

        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
