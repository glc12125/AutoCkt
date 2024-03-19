"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces

import numpy as np
import random
import psutil

from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import os
import IPython
import itertools
from eval_engines.util.core import *
import pickle
import os
import itertools

from eval_engines.ae.ArchitectExplorer import *

#way of ordering the way a yaml file is read
class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                                                    'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

class ArchitectExplorerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    PERF_LOW = -10000000
    PERF_HIGH = 10000000

    #obtains yaml file
    path = os.getcwd()
    CIR_YAML = os.path.join(path, 'eval_engines', 'ae', 'ae_inputs', 'yaml_files', 'simple_example.yaml')

    def __init__(self, env_config):
        self.multi_goal = env_config.get("multi_goal",False)
        self.generalize = env_config.get("generalize",False)
        num_valid = env_config.get("num_valid",50)
        self.specs_save = env_config.get("save_specs", False)
        self.valid = env_config.get("run_valid", False)

        self.env_steps = 0
        with open(ArchitectExplorerEnv.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        # design specs
        if self.generalize == False:
            specs = yaml_data['target_specs']
        else:
            load_specs_path = os.path.join(ArchitectExplorerEnv.path, "autockt", "gen_specs", "ae_specs_gen_simple_example.pkl")
            with open(load_specs_path, 'rb') as f:
                specs = pickle.load(f)
            
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        if self.specs_save:
            with open("specs_"+str(num_valid)+str(random.randint(1,100000)), 'wb') as f:
                pickle.dump(self.specs, f)
        
        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        print("self.specs_id: {}".format(self.specs_id))

        self.fixed_goal_idx = -1 
        self.num_os = len(list(self.specs.values())[0])
        
        self.num_process = yaml_data['num_process']
        self.build_params(yaml_data)
        
        #initialize sim environment
        self.sim_env = ArchitectExplorer(yaml_path=ArchitectExplorerEnv.CIR_YAML, num_process=int(self.num_process), path=ArchitectExplorerEnv.path) 
        self.action_meaning = [-1,0,2] 
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_meaning))]*len(self.params_id))
        #print("self.action_space: {}".format(self.action_space))
        #self.action_space = spaces.Discrete(len(self.action_meaning)**len(self.params_id))
        self.observation_space = spaces.Box(
            low=np.array([ArchitectExplorerEnv.PERF_LOW]*2*len(self.specs_id)+len(self.params_id)*[0]),
            high=np.array([ArchitectExplorerEnv.PERF_HIGH]*2*len(self.specs_id)+[3, 7, 624, 15]), dtype=np.float64)

        #initialize current param/spec observations
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)

        #Get the g* (overall design spec) you want to reach
        self.global_g = []
        #print("list(self.specs.values()): {}".format(list(self.specs.values())))
        for spec in list(self.specs.values()):
                self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.g_star = np.array(self.global_g)

        self.global_g = np.array(yaml_data['normalize'])
        
        #objective number (used for validation)
        self.obj_idx = 0

        #print("self.specs: {}".format(self.specs))

    def build_params(self, yaml_data):
        # param array
        self.runnable_num = yaml_data['runnable_num']
        self.runnable_load = yaml_data['runnable_load']
        params = yaml_data['params']
        self.params = {}
        self.params_id = list(params.keys())

        #for value in params.values():
        #    param_vec = np.arange(value[0], value[1], value[2])
        #    self.params.append(param_vec)
        
        for key, value in params.items():
            if key in ['cluster_num', 'core_per_cluster']:
                self.params[key] = np.arange(value[0], value[1], value[2])
            if key in ['freq_per_cluster', 'arch_per_cluster']:
                param_vec = np.arange(value[0], value[1], value[2])
                self.params[key] = {}
                for cluster_num in self.params['cluster_num']:
                    iterables = [param_vec]*cluster_num
                    self.params[key][cluster_num] = [t for t in itertools.product(*iterables)]


    def reset(self):
        #if multi-goal is selected, every time reset occurs, it will select a different design spec as objective
        if self.generalize == True:
            if self.valid == True:
                if self.obj_idx > self.num_os-1:
                    self.obj_idx = 0
                idx = self.obj_idx
                self.obj_idx += 1
            else:
                idx = random.randint(0,self.num_os-1)
            self.specs_ideal = []
            for spec in list(self.specs.values()):
                self.specs_ideal.append(spec[idx])
            self.specs_ideal = np.array(self.specs_ideal)
        else:
            if self.multi_goal == False:
                self.specs_ideal = self.g_star 
            else:
                idx = random.randint(0,self.num_os-1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)

        #print("num total:"+str(self.num_os))

        #applicable only when you have multiple goals, normalizes everything to some global_g
        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)
        print("Resetting, using {}th spec, initialize specs_dieal to {}, specs_ideal_norm to {}, global_g: {}".format(idx, self.specs_ideal, self.specs_ideal_norm, self.global_g))
        #initialize current parameters
        #self.cur_params_idx = np.array([33, 33, 33, 33, 33, 14, 20])
        self.cur_params_idx = np.array([1, 1, 11, 1])
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)

        #observation is a combination of current specs distance from ideal, ideal spec, and current param vals
        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])
        self.env_steps = 0

        return self.ob
 
    def step(self, action):
        """
        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
        """
        #print("self.params: {}".format(self.params))
        #for i, param in enumerate(self.params):
        #    print("len(param): {}".format(len(param)))
        #print("self.params_id: {}".format(self.params_id))
        #Take action that RL agent returns to change current params
        action = list(np.reshape(np.array(action),(np.array(action).shape[0],)))
        #print("action: {}".format(action))
        self.cur_params_idx = self.cur_params_idx + np.array([self.action_meaning[a] for a in action])
        #print("self.cur_params_idx: {}".format(self.cur_params_idx))
        #self.cur_params_idx = self.cur_params_idx + np.array(self.action_arr[int(action)])
        #self.cur_params_idx = np.clip(self.cur_params_idx, [0]*len(self.params_id), [(len(param_vec)-1) for param_key, param_vec in self.params.items()])
        self.cur_params_idx = np.clip(self.cur_params_idx, a_min = [0]*len(self.params_id), a_max = [len(self.params['cluster_num']) - 1, len(self.params['core_per_cluster']) - 1, len(self.params['freq_per_cluster'][len(self.params['cluster_num']) - 1]) - 1, len(self.params['arch_per_cluster'][len(self.params['cluster_num']) - 1]) - 1] )
        max_core_num = int(self.runnable_num / self.params['cluster_num'][self.cur_params_idx[0]])
        self.cur_params_idx = np.clip(self.cur_params_idx, a_min = [0]*len(self.params_id), a_max = [len(self.params['cluster_num']) - 1, min(max_core_num - 1, len(self.params['core_per_cluster']) - 1), len(self.params['freq_per_cluster'][self.params['cluster_num'][self.cur_params_idx[0]]]) - 1, len(self.params['arch_per_cluster'][self.params['cluster_num'][self.cur_params_idx[0]]]) - 1] )
        
        #print("self.cur_params_idx 2: {}".format(self.cur_params_idx))
        #Get current specs and normalize
        self.cur_specs = self.update(self.cur_params_idx)
        #print("self.cur_specs: {}".format(self.cur_specs))
        cur_spec_norm  = self.lookup(self.cur_specs, self.global_g)
        #print("cur_spec_norm: {}".format(cur_spec_norm))
        reward = self.reward(self.cur_specs, self.specs_ideal)
        #print("reward: {}".format(reward))
        done = False

        #incentivize reaching goal state
        if (reward >= 10):
            done = True
            print('-'*10)
            print('params = ', self.cur_params_idx)
            print('specs:', self.cur_specs)
            print('ideal specs:', self.specs_ideal)
            print('re:', reward)
            print('-'*10)
        #print("self.specs_ideal: {}".format(self.specs_ideal))
        print("env steps: {}".format(self.env_steps))
        print("self.specs_ideal_norm: {}".format(self.specs_ideal_norm))
        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])
        print("observations: {}".format(self.ob))
        self.env_steps = self.env_steps + 1

        print('cur specs:' + str(self.cur_specs))
        print('ideal spec:' + str(self.specs_ideal))
        print('reward: {}'.format(reward))
        return self.ob, reward, done, {}

    def lookup(self, spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec-goal_spec)/(goal_spec+spec)
        for idx, goal_min_diff in enumerate(goal_spec):
            # assuming idle_percentage_min_diff after ordered by key
            if idx == 1:
                max_idle = goal_spec[0] + goal_min_diff
                spec_idle = spec[0]
                norm_spec[1] = (spec_idle-max_idle)/(max_idle+spec_idle)
        return norm_spec
    
    def reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting some spec, is negative
        '''
        print("spec: {}, goal_spec: {}".format(spec, goal_spec))
        rel_specs = self.lookup(spec, goal_spec)
        print("rel_specs: {}".format(rel_specs))
        pos_val = [] 
        reward = 0.0
        for i,rel_spec in enumerate(rel_specs):
            if(self.specs_id[i] == 'mean_interval_deviation'):
                rel_spec = rel_spec*-1.0 # the smaller the better, smaller negative value is better. If it is positive, meaning it overshoots, do not penalize
            if(self.specs_id[i] == 'max_interval_deviation_diff'):
                rel_spec = rel_spec*-1.0 # the more stable the better, smaller negative value is better. If it is positive, meaning it overshoots, do not penalize
            if(self.specs_id[i] == 'idle_percentage_min_diff'):
                rel_spec = rel_spec*-1.0 # the less the better, smaller negative value is better. If it is positive, meaning it overshoots, penalize
            
            #if(self.specs_id[i] == 'idle_percentage'):
            #    rel_spec = rel_spec*-1.0 # Use the resource as much as possible, smaller negative value is better. If it is positive, meaning it overshoots, do not penalize
            if rel_spec < 0:
                reward += rel_spec
                pos_val.append(0)
            else:
                pos_val.append(1)

        return reward if reward < -0.02 else 10

    def update(self, params_idx):
        """

        :param action: an int between 0 ... n-1
        :return:
        """
        #impose constraint tail1 = in
        #params_idx[0] = params_idx[3]

        #params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        #print("in update, params: {}".format(params))
        #print("in update, params_idx: {}".format(params_idx))
        #param_val = [OrderedDict(list(zip(self.params_id,params)))]
        
        ##run param vals and simulate
        #cur_specs = OrderedDict(sorted(self.sim_env.create_design_and_simulate(param_val[0])[1].items(), key=lambda k:k[0]))
        #cur_specs = np.array(list(cur_specs.values()))


        cluster_num = self.params['cluster_num'][params_idx[0]]
        core_per_cluster = self.params['core_per_cluster'][params_idx[1]]
        freq_per_cluster = self.params['freq_per_cluster'][cluster_num][params_idx[2]]
        arch_per_cluster = self.params['arch_per_cluster'][cluster_num][params_idx[3]]
        params = [cluster_num, core_per_cluster, freq_per_cluster, arch_per_cluster]
        #print("in update, params: {}".format(params))
        #print("in update, params_idx: {}".format(params_idx))
        param_val = [OrderedDict(list(zip(self.params_id,params)))]
        ordered_specs = sorted(self.sim_env.create_design_and_simulate(param_val[0])[1].items(), key=lambda k:k[0])
        cur_specs = OrderedDict(ordered_specs)
        print("ordered_specs: {}".format(cur_specs))
        cur_specs = np.array(list(cur_specs.values()))

        return cur_specs

def main():
  env_config = {"generalize":True, "valid":True}
  env = ArchitectExplorerEnv(env_config)
  env.reset()
  #env.step([2,2,2,2,2,2,2])

  IPython.embed()

if __name__ == "__main__":
  main()
