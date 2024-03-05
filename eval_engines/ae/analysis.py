#!/usr/bin/env python

import argparse
import json
import os
import numpy as np
import os
import glob
import time
from collections import deque

FUNCTION_ENTRY_VALUE = "ust.function.entry"
FUNCTION_EXIT_VALUE = "ust.function.exit"

MEAN_INTERVAL_DIFF_METRIC_NAME = "mean_interval_deviation"
MAX_INTERVAL_DIFF_METRIC_NAME = "max_interval_deviation"
CPU_IDLE_PERCENTAGE_METRIC_NAME = "idle_percentage"

def process_function_event(event_time, function_type, this_fn, call_site, cpu_state):
    #print("calling process_function_event event_time: {}, function_type: {}, this_fn: {}, call_site: {}".format(event_time, function_type, this_fn, call_site))
    if function_type == FUNCTION_ENTRY_VALUE:
        if cpu_state['init_time'] == 0:
            cpu_state['init_time'] = int(event_time)
        if len(cpu_state['running_functions']) == 0:
            cpu_state['busy_start_time'] = int(event_time)
        cpu_state['running_functions'].append(int(event_time))
    elif function_type == FUNCTION_EXIT_VALUE:
        cpu_state['running_functions'].pop()
        if len(cpu_state['running_functions']) == 0:
            cpu_state['busy_time'] += int(event_time) - cpu_state['busy_start_time']
        cpu_state['last_exit_time'] = int(event_time)

def parse_function_activity_per_core(core_path, algo_names, algo_event_intervals):
    activity_file = glob.glob(core_path + "/function_activity0.csv")[0]
    #print("\tProcessing activity file: {}".format(activity_file))
    algo_timestamp_mapping = {}
    for algo_name in algo_names:
        algo_timestamp_mapping[algo_name] = []
    with open(activity_file, "r") as file:
        while line := file.readline():
            line = line.rstrip()
            fields = line.split(';')
            is_event_line = fields[0].isnumeric()
            if not is_event_line:
                continue
            if len(fields) < 7:
                continue
            event_time = fields[0]
            function_type = fields[1]
            #if "Rte_IWrite_SWC1_R1_Algo1_swc1PP1_R1Data" in function_type:
            #    #print("R1_Algo1: {}".format(event_time))
            #    time_stamps_algo1.append(int(event_time))
            #if "Rte_IWrite_SWC2_R5_Algo5_SWC2PP1_R5Data" in function_type:
            #    #print("R5_Algo5: {}".format(event_time))
            #    time_stamps_algo5.append(int(event_time))
            for index, algo_name in enumerate(algo_names):
                if algo_name in function_type and algo_event_intervals[index] > 0:
                    algo_timestamp_mapping[algo_name].append(int(event_time))
    core_key = os.path.basename(core_path)
    metrics = {
        core_key: {
            MEAN_INTERVAL_DIFF_METRIC_NAME: [],
            MAX_INTERVAL_DIFF_METRIC_NAME: []
        }
    }

    for index, algo_name in enumerate(algo_names):
        if len(algo_timestamp_mapping[algo_name]) > 0:
            #print("\t\talgo_name: {}".format(algo_name))
            mean_interval_ms = 0.0
            max_interval_ms = 0.0
            counter = 0
            for idx, time in enumerate(algo_timestamp_mapping[algo_name]):
                if idx > 0:
                    prev_time = algo_timestamp_mapping[algo_name][idx - 1]
                    diff = time - prev_time
                    if diff > 0 and diff > algo_event_intervals[index] * 1000000000:
                        #print("\t\t\t{} and prev gap: {}, cur_stamp: {}, prev_stamp: {}".format(idx, diff, time, prev_time))
                        cur_interval_diff_ms = abs(diff - algo_event_intervals[index]*1000000000) / 1000000
                        max_interval_ms = max(max_interval_ms, cur_interval_diff_ms)
                        mean_interval_ms = (mean_interval_ms * counter + cur_interval_diff_ms) / (counter + 1)
                        counter += 1
            metrics[core_key][MEAN_INTERVAL_DIFF_METRIC_NAME] += [mean_interval_ms]
            metrics[core_key][MAX_INTERVAL_DIFF_METRIC_NAME] += [max_interval_ms]
    #print("metrics: {}".format(metrics))

    return metrics

def parse_idle_per_core(core_path):
    cpu_state = {
        "busy_time": 0,
        "busy_start_time": 0,
        "simulation_time": 700000000,
        "idle_time": 700000000,
        "last_exit_time": 0,
        "init_time": 0,
        "last_exit_time": 0,
        "running_functions": deque()
    }
    gantt_file = glob.glob(core_path + "/function_activity_gantt0.csv")[0]
    #print("Processing gantt_file: {}".format(gantt_file))
    cpu_key = os.path.basename(core_path)
    metrics = {
        cpu_key: {}
    }
    with open(gantt_file, "r") as file:
        while line := file.readline():
            line = line.rstrip()
            fields = line.split(';')
            is_event_line = fields[0].isnumeric()
            if not is_event_line:
                continue
            if len(fields) < 8:
                continue
            event_time = fields[0]
            function_type = fields[1]
            this_fn = fields[4].split('=')[-1]
            call_site = fields[5].split('=')[-1]
            process_function_event(event_time, function_type, this_fn, call_site, cpu_state)
    cpu_state['idle_time'] -= cpu_state['busy_time']
    system_time = cpu_state['init_time'] + cpu_state['simulation_time'] - cpu_state['last_exit_time']
    cpu_state['idle_time'] -= system_time
    #print("\tbusy_time: {}, idle_time: {}, system_time: {}".format(cpu_state['busy_time'], cpu_state['idle_time'], system_time))
    #print("\tidle percentage: {}, task busy percentage: {}, system percentage: {}".format(cpu_state['idle_time']/cpu_state['simulation_time'] * 100, cpu_state['busy_time']/cpu_state['simulation_time'] * 100, system_time/cpu_state['simulation_time'] * 100))
    
    metrics[cpu_key]["idle_percentage"] = cpu_state['idle_time']/cpu_state['simulation_time']
    metrics[cpu_key]["busy_percentage"] = cpu_state['busy_time']/cpu_state['simulation_time']
    metrics[cpu_key]["system_percentage"] = system_time/cpu_state['simulation_time']
    
    return metrics
            

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Get metrics given a sim_dir folder generated by Architect Explorer",
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "simdir", type=str, help="path to the generated sim_dir.")
    #required_named = parser.add_argument_group("required named arguments")
    #required_named.add_argument(
    #    "--run",
    #    type=str,
    #    required=True,
    #    help="The algorithm or model to train. This may refer to the name "
    #    "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
    #    "user-defined trainable function or class registered in the "
    #    "tune registry.")
    #required_named.add_argument(
    #    "--env", type=str, help="The gym environment to use.")
    #parser.add_argument(
    #    "--no-render",
    #    default=False,
    #    action="store_const",
    #    const=True,
    #    help="Surpress rendering of the environment.")
    #parser.add_argument(
    #    "--steps", default=10000, help="Number of steps to roll out.")
    #parser.add_argument("--out", default=None, help="Output filename.")
    #parser.add_argument(
    #    "--config",
    #    default="{}",
    #    type=json.loads,
    #    help="Algorithm-specific configuration (e.g. env, hyperparams). "
    #    "Surpresses loading of configuration from checkpoint.")
    #parser.add_argument(
    #    "--num_val_specs",
    #    type=int,
    #    default=50,
    #    help="Number of untrained objectives to test on")
    #parser.add_argument(
    #    "--traj_len",
    #    type=int,
    #    default=60,
    #    help="Length of each trajectory")
    return parser

def translate_result(sim_dir, algo_names, algo_event_intervals):
    metrics = {}
    cpu_clusters = [ f.path for f in os.scandir(sim_dir) if f.is_dir() ]

    for cpu_cluster in cpu_clusters:
        cores_dir = os.path.join(cpu_cluster, "system_analyzer")
        #print("cores_dir: {}".format(cores_dir))
        core_folders = [ f.path for f in os.scandir(cores_dir) if f.is_dir() ]
        #print("core_folders: {}".format(core_folders))
        for core_folder in core_folders:
            #print("Processing core_folder: {}".format(core_folder))
            #metrics = metrics | parse_function_activity_per_core(core_folder, algo_names, algo_event_intervals)
            #metrics = metrics | parse_idle_per_core(core_folder)
            metrics1 = parse_function_activity_per_core(core_folder, algo_names, algo_event_intervals)
            metrics = {x: metrics.get(x, {}) | metrics1.get(x, {})
                    for x in set(metrics).union(metrics1)}
            metrics1 = parse_idle_per_core(core_folder)
            metrics = {x: metrics.get(x, {}) | metrics1.get(x, {})
                    for x in set(metrics).union(metrics1)}
        #parse_function_activity_per_core(core_folders[1], algo_names)
        #parse_idle_per_core(core_folders[1])

    #print("final metrics before merging: {}".format(metrics))

    specs = {
        MEAN_INTERVAL_DIFF_METRIC_NAME: [],
        MAX_INTERVAL_DIFF_METRIC_NAME: [],
        CPU_IDLE_PERCENTAGE_METRIC_NAME: []
    }
    for key, value in metrics.items():
        specs[MEAN_INTERVAL_DIFF_METRIC_NAME] += value[MEAN_INTERVAL_DIFF_METRIC_NAME]
        specs[MAX_INTERVAL_DIFF_METRIC_NAME] += value[MAX_INTERVAL_DIFF_METRIC_NAME]
        specs[CPU_IDLE_PERCENTAGE_METRIC_NAME].append(value['idle_percentage'])

    specs[MEAN_INTERVAL_DIFF_METRIC_NAME] = np.mean(specs[MEAN_INTERVAL_DIFF_METRIC_NAME])
    specs[MAX_INTERVAL_DIFF_METRIC_NAME] = np.max(specs[MAX_INTERVAL_DIFF_METRIC_NAME])
    specs[CPU_IDLE_PERCENTAGE_METRIC_NAME] = np.mean(specs[CPU_IDLE_PERCENTAGE_METRIC_NAME])
    print("final specs: {}".format(specs))

    return specs

def run(args, parser):
    sim_dir = args.simdir
    #cpu_clusters = [ f.path for f in os.scandir("data/simple_sim_dir/sim_dir") if f.is_dir() ]
    cpu_clusters = [ f.path for f in os.scandir(sim_dir) if f.is_dir() ]
    #print("cpu_clusters: {}".format(cpu_clusters))

    algo_names = ["R1_Algo1", "R2_Algo2", "R3_Algo3", "R4_Algo4", "R5_Algo5", "R6_Algo6", "R7_Algo7", "R8_Algo8"]
    algo_event_intervals = [0.07, 0, 0.1, 0, 0.12, 0, 0.14, 0]
    
    start = time.time()
    translate_result(sim_dir, algo_names, algo_event_intervals)
    end = time.time()
    print("translate result time spent: {}".format(end - start))

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)