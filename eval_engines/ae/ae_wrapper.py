import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint
import yaml
import IPython
from collections import deque
from xml.etree import ElementTree as ET
from subprocess import check_output
from pathlib import Path

from eval_engines.ae.designer import AE_Designer

debug = False

class AeWrapper(object):

    BASE_TMP_DIR = os.path.abspath("./ae_data")
    RUNNABLE_LOAD = 10000
    RUNNABLE_PRIORITY = 7

    def __init__(self, num_process, yaml_path, path, root_dir=None):
        if root_dir == None:
            self.root_dir = AeWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f)
        design_xml = path
        folders = yaml_data['dsn_xml'].split('/')
        for folder in folders:
            design_xml = os.path.join(design_xml, folder)
 
        _, dsg_xml_fname = os.path.split(design_xml)
        self.base_design_name = os.path.splitext(dsg_xml_fname)[0]
        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        self.tree = ET.parse(design_xml)
        self.schema_path = os.path.join(path, 'eval_engines', 'ae', 'S2S_VSE_XSD_schema.xsd')

    def get_design_name(self, state):
        fname = self.base_design_name
        for value in state.values():
            if isinstance(value, tuple) or isinstance(value, list):
                fname += "_" + '-'.join(str(int(x)) for x in value)
            else:
                fname += "_" + str(value)
        return fname

    def create_design(self, state, new_fname):
        design_folder = os.path.join(self.gen_dir, new_fname)+ "_" + str(random.randint(0,10000))
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.xml')

        tree_copy = copy.deepcopy(self.tree)
        root = tree_copy.getroot()
        print("root.tag: {}".format(root.tag)) # AR-PACKAGE
        print("root[0]: {}".format(root[0])) # ELEMENTS
        # TODO: create a new design according to the state
        runnable_names, timing_events = self.get_runnable_names(root)
        self.build_params_for_result_translation(runnable_names, timing_events)
        print("runnable_names: {}".format(runnable_names))
        self.assign_runnables(root, state, runnable_names)
        tree_copy.write(fpath)
        AE_Designer(design_file_path=fpath, schema_path=self.schema_path)
        return design_folder, fpath

    def build_params_for_result_translation(self, runnable_names, timing_events):
        self.algo_names = [runnable_name.split('/')[-1] for runnable_name in runnable_names]
        self.algo_event_intervals = [timing_events.get(runnable_name, 0) for runnable_name in runnable_names]
        print("self.algo_names: {}, self.algo_event_intervals: {}".format(self.algo_names, self.algo_event_intervals))

    def get_runnable_names(self, xml_root):
        runnables = []
        timing_events = {}
        print("ELEMENTS")
        for child in xml_root[0]:
            print("\t",child.tag, child.attrib)
            if child.tag == 'APPLICATION-SW-COMPONENT-TYPE':
                runnable_name = deque()
                for grandchild in child:
                    print("\t\t",grandchild.tag, grandchild.attrib)
                    if grandchild.tag == 'SHORT-NAME':
                        # Get first part of the runnable name
                        runnable_name.append(grandchild.attrib['ID'])
                    if grandchild.tag == 'INTERNAL-BEHAVIORS':
                        swc_internal_behaviour = grandchild[0]
                        print("\t\t\tSWC-INTERNAL-BEHAVIOR")
                        for element in swc_internal_behaviour:
                            print("\t\t\t\t", element.tag, element.attrib)
                            if element.tag == "SHORT-NAME":
                                # Get the second part of the runnable name
                                runnable_name.append(element.attrib['ID'])
                            if element.tag == "RUNNABLES":
                                for runnable in element:
                                    if runnable.tag == 'RUNNABLE-ENTITY':
                                        for runnable_element in runnable:
                                            print("\t\t\t\t\t",runnable_element.tag, runnable_element.attrib)
                                            if runnable_element.tag == 'SHORT-NAME':
                                                runnable_name.append(runnable_element.attrib['ID'])
                                                print("\t\t\t\t\tappending runnable name: {}".format('/'.join(runnable_name)))
                                                runnables.append('/'+'/'.join(runnable_name))
                                                runnable_name.pop()
                                runnable_name.pop()
                            if element.tag == "EVENTS":
                                timing_events1 = self.get_timing_events(element)
                                timing_events = {x: timing_events.get(x, 0.0) + timing_events1.get(x, 0.0) for x in set(timing_events).union(timing_events1)}
                        runnable_name.pop()
        return runnables, timing_events

    def get_timing_events(self, events_element):
        timing_events = {}
        for event in events_element.findall('TIMING-EVENT'):
            print("\t\t\t\t",event.tag, event.attrib)
            mapped_runnable_name = ""
            period = 0.0
            for child in event:
                print("\t\t\t\t\t",child.tag, child.attrib)
                if child.tag == 'SHORT-NAME':
                    print("\t\t\t\t\t\t",child.attrib['ID'])
                if child.tag == 'START-ON-EVENT-REF':
                    mapped_runnable_name = child.attrib['DEST']
                if child.tag == 'PERIOD':
                    period = float(child.attrib['value'])
            timing_events[mapped_runnable_name] = period
        return timing_events

    def get_arch_name(self, arch_index):
        if arch_index == 1:
            return "CortexA53"
        elif arch_index == 2:
            return "CortexA72"
        else:
            raise ValueError("Invalid arch index: {}, supported indexes [1, 2]".format(arch_index))

    def get_core_name(self, arch_index, core_index):
        arch_prefix = ""
        if arch_index == 1:
            arch_prefix = "A53Core" + str(core_index)
        elif arch_index == 2:
            arch_prefix = "A72Core" + str(core_index)
        else:
            raise ValueError("Invalid arch index: {}, supported indexes [1, 2]".format(arch_index))
        return arch_prefix

    def assign_runnables(self, xml_root, state, runnable_names):
        print("Using state: {} to build CPU clusters".format(state))
        soc_element = None
        for child in xml_root[0]:
            print("\t",child.tag, child.attrib)
            if child.tag == 'ECUs':
                for grandchild in child:
                    print("\t\t",grandchild.tag, grandchild.attrib)
                    if grandchild.tag == 'SoCs':
                        soc_element = grandchild
                        for element in soc_element.findall('CPU_Cluster'):
                            soc_element.remove(element)
        cluster_num = state['cluster_num']
        core_per_cluster = state['core_per_cluster']
        quick_core_index_mapping = [[ET.Element('Core') for j in range(core_per_cluster)] for i in range(cluster_num)]
        print("quick_core_index_mapping: {}".format(quick_core_index_mapping))
        for cluster_index in range(cluster_num):
            cluster = ET.Element('CPU_Cluster')
            arm_family = ET.Element('ARMV8-Family')
            arch_index = state['arch_per_cluster'][cluster_index]
            arch_name = self.get_arch_name(arch_index)
            cluster_arch = ET.Element(arch_name)
            arm_family.append(cluster_arch)
            cluster.append(arm_family)

            shortname = ET.Element('SHORT-NAME')
            shortname.set('ID', arch_name)
            shortname.set('name', arch_name)
            cluster_arch.append(shortname)

            frequency = ET.Element('Frequency')
            frequency.set('name', str(state['freq_per_cluster'][cluster_index]))
            cluster_arch.append(frequency)

            # Add Cores according to core_per_cluster
            for core_index in range(core_per_cluster):
                core = ET.Element('Core')
                shortname = ET.Element('SHORT-NAME')
                core_name = self.get_core_name(arch_index, core_index)
                shortname.set('ID', core_name)
                shortname.set('name', core_name)
                core.append(shortname)
                cluster_arch.append(core)
                quick_core_index_mapping[cluster_index][core_index] = core

            soc_element.append(cluster)
        
        # Assign runnables to the cores in CPU clusters Round Robin
        for i, runnable_name in enumerate(runnable_names):
            cluster_i = int(i / core_per_cluster) % cluster_num
            core_i = int(i % core_per_cluster)
            print("cluster_i: {}, core_i: {}, runnable_name: {}".format(cluster_i, core_i, runnable_name))
            runnable_element = quick_core_index_mapping[cluster_i][core_i]
            mapping_element = ET.Element('Core-runnable-Mapping')
            mapping_element.set('DEST', runnable_name)
            mapping_element.set('Load', str(AeWrapper.RUNNABLE_LOAD))
            mapping_element.set('priority', str(AeWrapper.RUNNABLE_PRIORITY))
            runnable_element.append(mapping_element)


    def simulate(self, fpath, design_folder):
        
        workding_directory_path = os.path.join(design_folder, "ae_run", "WD")
        Path(workding_directory_path).mkdir(parents=True, exist_ok=True)
        print("Starting AE Engine")
        start_time = time.time()
        #workding_directory = "C:\\development\\AE\\Architecture_Explorer\\AutoCkt\\ae_data\\designs_SimpleExample\\SimpleExample_1_4_200000000_1_5364\\ae_run\\WD"
        ae_engine_output = check_output("C:\\siemens\\SystemExplorer\\Automation_Engine\\AE_Engine.exe all --working_dir %s --root_dir C:\\siemens\\SystemExplorer --xml_file %s --xsd_schema C:\\siemens\\SystemExplorer\\config\\VSE_XSD_Schema\\S2S_VSE_XSD_schema.xsd --vista C:\\siemens\\VirtualPlatform --nucleus C:\\siemens" % (workding_directory_path, fpath))
        end_time = time.time()
        print("Done")
        print("AE_Engine took {} seconds".format(end_time - start_time))
        
        return ae_engine_output


    def create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        if dsn_name == None:
            dsn_name = self.get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print(dsn_name)
        design_folder, fpath = self.create_design(state, dsn_name)
        info = self.simulate(fpath, design_folder)
        specs = self.translate_result(design_folder)
        return state, specs, info


    def run(self, states, design_names=None, verbose=False):
        """

        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self.create_design_and_simulate, arg_list)
        pool.close()
        return specs

    def translate_result(self, output_path):
        """
        This method needs to be overwritten according to cicuit needs,
        parsing output, playing with the results to get a cost function, etc.
        The designer should look at his/her netlist and accordingly write this function.

        :param output_path:
        :return:
        """
        result = None
        return result
