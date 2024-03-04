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

debug = False

class AeWrapper(object):

    BASE_TMP_DIR = os.path.abspath("./ae_data")

    def __init__(self, num_process, yaml_path, path, root_dir=None):
        if root_dir == None:
            self.root_dir = AeWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f)
        design_xml = yaml_data['dsn_xml']
        design_xml = path+'/'+design_xml
 
        _, dsg_xml_fname = os.path.split(design_xml)
        self.base_design_name = os.path.splitext(dsg_xml_fname)[0]
        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        self.tree = ET.parse(design_xml)

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
        runnables = []
        for child in root[0]:
            print(child.tag, child.attrib)
            if child.tag == 'APPLICATION-SW-COMPONENT-TYPE':
                runnable_name = deque()
                for grandchild in child:
                    print(grandchild.tag, grandchild.attrib)
                    if grandchild.tag == 'SHORT-NAME':
                        # Get first part of the runnable name
                        runnable_name.append(grandchild.attrib['ID'])
                    if grandchild.tag == 'INTERNAL-BEHAVIORS':
                        swc_internal_behaviour = grandchild[0]
                        for element in swc_internal_behaviour:
                            print(element.tag, element.attrib)
                            if element.tag == "SHORT-NAME":
                                # Get the second part of the runnable name
                                runnable_name.append(element.attrib['ID'])
                            if element.tag == "RUNNABLES":
                                for runnable in element:
                                    if runnable.tag == 'RUNNABLE-ENTITY':
                                        for runnable_element in runnable:
                                            print(runnable_element.tag, runnable_element.attrib)
                                            if runnable_element.tag == 'SHORT-NAME':
                                                runnable_name.append(runnable_element.attrib['ID'])
                                                print("appending runnable name: {}".format('/'.join(runnable_name)))
                                                runnables.append('/'.join(runnable_name))
                                                runnable_name.pop()
                                runnable_name.pop()
                        runnable_name.pop()
        print("runnables: {}".format(runnables))

        tree_copy.write(fpath)
        return design_folder, fpath

    def simulate(self, fpath):
        info = 0 # this means no error occurred
        command = "ngspice -b %s >/dev/null 2>&1" %fpath
        exit_code = os.system(command)
        if debug:
            print(command)
            print(fpath)

        if (exit_code % 256):
           # raise RuntimeError('program {} failed!'.format(command))
            info = 1 # this means an error has occurred
        return info


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
        info = self.simulate(fpath)
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
