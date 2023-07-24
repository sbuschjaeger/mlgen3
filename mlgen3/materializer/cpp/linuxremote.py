import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from importlib_resources import files

from .linuxstandalone import LinuxStandalone
#from linuxstandalone import LinuxStandalone

class LinuxRemote(LinuxStandalone):

    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self, implementation, filename = None, measure_accuracy=False, measure_time=False, measure_perf=False, compiler = "g++", remote_compile = False, hostname = "localhost", ssh_config=""):
        super().__init__(implementation, filename, measure_accuracy, measure_time, measure_perf, compiler)
        self.remote_compile = remote_compile
        self.hostname = hostname
        self.ssh_config = ssh_config
        self.tmpdir = None

    def beautify(self, s):
        # TODO this seems to die sometimes, especially when the c++-code contains errors 
        try:
            from astyle_py import Astyle
            formatter = Astyle()
            formatter.set_options('--style=google --mode=c --delete-empty-lines')
            return formatter.format(s)
        except ImportError:
            return s

    def materialize(self, path):
        # I dont why we need to call this here. This does not really make sense? Why do we need to store the path? Simply for the deploy step? 
        # if os.path.exists(os.path.join(tempfile.gettempdir(), "mlgen3", "TestLinearClassifierNative")):
        #     shutil.rmtree(os.path.join(tempfile.gettempdir(), "mlgen3", "TestLinearClassifierNative"))

        self.tmpdir = tempfile.TemporaryDirectory()
        super().materialize(self.tmpdir.name)
        self.path = path

        # with open(os.path.join(self.tmpdir.name, self.filename + ".cpp"), 'w') as f:
        #     f.write(self.beautify(self.implementation.code))

        # with open(os.path.join(self.tmpdir.name, self.filename + ".h"), 'w') as f:
        #     f.write(self.beautify(self.implementation.header))
    
    def deploy(self, verbose=False):
        tmp_path = self.path
        assert self.tmpdir is not None, "Did not find any code. Did you call materialize() beforehand?"
        self.path = self.tmpdir.name
        super().deploy()

        self.path = tmp_path
        #if self.remote_compile:
        cfg = "" if self.ssh_config is None else f"-F {self.ssh_config}"
        create_res = subprocess.run(f"ssh {cfg} {self.hostname} 'mkdir -p {os.path.dirname(self.path)}'", capture_output=True, text=True, shell=True)
        if verbose:
            print(f"ssh {cfg} 'mkdir {self.path}'")
            print(f"stdout: \n{create_res.stdout}")
            print(f"stderr: \n{create_res.stderr}")

        scp_res = subprocess.run(f"scp -r {cfg} {self.tmpdir.name} {self.hostname}:{self.path}", capture_output=True, text=True, shell=True)
        if verbose:
            print(f"scp {self.ssh_config} {self.tmpdir.name} {self.hostname}:{self.path}")
            print(f"stdout: \n{scp_res.stdout}")
            print(f"stderr: \n{scp_res.stderr}")
        
    def run(self, verbose = False):
        cfg = "" if self.ssh_config is None else f"-F {self.ssh_config}"
        if self.remote_compile:
            make_res = subprocess.run(f"ssh {cfg} {self.hostname} 'cd {self.path} && make && exit'", capture_output=True, text=True, shell=True)
            if verbose:
                print(f"ssh {cfg} {self.hostname} 'cd {self.path} && make && exit'")
                print(f"stdout: \n{make_res.stdout}")
                print(f"stderr: \n{make_res.stderr}")
        else:
            make_res = subprocess.run(f"cd {self.tmpdir.name} && make", capture_output=True, text=True, shell=True)
            if verbose:
                print(f"Running cd {self.path} && make")
                print(f"stdout: \n{make_res.stdout}")
                print(f"stderr: \n{make_res.stderr}")
        

        run_res = subprocess.run(f"ssh {cfg} {self.hostname} 'cd {self.path} && ./{self.filename} testing.csv 2'", capture_output=True, text=True, shell=True)
        
        if verbose:
            print(f"ssh {cfg} {self.hostname} 'cd {self.path} && ./{self.filename} testing.csv 2'")
            print(f"stdout: \n{run_res.stdout}")
            print(f"stderr: \n{run_res.stderr}")

        metrics = {}
        lines = run_res.stdout.split("\n")
        for cur_line in lines:
            if len(cur_line) > 0:
                l = cur_line.split(":")
                metrics[l[0]] = l[1].split(" ")[1]
        
        return metrics