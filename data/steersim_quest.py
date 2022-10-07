import subprocess 
import logging 
import numpy as np 

from multiprocessing import Pool 
from os import environ, makedirs 
from subprocess import Popen 
from dotenv import load_dotenv 

logger = logging.getLogger(__name__)

def steersim_call(query, env):
    steersim_command_path = env["SteersimCommandPath"]
    steersim_command_exec = env["SteersimCommandExec"]
    p = Popen(steersim_command_exec.split(), cwd=steersim_command_path,
              stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    stdout_line, stderr_line = p.communicate(input=query.encode())
    p.wait()
    if p.returncode != 0:
        logging.error('From subprocess out: %r', stdout_line)
        logging.error('From subprocess err: %r', stderr_line)


def steersim_call_parallel(queries, generate_for_testcases=False, subfix=""):
    """
    Steersim arguments must be a list of numbers [len_query, len_parameters]
    """
    SteersimRecordPath = "SteersimRecordPath"
    env = environ.copy()
    if generate_for_testcases:
        env[SteersimRecordPath] = env[SteersimRecordPath] + "/test" + subfix

    makedirs(env[SteersimRecordPath], exist_ok=True)
    queries = np.clip(queries, 0, 1)
    query_strings = [" ".join([str(x) for x in numbers]) for numbers in queries]
    with Pool() as p:
        p.starmap(steersim_call, [(q, env) for q in query_strings])


def initial_sample_steersim():
    import os
    import shutil
    import numpy as np
    from dotenv import load_dotenv

    load_dotenv(verbose=True)
    # ask_for_regenerate = input("Remove steersim record path and regen?")
    # if ask_for_regenerate == "n":
    #     return
    shutil.rmtree(os.getenv("SteersimRecordPath"), ignore_errors=True)
    os.makedirs(os.getenv("SteersimRecordPath"), exist_ok=True)
    numbers = np.random.uniform(0, 1, (5, 43))
    steersim_call_parallel(numbers)
    numbers = np.random.uniform(0, 1, (1, 43))
    steersim_call_parallel(numbers, generate_for_testcases=True)

if __name__ == '__main__':
    initial_sample_steersim()
