import yaml
import subprocess
import time
import sys
import os
import PIL
from data.steersim_quest import steersim_call_parallel

#os.environ["PYTHONPATH"] = os.path.realpath(__file__) + ";" + os.getenv("PYTHONPATH", "")
PYTHON_EXECUTABLE = sys.executable

def python_call(file, args):
    print(" ".join(["python", file, args]), flush=True)
    with open("agentformer.log", "a+", buffering=1) as f:
        p = subprocess.Popen([PYTHON_EXECUTABLE, file] + args.split(), stdout=f, stderr=f, bufsize=0)
        while p.poll() is None:
            f.flush()
            time.sleep(0.1)
        if p.returncode != 0:
            f.seek(0)
            print(f.read())
            raise

def agentFormerStepTrain(from_epoch, to_epoch):
    def prepareAgentFormerConfig(**update_dict):
        with open("./cfg/steersim/steersim_pre.yml", "r") as fi, \
            open("./cfg/tmp/ss_pre_generated.yml", "w") as fo:
                cfg = yaml.safe_load(fi)
                cfg.update(update_dict)
                yaml.dump(cfg, fo)


    prepareAgentFormerConfig(num_epochs=to_epoch, model_save_freq=5)
    python_call("model_train.py", f"--cfg ss_pre_generated --gpu 0 --start_epoch={from_epoch}")

def envPredStepTrain(from_epoch, to_epoch, af_epoch):
    def prepareEnvPredConfig(**update_dict):
        with open("./cfg/steersim/steersim_env.yml", "r") as fi, \
            open("./cfg/tmp/ss_env_generated.yml", "w") as fo:
                cfg = yaml.safe_load(fi)
                cfg.update(update_dict)
                yaml.dump(cfg, fo)

    prepareEnvPredConfig(pred_cfg="ss_pre_generated", pred_epoch=af_epoch, num_epochs=to_epoch, print_freq=1, model_save_freq=3)
    python_call("model_train.py", f"--cfg ss_env_generated --gpu 0 --start_epoch={from_epoch}")

def generateRandomConfig(ev_epoch):
    python_call("test_env.py", f"--cfg ss_env_generated --gpu 0 --epoch={ev_epoch} --random_latent")

def initial_sample_steersim():
    import os
    import shutil
    import numpy as np
    from dotenv import load_dotenv

    load_dotenv(verbose=True)
    shutil.rmtree(os.getenv("SteersimRecordPath"), ignore_errors=True)
    os.makedirs(os.getenv("SteersimRecordPath"), exist_ok=True)
    numbers = np.rint(np.random.uniform(0, 1, (50, 43)))
    steersim_call_parallel(numbers)
    numbers = np.rint(np.random.uniform(0, 1, (50, 43)))
    steersim_call_parallel(numbers, generate_for_testcases=True)

if True:
    initial_sample_steersim()

    bs = 0
    es = 0
    if bs != 0:  # warm up
        agentFormerStepTrain(from_epoch=0, to_epoch=bs)
        envPredStepTrain(from_epoch=0, to_epoch=es, af_epoch=bs)
        generateRandomConfig(ev_epoch=es)

    be=10
    ee=3
    for step in range(20):
        agentFormerStepTrain(from_epoch=be*step+bs, to_epoch=be*(step+1)+bs)
        envPredStepTrain(from_epoch=ee*step+es, to_epoch=ee*(step+1)+es, af_epoch=be*(step+1)+bs)
        generateRandomConfig(ev_epoch=ee*(step+1)+es)

#generateRandomConfig(ev_epoch=1)