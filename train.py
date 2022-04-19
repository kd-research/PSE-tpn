import yaml
import subprocess
import time
import sys
import os

os.environ["PYTHONPATH"] += ";" + os.path.realpath(__file__)
PYTHON_EXECUTABLE = sys.executable

def python_call(file, args):
    print(" ".join(["python", file, args]))
    with open("/tmp/agentformer.log", "a", buffering=1) as f:
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


    prepareAgentFormerConfig(num_epochs=to_epoch)
    python_call("model_train.py", f"--cfg ss_pre_generated --gpu 0 --start_epoch={from_epoch}")

def envPredStepTrain(from_epoch, to_epoch, af_epoch):
    def prepareEnvPredConfig(**update_dict):
        with open("./cfg/steersim/steersim_env.yml", "r") as fi, \
            open("./cfg/tmp/ss_env_generated.yml", "w") as fo:
                cfg = yaml.safe_load(fi)
                cfg.update(update_dict)
                yaml.dump(cfg, fo)

    prepareEnvPredConfig(pred_cfg="ss_pre_generated", pred_epoch=af_epoch, num_epochs=to_epoch, print_freq=1)
    python_call("model_train.py", f"--cfg ss_env_generated --gpu 0 --start_epoch={from_epoch}")

def generateRandomConfig(ev_epoch):
    python_call("test_env.py", f"--cfg ss_env_generated --gpu 0 --epoch={ev_epoch}")


for step in range(1, 20):
    agentFormerStepTrain(from_epoch=10*step, to_epoch=10*(step+1))
    envPredStepTrain(from_epoch=5*step, to_epoch=5*(step+1), af_epoch=10*(step+1))
    generateRandomConfig(ev_epoch=5*(step+1))