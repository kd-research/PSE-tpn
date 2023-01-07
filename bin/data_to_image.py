import os
import sys
from matplotlib import pyplot as plt
from data.dataloader import data_generator
from utils.config import Config

os.chdir(os.path.abspath(__file__ + "/../.."))
config = Config("steersim_pre", tmp=False, create_dirs=False)
config.ss_record_path = sys.argv[1]
nolog = open("/dev/null", "w")
data = data_generator(config, nolog)

chart = open("seqlist.txt", 'w')

while not data.is_epoch_end():
    d = data()
    env_list = d['env_parameter']
    env_list = map(str, env_list)
    estr = ",".join(env_list)
    chart.write(f"{d['seq']}, {estr}\n")

chart.close()
