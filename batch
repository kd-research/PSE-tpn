#!/bin/bash

##SBATCH --chdir=
#SBATCH --job-name=batch-kaidong
#SBATCH --output=job.%j.out
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx2080|gtx1080ti"
 
export SINGULARITY_LOCALCACHEDIR=`pwd`/tmp
export SINGULARITY_TMPDIR=`pwd`/tmp
mkdir -p $SINGULARITY_TMPDIR
set -xe
module load singularity || true

remove_file='false'
if [[ $remove_file == 'true' ]]; then
rm -rf cfg/tmp/* || true
rm -rf results || true
rm -rf tmp || true
fi

jobtype='batch'
sif_path=$HOME/sifs/hlsix-220922.sif
py_path=/opt/conda/envs/torch/bin/python3

if [[ $jobtype == 'active' ]]; then
rm -rf steersimRecord
singularity exec --nv $sif_path $py_path train.py
fi

if [[ $jobtype == 'batch' ]]; then
singularity exec --nv $sif_path $py_path model_train.py --cfg steersim_pre --gpu 0
singularity exec --nv $sif_path $py_path model_train.py --cfg steersim_env --gpu 0
singularity exec --nv $sif_path $py_path test_env.py --cfg steersim_env --gpu 0 --data_eval train > train_result.txt
singularity exec --nv $sif_path $py_path test_env.py --cfg steersim_env --gpu 0 --data_eval test > test_result.txt
singularity exec --nv $sif_path $py_path test_env.py --cfg steersim_env --gpu 0 --data_eval val > val_result.txt
singularity exec --nv $sif_path $py_path test.py --cfg steersim_pre --gpu 0 || true
fi

echo 'Finished on ' `date`
