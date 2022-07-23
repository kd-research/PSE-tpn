import argparse
import os
import shutil
import subprocess
import multiprocessing
import sys
import numpy as np
import multiprocessing

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing


def get_model_prediction(data, sample_k, random_latent):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k, random_latent=random_latent)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False, random_latent=random_latent)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D



def steersim_call(query, env):
    steersim_command_path = env["SteersimCommandPath"]
    steersim_command_exec = env["SteersimCommandExec"]
    p = subprocess.Popen(steersim_command_exec.split(), cwd=steersim_command_path,
                         stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    stdout_line, stderr_line = p.communicate(input=query.encode())
    print("send request to steersim...")
    p.wait()
    if p.returncode != 0:
        print('From subprocess out: %r' % stdout_line)
        print('From subprocess err: %r' % stderr_line)


def steersim_call_parallel(query, generate_for_testcases=False):
    """
    Steersim arguments must be a list of numbers [len_query, len_parameters]
    """
    SteersimRecordPath = "SteersimRecordPath"
    env = os.environ.copy()
    if generate_for_testcases:
        env[SteersimRecordPath] = env[SteersimRecordPath] + "/test"

    os.makedirs(env[SteersimRecordPath], exist_ok=True)
    queries = np.clip(query, 0, 1)
    query_string = " ".join([str(x) for x in queries.ravel()])
    PROCESS_POOL.apply_async(steersim_call, args=(query_string, env))


def save_prediction(pred, data, suffix, save_dir):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], \
                                                     data['pred_mask']

    # save results
    fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
    mkdir_if_missing(fname)
    np.savetxt(fname, pred.cpu().numpy(), fmt="%.3f")
    return 1


def test_model(generator, save_dir, cfg, random_latent):
    total_num_pred = 0
    losses = []

    def RMSELoss(yhat, y):
        yhat = yhat.cpu().numpy()
        y = y.cpu().numpy()
        #return np.sum((yhat-y)**2)
        return np.sum(np.abs(yhat-y) < 1) / y.ravel().size

    def test_once(data):
        nonlocal total_num_pred
        nonlocal losses
        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)
        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))
        sys.stdout.flush()

        gt_motion_3D = torch.tensor(data['env_parameter'])
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k, random_latent)

        """save samples"""
        recon_dir = os.path.join(save_dir, 'recon');
        mkdir_if_missing(recon_dir)
        sample_dir = os.path.join(save_dir, 'samples');
        mkdir_if_missing(sample_dir)
        gt_dir = os.path.join(save_dir, 'gt');
        mkdir_if_missing(gt_dir)
        for i in range(sample_motion_3D.shape[0]):
            save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir)
        save_prediction(recon_motion_3D, data, '', recon_dir)  # save recon
        num_pred = save_prediction(gt_motion_3D, data, '', gt_dir)  # save gt
        losses.append(float(RMSELoss(gt_motion_3D, recon_motion_3D)))
        if random_latent:
            params_cont = recon_motion_3D.detach().cpu().numpy()
            print("params_cont", params_cont)
            params_desc = np.rint(params_cont)
            print("params_desc", params_cont)
            steersim_call_parallel(params_desc)
        total_num_pred += num_pred

    while not generator.is_epoch_end():
        data = generator()
        if data is None:
            continue
        if random_latent:
            for i in range(50):
                test_once(data)
            break
        else:
            test_once(data)

    PROCESS_POOL.close()
    PROCESS_POOL.join()
    print()
    print("Average loss: ", sum(losses) / len(losses))
    print_log(f'\n\n total_num_pred: {total_num_pred}', log)


import dotenv

if __name__ == '__main__':
    PROCESS_POOL = multiprocessing.Pool() #use all available cores, otherwise specify the number you want as an argument
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--random_latent', action='store_true', default=False)

    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device(
        'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    for epoch in epochs:
        prepare_seed(cfg.seed)
        """ model """
        if not args.cached:
            model_id = cfg.get('model_id', 'agentformer')
            model = model_dict[model_id](cfg)
            model.set_device(device)
            model.eval()
            if epoch > 0:
                cp_path = cfg.model_path % epoch
                print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
                model_cp = torch.load(cp_path, map_location='cpu')
                model.load_state_dict(model_cp['model_dict'], strict=False)

        """ save results and compute metrics """
        data_splits = [args.data_eval]

        for split in data_splits:
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}';
            mkdir_if_missing(save_dir)
            eval_dir = f'{save_dir}/samples'
            if not args.cached:
                test_model(generator, save_dir, cfg, args.random_latent)

            log_file = os.path.join(cfg.log_dir, 'log_eval.txt')

            # remove eval folder to save disk space
            if args.cleanup:
                shutil.rmtree(save_dir)
