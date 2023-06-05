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
    try:
        model.inference(mode='recon', sample_num=sample_k, random_latent=random_latent)
    except:
        model.inference(mode='recon', sample_num=sample_k)


def test_model(generator, save_dir, cfg, random_latent):
    pickle_obj = []

    def test_once(data):
        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)
        print('testing seq: %s, frame: %06d                \r' % (seq_name, frame), end="", flush=True)

        with torch.no_grad():
            get_model_prediction(data, cfg.sample_k, random_latent)

        seq_name = data['seq']
        context_v = model.data['context_enc'].detach().cpu().numpy()
        z = model.data['q_z_dist'].mode().detach().cpu().numpy()

        assert (isinstance(seq_name, str))
        assert (isinstance(context_v, np.ndarray))
        assert (isinstance(z, np.ndarray))

        env_param = None
        try:
            env_param = data['env_parameter']
            assert (isinstance(env_param, np.ndarray))
        except KeyError:
            pass

        pickle_obj.append({
            "seq_name": seq_name,
            "env_param": env_param.tolist(),
            "context_v": context_v.tolist(),
            "z": z.tolist(),
            "frame": frame,
        })


    while not generator.is_epoch_end():
        data = generator()
        if data is None:
            continue
        test_once(data)

    import json
    with open(f"{cfg.latent_dir}/{args.data_eval}.json", "w") as f:
        print(f"Serialized {len(pickle_obj)} data")
        for d in pickle_obj:
            f.write(json.dumps(d))
            f.write("\n")


import dotenv

if __name__ == '__main__':
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='steersim_pre')
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
    device = (torch.device('cuda', index=args.gpu)
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
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
