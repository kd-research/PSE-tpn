import logging
import os.path
import random
from pathlib import Path
import glob
import numpy as np

from .preprocessor import preprocess

logger = logging.Logger(__name__)

STEERSIM_BINARIES = {}

def get_base_path(full):
    base = Path(full).stem
    STEERSIM_BINARIES[base] = full
    return base

def get_full_path(base):
    return STEERSIM_BINARIES.get(base)

def steersim_config_spec_split(config):
    train_list = [get_base_path(file) for file in config['train_source']]
    valid_list = [get_base_path(file) for file in config['valid_source']]
    test_list = [get_base_path(file) for file in config['test_source']]
    return train_list, valid_list, test_list

def get_steersim_split(parser):
    split = [f"steersim{i:02}" for i in range(1, 50)], \
        [f"steersim{i:02}" for i in range(201, 211)], \
        [f"steersim{i:02}" for i in range(221, 231)]

    split = [], [], []

    # Priority 1: get steersim binaries from config file
    config_embed_source = parser.get('steersim_data_source')
    if config_embed_source:
        return steersim_config_spec_split(config_embed_source)

    # Priority 2: collect binaries from give record path
    ssRecordPath = parser.get("ss_record_path", os.getenv("SteersimRecordPath"))
    print(f"Steersim.py: getting data from {ssRecordPath}")
    assert ssRecordPath

    trainGlob = glob.glob(ssRecordPath + "/*.bin")
    train_seq_name = [os.path.basename(x)[:-4] for x in trainGlob]
    cvGlob = glob.glob(ssRecordPath + "/test/*.bin")
    cv_seq_name = [os.path.basename(x)[:-4] for x in cvGlob]
    tstGlob = glob.glob(ssRecordPath + "/test1/*.bin")
    tst_seq_name = [os.path.basename(x)[:-4] for x in tstGlob]
    split[0].extend(train_seq_name)
    split[1].extend(cv_seq_name)
    split[2].extend(tst_seq_name)

    return split


class steersimProcess(preprocess):
    def __init__(self,
                 data_root,
                 seq_name,
                 parser,
                 log,  # Unused
                 split='train',
                 phase='training',
                 *,
                 _fn_read_traj_binary=None):
        self.parser = parser
        self.dataset = parser.dataset
        self.data_root = data_root
        self.past_frames = parser.past_frames
        self.future_frames = parser.future_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.play_speed = parser.get('play_speed', 1)
        self.min_past_frames = parser.get('min_past_frames', self.past_frames)
        self.min_future_frames = parser.get('min_future_frames', self.future_frames)
        self.traj_scale = parser.traj_scale
        self.past_traj_scale = parser.traj_scale
        self.load_map = parser.get('load_map', False)
        self.map_version = parser.get('map_version', '0.1')
        self.seq_name = seq_name
        self.split = split
        self.phase = phase
        self.log = log

        if parser.dataset == 'steersim':
            # Priority 1: find binary from config
            label_path = get_full_path(seq_name)

            # Priority 2: find binary in all possible directory
            if not os.path.exists(label_path):
                self.ssRecordPath = parser.get("ss_record_path", os.getenv("SteersimRecordPath"))
                assert self.ssRecordPath
                label_path = f'{data_root}/{seq_name}.bin'
            if not os.path.exists(label_path):
                label_path = f'{self.ssRecordPath}/{seq_name}.bin'
            if not os.path.exists(label_path):
                label_path = f'{self.ssRecordPath}/test/{seq_name}.bin'
            if not os.path.exists(label_path):
                label_path = f'{self.ssRecordPath}/test1/{seq_name}.bin'
            assert os.path.exists(label_path) or _fn_read_traj_binary is not None
        else:
            assert False, 'error'
        if _fn_read_traj_binary is None:
            _fn_read_traj_binary = self.read_trajectory_binary
        self.gt, self.parm = _fn_read_traj_binary(label_path, self.play_speed)
        frames = self.gt[:, 0].astype(np.float32).astype(np.int32)
        fr_start, fr_end = frames.min(), frames.max()
        self.init_frame = fr_start
        self.num_fr = fr_end + 1 - fr_start

        if self.load_map:
            self.load_scene_map()
        else:
            self.geom_scene_map = None

        self.xind, self.zind = 13, 15

    @staticmethod
    def read_trajectory_binary(filename, playspeed=1):
        """
        :param filename:
        :return:
        """
        with open(filename, "rb") as file:
            logger.debug("Beginning read file: %s", filename)
            eof = file.seek(0, 2)
            file.seek(0, 0)

            obsTypeSize = np.fromfile(file, np.int32, 1)[0]
            logger.debug("Reading obstacle section, size %d", obsTypeSize)
            _ = np.fromfile(file, dtype=np.int32, count=obsTypeSize)

            obsInfoSize = np.fromfile(file, np.int32, 1)[0]
            logger.debug("Reading obstacle info section, size %d", obsInfoSize)
            _ = np.fromfile(file, dtype=np.float32, count=obsInfoSize)

            parameter_size = np.fromfile(file, np.int32, 1)[0]
            logger.debug("Reading parameter info section, size %d", parameter_size)
            parameters = np.fromfile(file, dtype=np.float32, count=parameter_size)
            parameters = parameters * 2 - 1

            agent_array = []
            while file.tell() < eof:
                trajectoryLength = np.fromfile(file, np.int32, 1)[0]
                logger.debug("Reading agent info for agentId %d, size %d", len(agent_array), trajectoryLength)
                trajectory_matrix = np.fromfile(file, np.float32, trajectoryLength).reshape((-1, 2))
                agent_array.append(trajectory_matrix)

            gt_matrix = []
            for agentId, agent_matrix in enumerate(agent_array):
                agent_matrix = agent_matrix[:playspeed * 50:playspeed, :]
                frame_length = agent_matrix.shape[0]
                gt_extend = np.full([frame_length, 17], -1, dtype=np.float32)
                gt_extend[:, 0] = np.arange(frame_length)  # frame_num
                gt_extend[:, 1] = agentId  # agent_id
                gt_extend[:, 2] = 1  # pedestrian
                gt_extend[:, [13, 15]] = agent_matrix  # position x, z
                gt_matrix.append(gt_extend)

        all_matrix = np.concatenate(gt_matrix)
        return all_matrix[all_matrix[:, 0].argsort(kind="stable")], parameters

    env1_rect = {"xmin": -70, "xmax": 70, "ymin": -100, "ymax": 100}

    def get_min_total_frame(self):
        agent_ids = set(self.gt[:, 1])
        frame_by_id = []
        for aid in agent_ids:
            agent_frame = self.gt[self.gt[:, 1] == aid]
            frame_by_id.append(np.max(agent_frame) + 1)

        return int(min(frame_by_id))

    def __call__(self, frame, *args, **kwargs):
        # Steersim.call(8) have past_frame [1..8] and future frame (9...)
        min_total_frame = self.get_min_total_frame()
        min_split = max(self.min_past_frames - 1,
                        min_total_frame - self.future_frames - 1)  # Must include entire future frames
        max_split = min(self.past_frames - 1,
                        min_total_frame - self.min_future_frames - 1)  # Must include entire past frames
        assert min_split <= max_split
        if min_split > max_split:  # may happens if min_past + min_future > min_total
            print(f"warning: Dataset not fulfill require, mf={min_total_frame} s={self.TotalFrame()} ms={[min_split, max_split]}")
            max_split = min_split
        candidate_split = range(min_split, max_split + 1)

        split_1 = random.choice(candidate_split)
        split_2 = random.choice(candidate_split)
        frame = int((split_1 + split_2) / 2)

        assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.TotalFrame() - 1, 'frame is %d, total is %d' % (
            frame, self.TotalFrame())

        pre_data = self.PreData(frame)
        fut_data = self.FutureData(frame)
        valid_id = self.get_valid_id(pre_data, fut_data)
        assert len(valid_id) == 2, f"Some trajectory was rejected by valid_id{valid_id}"

        pred_mask = None
        heading = None

        pre_motion_3D, pre_motion_mask = self.PreMotion(pre_data, valid_id)
        fut_motion_3D, fut_motion_mask = self.FutureMotion(fut_data, valid_id)

        data = {'pre_motion_3D': pre_motion_3D, 'fut_motion_3D': fut_motion_3D, 'fut_motion_mask': fut_motion_mask,
                'pre_motion_mask': pre_motion_mask, 'pre_data': pre_data, 'fut_data': fut_data, 'heading': heading,
                'valid_id': valid_id, 'traj_scale': self.traj_scale, 'pred_mask': pred_mask,
                'scene_map': self.geom_scene_map, 'seq': self.seq_name, 'frame': frame, "env_parameter": self.parm}

        return data
