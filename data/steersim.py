import logging
import os.path
from pathlib import Path
import glob
import numpy as np

from .preprocessor import preprocess

logger = logging.Logger(__name__)


def get_steersim_split(parser):
    split = [f"steersim{i:02}" for i in range(1, 50)], \
           [f"steersim{i:02}" for i in range(201, 211)], \
           [f"steersim{i:02}" for i in range(221, 231)]

    split = [], [], []

    ssRecordPath = parser.get("ss_record_path", os.getenv("SteersimRecordPath"))
    assert ssRecordPath

    trainGlob = glob.glob(ssRecordPath+"/*.bin")
    train_seq_name = [os.path.basename(x)[:-4] for x in trainGlob]
    cvGlob = glob.glob(ssRecordPath+"/test/*.bin")
    cv_seq_name = [os.path.basename(x)[:-4] for x in cvGlob]
    tstGlob = glob.glob(ssRecordPath+"/test1/*.bin")
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
                 log, # Unused
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
        self.ssRecordPath = parser.get("ss_record_path", os.getenv("SteersimRecordPath"))
        assert self.ssRecordPath

        if parser.dataset == 'steersim':
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
                agent_matrix = agent_matrix[:playspeed*50:playspeed, :]
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

    def __call__(self, *args, **kwargs):
        data = super(steersimProcess, self).__call__(*args, **kwargs)
        data["env_parameter"] = self.parm
        return data

