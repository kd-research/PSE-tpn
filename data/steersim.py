import logging
import numpy as np

from .preprocessor import preprocess

logger = logging.Logger(__name__)


def get_steersim_split(_):
    return [f"steersim{i:02}" for i in range( 1, 31)], \
           [f"steersim{i:02}" for i in range(31, 41)], \
           [f"steersim{i:02}" for i in range(50, 51)], \


class steersimProcess(preprocess):
    def __init__(self,
                 data_root,
                 seq_name,
                 parser,
                 log,
                 split='train',
                 phase='training'):
        self.parser = parser
        self.dataset = parser.dataset
        self.data_root = data_root
        self.past_frames = parser.past_frames
        self.future_frames = parser.future_frames
        self.frame_skip = parser.get('frame_skip', 1)
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
            label_path = f'{data_root}/{seq_name}.bin'
        else:
            assert False, 'error'

        self.gt = self.read_trajectory_binary(label_path)
        frames = self.gt[:, 0].astype(np.float32).astype(np.int)
        fr_start, fr_end = frames.min(), frames.max()
        self.init_frame = fr_start
        self.num_fr = fr_end + 1 - fr_start

        if self.load_map:
            self.load_scene_map()
        else:
            self.geom_scene_map = None

        self.xind, self.zind = 13, 15

    @staticmethod
    def read_trajectory_binary(filename):
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
            _ = np.fromfile(file, dtype=np.float32, count=parameter_size)

            agent_array = []
            while file.tell() < eof:
                trajectoryLength = np.fromfile(file, np.int32, 1)[0]
                logger.debug("Reading agent info for agentId %d, size %d", len(agent_array), trajectoryLength)
                trajectory_matrix = np.fromfile(file, np.float32, trajectoryLength).reshape((-1, 2))
                agent_array.append(trajectory_matrix)

            gt_matrix = []
            for agentId, agent_matrix in enumerate(agent_array):
                agent_matrix = agent_matrix[::30, :]        # sample in 1 fps
                frame_length = agent_matrix.shape[0]
                gt_extend = np.full([frame_length, 17], -1, dtype=np.float32)
                gt_extend[:, 0] = np.arange(frame_length)   # frame_num
                gt_extend[:, 1] = agentId                   # agent_id
                gt_extend[:, 2] = 1                         # pedestrian
                gt_extend[:, [13, 15]] = agent_matrix       # position x, z
                gt_matrix.append(gt_extend)

        all_matrix = np.concatenate(gt_matrix)
        return all_matrix[all_matrix[:, 0].argsort(kind="stable")]

    env1_rect = {"xmin": -70, "xmax": 70, "ymin": -100, "ymax": 100}