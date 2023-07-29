import math
import pandas
import logging
import numpy as np
import copy
import bisect
import random
import subprocess

from .steersim import steersimProcess
from .steersim_segmented import SteersimSegmentedProcess
from .agent_grouping import AgentGrouping, gt_to_group_gt_params

logger = logging.Logger(__name__)


class SteersimEvacProcess(steersimProcess):
    def __init__(self, *args, **kwargs):
        eager_init = kwargs.pop("eager_init", True)
        # self.min_past_frames is meaningless in this context
        # self.min_future_frame is used to validate agents in each sample
        super().__init__(*args, **kwargs)
        self._async_result = None
        if np.isnan(self.gt).any():
            raise ValueError('gt contains NaN')

        self.segmented = True
        # remove frames when agent not in the scene
        d = pandas.DataFrame(self.gt[:, [0, 1, 15]], columns='nframe aid z'.split())
        d['nframe'] = d.nframe.astype(int)
        d['aid'] = d.aid.astype(int)
        d['origin_index'] = d.index

        clamp_max = d[d.z < 1].groupby('aid').nframe.max()
        merged = pandas.merge(d, clamp_max, on="aid", suffixes=("", "_clamp"))
        filtered = merged[merged.nframe > merged.nframe_clamp]

        self.gt = np.take(self.gt, filtered.origin_index, axis=0)
        self.args = args
        self.kwargs = kwargs
        if eager_init:
            self.initialize_single_thread()


    def initialize_single_thread(self):
        agent_grouping = AgentGrouping(self.gt, seq_name=self.seq_name, param=self.parm, num_agent=self.agent_num)

        self.accu_sample_size = []
        self.sub_dataset = []
        total_sample_size = 0

        for group_agent_gt, group_params in zip(*agent_grouping.group_gt_params()):
            kwargs_copy = copy.copy(self.kwargs)
            kwargs_copy['_fn_read_traj_binary'] = lambda *_: [group_agent_gt, group_params]
            group_dataset = SteersimSegmentedProcess(
                *self.args,
                **kwargs_copy
            )
            self.accu_sample_size.append(total_sample_size)
            self.sub_dataset.append(group_dataset)
            total_sample_size += group_dataset.get_total_available_sample_size()
            #self.visulaize_group(aid, group_agent_indices)

        self.accu_sample_size.append(total_sample_size)
        self.parameter_size = self.sub_dataset[0].parameter_size

    def initialize_multi_thread(self, pool_worker):
        self._async_result = pool_worker.apply_async(gt_to_group_gt_params,
                                                     (self.gt, self.seq_name, self.parm, self.agent_num))

    def wait_for_initialization(self):
        if self._async_result is None:
            raise ValueError("promise is not initialized")
        group_agent_gt, group_params = self._async_result.get()
        total_sample_size = 0
        self.accu_sample_size = []
        self.sub_dataset = []

        for group_agent_gt, group_params in zip(group_agent_gt, group_params):
            kwargs_copy = copy.copy(self.kwargs)
            kwargs_copy['_fn_read_traj_binary'] = lambda *_: [group_agent_gt, group_params]
            group_dataset = SteersimSegmentedProcess(
                *self.args,
                **kwargs_copy
            )
            self.accu_sample_size.append(total_sample_size)
            self.sub_dataset.append(group_dataset)
            total_sample_size += group_dataset.get_total_available_sample_size()

        self.accu_sample_size.append(total_sample_size)
        self.parameter_size = self.sub_dataset[0].parameter_size

    def visulaize_group(self, aid, group_id):
        # visualizing using tool /home/kaidong/RubymineProjects/opencvpp/evac.rb
        # this tool receives four lines of data
        # 1. the first line is the path of trajectory file
        # 2. output path
        # 3. aid
        # 4. group id
        try:
            process = subprocess.Popen(['bash', '-c', 'cd /home/kaidong/RubymineProjects/opencvpp/ && ruby evac.rb'],
                                       stdin=subprocess.PIPE, text=True)
            process.stdin.write(f"{self.label_path}\n")
            process.stdin.write(f"{self.seq_name}_{aid}\n")
            process.stdin.write(f"{aid}\n")
            process.stdin.write(f"{group_id}\n")
            process.stdin.close()
        except Exception as e:
            logger.error(e)
            raise

    def get_total_available_sample_size(self):
        return self.accu_sample_size[-1]

    def get_min_total_frame(self):
        # length of the fastest agent achieves the goal
        raise NotImplementedError

    def get_max_total_frame(self):
        # length of the slowest agent achieves the goal
        raise NotImplementedError

    def __call__(self, sample_index, *args, **kwargs):
        assert self.frame_skip == 1, "frame skip is not considered yet"

        sub_data_index = bisect.bisect_right(self.accu_sample_size, sample_index) - 1
        sub_sample_index = sample_index - self.accu_sample_size[sub_data_index]

        data = self.sub_dataset[sub_data_index](sub_sample_index, *args, **kwargs)
        data['seq'] = f"{self.seq_name}_{sub_data_index}"

        return data
