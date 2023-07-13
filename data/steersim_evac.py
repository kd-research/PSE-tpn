import math
import pandas
import logging
import numpy as np
import copy
import bisect
import random

from .steersim import steersimProcess
from .steersim_segmented import SteersimSegmentedProcess
from .agent_grouping import AgentGrouping

logger = logging.Logger(__name__)

class SteersimEvacProcess(steersimProcess):
    def __init__(self, *args, **kwargs):
        # self.min_past_frames is meaningless in this context
        # self.min_future_frame is used to validate agents in each sample
        super().__init__(*args, **kwargs)
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

        agent_params_begin = 1
        agent_params = np.array(self.parm[agent_params_begin:]).reshape(-1, 5)
        self.agent_grouping = AgentGrouping(self.gt)

        self.accu_sample_size = []
        self.sub_dataset = []
        total_sample_size = 0

        agent_pool = set(range(self.agent_grouping.get_num_agents()))
        while len(agent_pool) > 0:
            aid = random.choice(list(agent_pool))
            kwargs_copy = copy.copy(kwargs)
            group_agent_indices = sorted(self.agent_grouping.get_group_agent_indices(aid, self.agent_num))
            agent_pool -= set(group_agent_indices)

            group_params = np.concatenate(
                (self.parm[:agent_params_begin], agent_params[group_agent_indices].flatten()))
            group_agent_gt = self.gt[np.isin(self.gt[:, 1], group_agent_indices)]

            kwargs_copy['_fn_read_traj_binary'] = lambda *_: [group_agent_gt, group_params]
            group_dataset = SteersimSegmentedProcess(
                *args,
                **kwargs_copy
            )
            self.accu_sample_size.append(total_sample_size)
            self.sub_dataset.append(group_dataset)
            total_sample_size += group_dataset.get_total_available_sample_size()

        self.accu_sample_size.append(total_sample_size)
        self.parameter_size = self.sub_dataset[0].parameter_size

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
