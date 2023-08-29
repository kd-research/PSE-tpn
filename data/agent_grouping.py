import logging
import numpy
import pandas
import random

logger = logging.Logger(__name__)


# wrap agent grouping into function so it can be used in parallel
def gt_to_group_gt_params(gt, seq_name, param, num_agent, psizes):
    agent_grouping_obj = AgentGrouping(gt, seq_name=seq_name, param=param, num_agent=num_agent, psizes=psizes)
    return agent_grouping_obj.group_gt_params()


class AgentGrouping:
    """
    Given a set of trajectory, find overall n-nearest agents for given agent
    Usage:
      # get 6 nearest agents for agent id 51
      AgentGrouping(data.gt).get_group_agent_indices(51, 6)
    """

    def __init__(self, gt, seq_name=None, param=None, num_agent=6, psizes=[1, 1]):
        self.gt = gt
        self.agent_num = num_agent
        self.param = param
        self.seq_name = seq_name
        self._prepare(gt)
        self.env_parameter_size = psizes[0]
        self.agent_parameter_size = psizes[1]
        assert self.agent_parameter_size * self.get_num_agents() + self.env_parameter_size == len(param)

    @staticmethod
    def _compute_ADE(d, a1, a2):
        d1 = d[d.aid == a1]
        d2 = d[d.aid == a2]
        # get frames for both agents appears, align trajectory
        # and put trajectory data in x1, z1 / x2, z2
        dm = d1.merge(d2, on="nframe", suffixes=["1", "2"])
        if dm.empty:
            return numpy.Infinity
        t1 = dm[['x1', 'z1']].values  # extract values
        t2 = dm[['x2', 'z2']].values
        return numpy.mean(numpy.linalg.norm(t1 - t2, axis=1))

    def _prepare(self, gt):
        # initialize dataframe used for pandas join
        colidx = [0, 1, 13, 15]
        cols = 'nframe aid x z'.split()
        self.d = pandas.DataFrame(gt[:, colidx], columns=cols)
        self.d['nframe'] = self.d.nframe.astype(int)
        self.d['aid'] = self.d.aid.astype(int)

        # discard initialize area and select main area only
        self.d = self.d[self.d.z > 1]

        # sub sample data to reduce computation
        self.d = self.d[self.d.nframe % 10 == 0]

        # cache ADE matrix so it can be reused
        nagents = self.d.aid.max() + 1
        self.ADEm = ADEm = numpy.zeros([nagents, nagents])
        # ADE matrix should be semmetric and main diagnonal should be 0s
        meshX, meshY = numpy.tril_indices_from(ADEm, k=-1)
        for x, y in zip(meshX, meshY):
            ADEm[y, x] = ADEm[x, y] = self._compute_ADE(self.d, x, y)

    def get_group_agent_indices(self, target_idx, num_agent):
        is_inf_ade = numpy.isinf(self.ADEm[target_idx])
        candidate_index = numpy.argsort(self.ADEm[target_idx])[:num_agent]
        return candidate_index[~is_inf_ade[candidate_index]]

    def group_gt_params(self):
        agent_params = None
        if self.agent_parameter_size != 0:
            agent_params_begin = self.env_parameter_size
            agent_params = numpy.array(self.param[agent_params_begin:]).reshape(-1, self.agent_parameter_size)

        agent_pool = set(range(self.get_num_agents()))

        all_group_gt = []
        all_group_params = []
        all_group_indices = []

        while len(agent_pool) > 0:
            aid = random.choice(list(agent_pool))
            group_agent_indices = sorted(self.get_group_agent_indices(aid, self.agent_num))
            aid_tried = set()
            while len(group_agent_indices) < self.agent_num:
                logging.info('Cannot find enough agents to form a group, try incrementally')
                group_candidate = list(agent_pool & set(group_agent_indices) - aid_tried)
                if len(group_candidate) == 0:
                    group_candidate = set(group_agent_indices) - aid_tried
                if len(group_candidate) == 0:
                    logging.error('seq: %s, aid: %d' % (self.seq_name, aid))
                    logging.error('agent_pool: %s' % agent_pool)
                    logging.error('formed group: %s' % group_agent_indices)
                    raise ValueError('Cannot find enough agents to form a group')
                aid = random.choice(list(group_candidate))
                aid_tried.add(aid)
                group_agent_indices_more = self.get_group_agent_indices(aid, self.agent_num)
                group_agent_indices_more = [i for i in group_agent_indices_more if i not in group_agent_indices]
                group_agent_indices += group_agent_indices_more[:self.agent_num - len(group_agent_indices)]

            agent_pool -= set(group_agent_indices)

            if self.agent_parameter_size == 0:
                group_params = self.param
            else:
                group_params = numpy.concatenate(
                    (self.param[:agent_params_begin], agent_params[group_agent_indices].flatten()))
            group_agent_gt = self.gt[numpy.isin(self.gt[:, 1], group_agent_indices)]

            all_group_gt.append(group_agent_gt)
            all_group_params.append(group_params)
            all_group_indices.append(group_agent_indices)

        return all_group_gt, all_group_params, all_group_indices

    def get_num_agents(self):
        return self.d.aid.max() + 1
