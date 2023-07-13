import logging
import numpy
import pandas

logger = logging.Logger(__name__)


class AgentGrouping:
    """
    Given a set of trajectory, find overall n-nearest agents for given agent
    Usage:
      # get 6 nearest agents for agent id 51
      AgentGrouping(data.gt).get_group_agent_indices(51, 6)
    """

    def __init__(self, gt):
        self._prepare(gt)

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
        return numpy.argpartition(self.ADEm[target_idx], num_agent)[:num_agent]

    def get_num_agents(self):
        return self.d.aid.max() + 1