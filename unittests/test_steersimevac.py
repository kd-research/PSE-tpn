import unittest
import os
from data.agent_grouping import AgentGrouping
from data.steersim import register_datafile, steersimProcess
from data.steersim_evac import SteersimEvacProcess
from utils.pyconfig import PyConfig


class SteersimEvacTest(unittest.TestCase):
    def test_agent_grouping_works(self):
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        config = PyConfig("steersim", 1, 1, 1, 1)
        register_datafile(f"{assets_dir}/sample.bin")
        sp = steersimProcess(assets_dir, 'sample', config, None)
        AgentGrouping(sp.gt).get_group_agent_indices(1, 5)

    def test_steersim_evac_works(self):
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        config = PyConfig("steersim-evac", 1, 1, 1, 1)
        register_datafile(f"{assets_dir}/sample.bin")
        SteersimEvacProcess(assets_dir, 'sample', config, None)



if __name__ == '__main__':
    unittest.main()
