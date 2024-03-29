import os
import unittest
import numpy as np
from unittest.mock import patch
from numpy.testing import assert_allclose
from data.steersim_segmented import SteersimSegmentedProcess as steersimProcess
from utils.pyconfig import PyConfig


def agent_traj_processed(frame, agentId, xpos, zpos):
    FRAME_ID, AID, XID, ZID = 0, 1, 13, 15
    CLASSNAME_ID = 2
    new_row = np.array([-1.0] * 17, dtype="float32")
    new_row[FRAME_ID] = frame
    new_row[AID] = agentId
    new_row[XID] = xpos
    new_row[ZID] = zpos

    new_row[CLASSNAME_ID] = 1.0
    return new_row


class TestPreprocessor(unittest.TestCase):
    maxDiff = None

    def test_ss_requirement(self):
        config = PyConfig("steersim", 1, 1, 1, 1)
        self.assertRaises(AssertionError, steersimProcess, "", "", config, None)

    @patch.dict(os.environ, {"SteersimRecordPath": "/tmp"})
    def test_preprocessor_one_frame(self):
        config = PyConfig("steersim", 1, 1, 1, 1)

        # one agent with 5 frames
        def _np_gen(*args, **kwargs):
            data = []
            for i in range(5):
                data.append(agent_traj_processed(i, 0, 0, 0))
            data = np.stack(data, axis=0)
            return data, np.zeros((8,))

        seq = steersimProcess("", "", config, None, _fn_read_traj_binary=_np_gen)
        self.assertEqual(seq.TotalFrame(), 5)
        self.assertEqual(seq.get_max_total_frame(), 5)

        # There will be 2 samples in the sequence
        # 1. frame 0, 1
        # 2. frame 2, 3
        self.assertEqual(seq.get_total_available_sample_size(), 2)
        data = seq(0)
        pre_data = data["pre_data"]
        fut_data = data["fut_data"]
        assert_allclose(pre_data[0][0], agent_traj_processed(0, 0, 0, 0))
        assert_allclose(fut_data[0][0], agent_traj_processed(1, 0, 0, 0))

        data = seq(1)
        pre_data = data["pre_data"]
        fut_data = data["fut_data"]
        assert_allclose(pre_data[0][0], agent_traj_processed(2, 0, 0, 0))
        assert_allclose(fut_data[0][0], agent_traj_processed(3, 0, 0, 0))

    @patch.dict(os.environ, {"SteersimRecordPath": "/tmp"})
    def test_preprocessor_two_frames(self):
        config = PyConfig("steersim", 2, 2, 1, 1)

        # one agent with 5 frames
        def _np_gen(*args, **kwargs):
            data = []
            for i in range(5):
                data.append(agent_traj_processed(i, 0, 99, -99))
            data = np.stack(data, axis=0)
            return data, np.zeros((8,))

        seq = steersimProcess("", "", config, None, _fn_read_traj_binary=_np_gen)
        self.assertEqual(seq.TotalFrame(), 5)
        self.assertEqual(seq.get_max_total_frame(), 5)

        self.assertEqual(seq.get_total_available_sample_size(), 1)

        data = seq(0)
        pre_data = data["pre_data"]
        fut_data = data["fut_data"]
        assert_allclose(pre_data[0][0], agent_traj_processed(1, 0, 99, -99))
        assert_allclose(pre_data[1][0], agent_traj_processed(0, 0, 99, -99))
        assert_allclose(fut_data[0][0], agent_traj_processed(2, 0, 99, -99))
        assert_allclose(fut_data[1][0], agent_traj_processed(3, 0, 99, -99))

    @unittest.skip("Not implemented yet")
    #@patch.dict(os.environ, {"SteersimRecordPath": "/tmp"})
    def test_preprocessor_skip_frames(self):
        config = PyConfig("steersim", 2, 2, 1, 1)
        config.set("frame_skip", 2)

        def _np_gen(*args, **kwargs):
            data = []
            for i in range(8):
                data.append(agent_traj_processed(i, 0, 0, 0))
            data = np.stack(data, axis=0)
            return data, np.zeros((8,))

        seq = steersimProcess("", "", config, None, _fn_read_traj_binary=_np_gen)
        self.assertEqual(seq.TotalFrame(), 8)

        data = seq(2)
        pre_data = data["pre_data"]
        fut_data = data["fut_data"]
        assert_allclose(pre_data[0][0], agent_traj_processed(2, 0, 0, 0))
        assert_allclose(pre_data[1][0], agent_traj_processed(0, 0, 0, 0))
        assert_allclose(fut_data[0][0], agent_traj_processed(4, 0, 0, 0))
        assert_allclose(fut_data[1][0], agent_traj_processed(6, 0, 0, 0))

        data = seq(3)
        pre_data = data["pre_data"]
        fut_data = data["fut_data"]
        assert_allclose(pre_data[0][0], agent_traj_processed(3, 0, 0, 0))
        assert_allclose(pre_data[1][0], agent_traj_processed(1, 0, 0, 0))
        assert_allclose(fut_data[0][0], agent_traj_processed(5, 0, 0, 0))
        assert_allclose(fut_data[1][0], agent_traj_processed(7, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
