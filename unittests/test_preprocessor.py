import unittest
import numpy as np
from numpy.testing import assert_allclose
from data.preprocessor import preprocess

class MockConfig(object):
    def __init__(self, dataset, past_frames, future_frames, min_past_frames, min_future_frames):
        self.dataset = dataset
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.min_past_frames = min_past_frames
        self.min_future_frames = min_future_frames

        self.traj_scale = 2

    def set(self, key, value):
        setattr(self, key, value)

    def get(self, key, default):
        return getattr(self, key, default)

def agent_traj_raw(frame, agentId, xpos, zpos):
    FRAME_ID, AID, XID, ZID = 0, 1, 13, 15
    CLASSNAME_ID = 2
    new_row = np.array(["-1.0"]*17, dtype="<U32")
    new_row[FRAME_ID] = str(frame)
    new_row[AID] = str(agentId)
    new_row[XID] = str(xpos)
    new_row[ZID] = str(zpos)

    new_row[CLASSNAME_ID] = "Pedestrian"
    return new_row

def agent_traj_processed(frame, agentId, xpos, zpos):
    FRAME_ID, AID, XID, ZID = 0, 1, 13, 15
    CLASSNAME_ID = 2
    new_row = np.array([-1.0]*17, dtype="float32")
    new_row[FRAME_ID] = frame
    new_row[AID] = agentId
    new_row[XID] = xpos
    new_row[ZID] = zpos

    new_row[CLASSNAME_ID] = 1.0
    return new_row

class TestPreprocessor(unittest.TestCase):
    maxDiff = None

    def test_preprocessor_one_frame(self):
        config = MockConfig("eth", 1, 1, 1, 1)
        def _np_gen(*args, **kwargs):
            data = []
            for i in range(5):
                data.append(agent_traj_raw(i, 0, 0, 0))
            data = np.stack(data, axis=0)
            return data

        seq = preprocess("", "", config, None, _fn_np_genfromtxt=_np_gen)
        self.assertEqual(seq.TotalFrame(), 5)

        data = seq(0)
        pre_data = data["pre_data"]
        fut_data = data["fut_data"]
        assert_allclose(pre_data[0][0], agent_traj_processed(0, 0, 0, 0))
        assert_allclose(fut_data[0][0], agent_traj_processed(1, 0, 0, 0))

        data = seq(1)
        pre_data = data["pre_data"]
        fut_data = data["fut_data"]
        assert_allclose(pre_data[0][0], agent_traj_processed(1, 0, 0, 0))
        assert_allclose(fut_data[0][0], agent_traj_processed(2, 0, 0, 0))

    def test_preprocessor_two_frames(self):
        config = MockConfig("eth", 2, 2, 1, 1)
        def _np_gen(*args, **kwargs):
            data = []
            for i in range(5):
                data.append(agent_traj_raw(i, 0, 0, 0))
            data = np.stack(data, axis=0)
            return data

        seq = preprocess("", "", config, None, _fn_np_genfromtxt=_np_gen)
        self.assertEqual(seq.TotalFrame(), 5)

        data = seq(1)
        pre_data = data["pre_data"]
        fut_data = data["fut_data"]
        assert_allclose(pre_data[0][0], agent_traj_processed(1, 0, 0, 0))
        assert_allclose(pre_data[1][0], agent_traj_processed(0, 0, 0, 0))
        assert_allclose(fut_data[0][0], agent_traj_processed(2, 0, 0, 0))
        assert_allclose(fut_data[1][0], agent_traj_processed(3, 0, 0, 0))

        data = seq(2)
        pre_data = data["pre_data"]
        fut_data = data["fut_data"]
        assert_allclose(pre_data[0][0], agent_traj_processed(2, 0, 0, 0))
        assert_allclose(pre_data[1][0], agent_traj_processed(1, 0, 0, 0))
        assert_allclose(fut_data[0][0], agent_traj_processed(3, 0, 0, 0))
        assert_allclose(fut_data[1][0], agent_traj_processed(4, 0, 0, 0))

    def test_preprocessor_skip_frames(self):
        config = MockConfig("eth", 2, 2, 1, 1)
        config.set("frame_skip", 2)
        def _np_gen(*args, **kwargs):
            data = []
            for i in range(8):
                data.append(agent_traj_raw(i, 0, 0, 0))
            data = np.stack(data, axis=0)
            return data

        seq = preprocess("", "", config, None, _fn_np_genfromtxt=_np_gen)
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
