class PyConfig(object):
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
