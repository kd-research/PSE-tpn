import copy
import multiprocessing
import random

from utils.utils import print_log
from .ethucy_split import get_ethucy_split
from .nuscenes_pred_split import get_nuscenes_pred_split
from .preprocessor import preprocess
from .steersim import get_steersim_split, steersimProcess
from .steersim_segmented import SteersimSegmentedProcess
from .steersim_evac import SteersimEvacProcess


class data_generator(object):
    _impl_get_data_splits = {
        "get_ethucy_split": get_ethucy_split,
        "get_nuscenes_pred_split": get_nuscenes_pred_split,
        "get_steersim_split": get_steersim_split
    }

    _impl_preprocess = {
        "preprocess": preprocess,
        "steersimProcess": steersimProcess,
        "steersimSegmented": SteersimSegmentedProcess,
        "steersimEvac": SteersimEvacProcess
    }

    def __init__(self, parser, log, split='train', phase='training', *,
                 _fn_get_data_splits=None, _cls_preprocess=None):

        _fn_get_data_splits = _fn_get_data_splits or data_generator._impl_get_data_splits
        _cls_preprocess = _cls_preprocess or data_generator._impl_preprocess

        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        self.dataset = parser.dataset
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred
            seq_train, seq_val, seq_test = _fn_get_data_splits["get_nuscenes_pred_split"].__call__(data_root)
            self.init_frame = 0
            process_cls = _cls_preprocess["preprocess"]

        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = parser.data_root_ethucy
            seq_train, seq_val, seq_test = _fn_get_data_splits["get_ethucy_split"].__call__(parser.dataset)
            self.init_frame = 0
            process_cls = _cls_preprocess["preprocess"]
        elif parser.dataset == "steersim":
            data_root = parser.data_root_steersim
            seq_train, seq_val, seq_test = _fn_get_data_splits["get_steersim_split"].__call__(parser)
            process_cls = _cls_preprocess["steersimProcess"]
        elif parser.dataset == "steersim-segmented":
            data_root = parser.data_root_steersim
            seq_train, seq_val, seq_test = _fn_get_data_splits["get_steersim_split"].__call__(parser)
            process_cls = _cls_preprocess["steersimSegmented"]
        elif parser.dataset == "steersim-evac":
            data_root = parser.data_root_steersim
            seq_train, seq_val, seq_test = _fn_get_data_splits["get_steersim_split"].__call__(parser)
            process_cls = _cls_preprocess["steersimEvac"]
        else:
            raise ValueError('Unknown dataset!')

        self.data_root = data_root

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        if self.dataset == "steersim-evac":
            with multiprocessing.Pool() as pool:
                for seq_name in self.sequence_to_load:
                    print_log("loading sequence {} in parallel...".format(seq_name),
                              log=log, same_line=True, to_begin=True)
                    preprocessor = process_cls(
                        data_root, seq_name, parser, log, self.split, self.phase, eager_init=False)
                    preprocessor.initialize_multi_thread(pool)
                    self.sequence.append(preprocessor)

                for preprocessor in self.sequence:
                    print_log("finishing sequence {} in parallel...".format(preprocessor.seq_name),
                              log=log, same_line=True, to_begin=True)
                    preprocessor.wait_for_initialization()
                    num_seq_samples = preprocessor.get_total_available_sample_size()
                    self.num_total_samples += num_seq_samples
                    self.num_sample_list.append(num_seq_samples)
        else:
            for seq_name in self.sequence_to_load:
                print_log("loading sequence {} ...".format(seq_name), log=log, same_line=True, to_begin=True)
                preprocessor = process_cls(data_root, seq_name, parser, log, self.split, self.phase)
                if self.dataset.startswith("steersim"):
                    num_seq_samples = preprocessor.get_total_available_sample_size()
                else:
                    num_seq_samples = preprocessor.num_fr - (parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip
                self.num_total_samples += num_seq_samples
                self.num_sample_list.append(num_seq_samples)
                self.sequence.append(preprocessor)

        print_log(f'total num samples: {self.num_total_samples}', log)
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)

    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                if self.dataset.startswith("steersim"):
                    return seq_index, index_tmp
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip
                frame_index += self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        sample_index = self.sample_list[self.index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        self.index += 1

        data = seq(frame)
        return data

    def __call__(self):
        return self.next_sample()
