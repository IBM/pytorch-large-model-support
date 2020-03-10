import csv
import ctypes
import os
import time
import torch
import statistics


STAT_KEYS = [('reclaimed_blocks',        'reclaimed'),
             ('reclaimed_bytes',         'reclaimed_bytes'),
             ('alloc_freelist',          'alloc_distribution.freelist'),
             ('alloc_cudamalloc',        'alloc_distribution.cudamalloc'),
             ('alloc_reclaim_one',       'alloc_distribution.reclaim_one'),
             ('alloc_reclaim_fragments', 'alloc_distribution.reclaim_fragments'),
             ('alloc_reclaim_all',       'alloc_distribution.reclaim_all'),
             ('alloc_cudamalloc_retry',  'alloc_distribution.cudamalloc_retry')]

ALL_KEYS = ['duration'] + [k[0] for k in STAT_KEYS]

class LMSStats():
    def __init__(self, gpu_id=0):
        self._gpu_id = gpu_id
        self._start_stats = {k:0 for k in ALL_KEYS}
        self._end_stats = self._start_stats.copy()
        self._delta = self._start_stats.copy()
        self._cumulative_stats = self._start_stats.copy()
        self._num_steps = 0
        self._step_times = []

    def _get_stats(self):
        s = torch.cuda.memory_stats(self._gpu_id)
        stats = {k[0]:s[k[1]] for k in STAT_KEYS}
        stats['duration'] = time.time()
        return stats

    def step_begin(self):
        self._start_stats = self._get_stats()

    def step_end(self):
        self._num_steps += 1
        self._end_stats = self._get_stats()
        self._delta = {k: self._end_stats[k] - self._start_stats[k] for k in ALL_KEYS}
        for k in ALL_KEYS:
            self._cumulative_stats[k] += self._delta[k]
        self._cumulative_stats['num_steps'] = self._num_steps
        self._step_times.append(self._delta['duration'])

    def get_last_step_delta(self):
        return self._delta.copy()

    def get_average_stats(self):
        if self._num_steps:
            s = self._num_steps * 1.0
            average = {k: self._cumulative_stats[k] / s for k in ALL_KEYS}
        else:
            average = {k: 0 for k in ALL_KEYS}
        average['num_steps'] = self._num_steps
        return average

    def get_median_time(self):
        if not self._step_times:
            return 0
        return statistics.median(self._step_times)


class LMSStatsLogger():
    def __init__(self, logfile, gpu_id=0):
        self._epoch = 0
        self._logfile = logfile
        self._lms_stats = LMSStats(gpu_id=gpu_id)
        self._write_header()

    def epoch_begin(self, epoch):
        self._epoch = epoch

    def train_batch_begin(self, batch):
        self._lms_stats.step_begin()

    def train_batch_end(self, batch):
        self._lms_stats.step_end()
        self._write_step_stats('t', batch)

    def train_end(self):
        pass

    def validation_batch_begin(self, batch):
        self._lms_stats.step_begin()

    def validation_batch_end(self, batch):
        self._lms_stats.step_end()
        self._write_step_stats('v', batch)

    def _write_header(self):
        header = ['step type', 'epoch', 'step'] + ALL_KEYS
        with open(self._logfile, 'w', newline='') as csvfile:
            statswriter = csv.writer(csvfile)
            statswriter.writerow(header)

    def _write_step_stats(self, step_type, step_num):
        delta = self._lms_stats.get_last_step_delta()
        row = [step_type, self._epoch, step_num] + [delta[k] for k in ALL_KEYS]
        with open(self._logfile, 'a+', newline='') as csvfile:
            statswriter = csv.writer(csvfile)
            statswriter.writerow(row)


class LMSStatsSummary():
    def __init__(self, logfile, input_shape, gpu_id=0,
                 batch_size=1, start_epoch=0, start_batch=2):
        self._epoch = 0
        self._logfile = logfile
        self._lms_stats = LMSStats(gpu_id=gpu_id)
        self._input_shape = input_shape
        self._start_epoch = start_epoch
        self._start_batch = start_batch
        self._batch_size = batch_size

    def _should_record(self, batch):
        if (batch >= self._start_batch) and (self._epoch >= self._start_epoch):
            return True
        return False

    def epoch_begin(self, epoch):
        self._epoch = epoch

    def train_batch_begin(self, batch):
        if not self._should_record(batch):
            return
        self._lms_stats.step_begin()

    def train_batch_end(self, batch):
        if not self._should_record(batch):
            return
        self._lms_stats.step_end()

    def train_end(self):
        stats_dict = self._lms_stats.get_average_stats()

        input_size_field = 'image_size'
        stats_dict[input_size_field] = self._input_shape[0]

        input_size = self._batch_size
        for dim in self._input_shape:
            input_size *= dim
        input_size /= 1000000.0

        rate_field = 'megapixels/sec' if len(self._input_shape) == 2 else 'megavoxels/sec'
        duration = stats_dict['duration']
        stats_dict[rate_field] = input_size / duration if duration != 0 else 0

        median_rate_field = 'median ' + rate_field
        duration = self._lms_stats.get_median_time()
        stats_dict[median_rate_field] = input_size / duration if duration != 0 else 0

        reclaimed_field = 'reclaimed_bytes'

        # Put these columns first
        fieldnames = [input_size_field, rate_field, median_rate_field, reclaimed_field]
        dictkeys = list(stats_dict)
        for k in fieldnames:
            dictkeys.remove(k)
        fieldnames.extend(dictkeys)

        write_header = not os.path.exists(self._logfile)
        with open(self._logfile, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(stats_dict)

    def validation_batch_begin(self, batch):
        pass

    def validation_batch_end(self, batch):
        pass
