import math
import torch
import random
import numpy as np
import torch.utils.data as tordata


def collate_fn(batch, frame_num, sample_type):
    r""" Following the sampler configurations in GaitPart, implement
    'sequential' sample_type:

        At the training phase, for the length of gait video is uncertain, the
        sampler should collect a fixed-length segments as input: intercept a
        30-40 frame-length segment first, and then randomly extract 30 sorted
        frames for training.

    """
    batch_size = len(batch)
    feature_num = len(batch[0][0])
    seqs = [batch[i][0] for i in range(batch_size)]
    frame_sets = [batch[i][1] for i in range(batch_size)]
    view = [batch[i][2] for i in range(batch_size)]
    seq_type = [batch[i][3] for i in range(batch_size)]
    label = [batch[i][4] for i in range(batch_size)]
    batch = [seqs, view, seq_type, label, None]

    def select_frame(index):
        sample = seqs[index]
        frame_set = frame_sets[index]
        if sample_type == 'random':
            frame_id_list = random.choices(frame_set, k=frame_num)
            sampled_seq = [
                feature.loc[frame_id_list].values for feature in sample
            ]

        elif sample_type == 'sequential':
            if len(frame_set) < 30:
                frame_set = frame_set + frame_set
            intercept_length = min(random.randint(30, 40), len(frame_set))
            start = random.randint(0, len(frame_set) - intercept_length)
            intercept_id_list = [
                frame_set[i] for i in range(start, start + intercept_length)
            ]
            intercept_index = sorted(
                np.random.permutation(list(
                    range(intercept_length)))[:frame_num])
            frame_id_list = [intercept_id_list[i] for i in intercept_index]
            sampled_seq = [
                feature.loc[frame_id_list].values for feature in sample
            ]

        else:
            sampled_seq = [feature.values for feature in sample]
        return sampled_seq

    seqs = list(map(select_frame, range(len(seqs))))

    if sample_type == 'random' or sample_type == 'sequential':
        # sequences have been sampled to the same length, so can
        # be directly concated together.
        seqs = [
            np.asarray([seqs[i][j] for i in range(batch_size)])
            for j in range(feature_num)
        ]
    else:
        gpu_num = min(torch.cuda.device_count(), batch_size)
        batch_per_gpu = math.ceil(batch_size / gpu_num)
        # batch_frames: length of every sequence in a batch. [gpu_num x batch_per_gpu]
        batch_frames = [[
            len(frame_sets[i])
            for i in range(batch_per_gpu * j, batch_per_gpu * (j + 1))
            if i < batch_size
        ] for j in range(gpu_num)]
        # pad the batch on last gpu, make sure it has the same number of sequence
        if len(batch_frames[-1]) != batch_per_gpu:
            for _ in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)
        # maximum number of frames on one gpu
        max_sum_frame = np.max(
            [np.sum(batch_frames[i]) for i in range(gpu_num)])
        # concat frames of all sequences from one gpu
        # seqs: [feature_num x gpu_num]
        seqs = [[
            np.concatenate([
                seqs[i][j]
                for i in range(batch_per_gpu * k, batch_per_gpu *
                               (k + 1)) if i < batch_size
            ], 0) for k in range(gpu_num)
        ] for j in range(feature_num)]
        # pad null frames to each gpu, make sure they have the same number of
        # frames
        # seqs: [feature_num x gpu_num x max_sum_frame x H x W]
        seqs = [
            np.asarray([
                np.pad(seqs[j][i], ((0, max_sum_frame - seqs[j][i].shape[0]),
                                    (0, 0), (0, 0)),
                       'constant',
                       constant_values=0) for i in range(gpu_num)
            ]) for j in range(feature_num)
        ]
        batch[4] = np.asarray(batch_frames)

    batch[0] = seqs
    return batch
