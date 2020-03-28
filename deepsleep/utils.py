import itertools
import numpy as np


def get_balance_class_downsample(x, y):
    """
    Balance the number of samples of all classes by (downsampling):
        1. Find the class that has a smallest number of samples
        2. Randomly select samples in each class equal to that smallest number
    """

    class_labels = np.unique(y)
    n_min_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_min_classes == -1:
            n_min_classes = n_samples
        elif n_min_classes > n_samples:
            n_min_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        idx = np.random.permutation(idx)[:n_min_classes]
        balance_x.append(x[idx])
        balance_y.append(y[idx])
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def get_balance_class_oversample(x, y):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """
    Generate a generator that return a batch of inputs and targets.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def iterate_seq_minibatches(inputs, targets, batch_size, seq_length, stride):
    """
    Generate a generator that return a batch of sequence inputs and targets.
    """
    assert len(inputs) == len(targets)
    n_loads = (batch_size * stride) + (seq_length - stride)
    for start_idx in range(0, len(inputs) - n_loads + 1, (batch_size * stride)):
        seq_inputs = np.zeros((batch_size, seq_length) + inputs.shape[1:],
                              dtype=inputs.dtype)
        seq_targets = np.zeros((batch_size, seq_length) + targets.shape[1:],
                               dtype=targets.dtype)
        for b_idx in range(batch_size):
            start_seq_idx = start_idx + (b_idx * stride)
            end_seq_idx = start_seq_idx + seq_length
            seq_inputs[b_idx] = inputs[start_seq_idx:end_seq_idx]
            seq_targets[b_idx] = targets[start_seq_idx:end_seq_idx]
        flatten_inputs = seq_inputs.reshape((-1,) + inputs.shape[1:])
        flatten_targets = seq_targets.reshape((-1,) + targets.shape[1:])
        yield flatten_inputs, flatten_targets


def iterate_batch_seq_minibatches(inputs, targets, batch_size, seq_length):
    assert len(inputs) == len(targets)
    n_inputs = len(inputs)
    batch_len = n_inputs // batch_size

    epoch_size = batch_len // seq_length
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or seq_length")

    seq_inputs = np.zeros((batch_size, batch_len) + inputs.shape[1:],
                          dtype=inputs.dtype)
    seq_targets = np.zeros((batch_size, batch_len) + targets.shape[1:],
                           dtype=targets.dtype)

    for i in range(batch_size):
        seq_inputs[i] = inputs[i*batch_len:(i+1)*batch_len]
        seq_targets[i] = targets[i*batch_len:(i+1)*batch_len]

    for i in range(epoch_size):
        x = seq_inputs[:, i*seq_length:(i+1)*seq_length]
        y = seq_targets[:, i*seq_length:(i+1)*seq_length]
        flatten_x = x.reshape((-1,) + inputs.shape[1:])
        flatten_y = y.reshape((-1,) + targets.shape[1:])
        yield flatten_x, flatten_y


def iterate_list_batch_seq_minibatches(inputs, targets, batch_size, seq_length):
    for idx, each_data in enumerate(zip(inputs, targets)):
        each_x, each_y = each_data
        seq_x, seq_y = [], []
        for x_batch, y_batch in iterate_seq_minibatches(inputs=each_x, 
                                                        targets=each_y, 
                                                        batch_size=1, 
                                                        seq_length=seq_length, 
                                                        stride=1):
            seq_x.append(x_batch)
            seq_y.append(y_batch)
        seq_x = np.vstack(seq_x)
        seq_x = seq_x.reshape((-1, seq_length) + seq_x.shape[1:])
        seq_y = np.hstack(seq_y)
        seq_y = seq_y.reshape((-1, seq_length) + seq_y.shape[1:])
        
        for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=seq_x, 
                                                              targets=seq_y, 
                                                              batch_size=batch_size, 
                                                              seq_length=1):
            x_batch = x_batch.reshape((-1,) + x_batch.shape[2:])
            y_batch = y_batch.reshape((-1,) + y_batch.shape[2:])
            yield x_batch, y_batch
