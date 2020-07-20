#! /usr/bin/python
# -*- coding: utf8 -*-

import argparse
import os
import re

import numpy as np

from sklearn.metrics import confusion_matrix, f1_score

from deepsleep.sleep_stage import W, N1, N2, N3, REM


def print_performance(cm):
    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float) # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float) # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    print("Sample: {}".format(np.sum(cm)))
    print("W: {}".format(tpfn[W]))
    print("N1: {}".format(tpfn[N1]))
    print("N2: {}".format(tpfn[N2]))
    print("N3: {}".format(tpfn[N3]))
    print("REM: {}".format(tpfn[REM]))
    print("Confusion matrix:")
    print(cm)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("Overall accuracy: {}".format(acc))
    print("Macro-F1 accuracy: {}".format(mf1))


def perf_overall(data_dir):
    # Remove non-output files, and perform ascending sort
    allfiles = os.listdir(data_dir)
    outputfiles = []
    for idx, f in enumerate(allfiles):
        if re.match("^output_.+\d+\.npz", f):
            outputfiles.append(os.path.join(data_dir, f))
    outputfiles.sort()

    y_true = []
    y_pred = []
    for fpath in outputfiles:
        with np.load(fpath,allow_pickle=True) as f:
            print((f["y_true"].shape))
            if len(f["y_true"].shape) == 1:
                if len(f["y_true"]) < 10:
                    f_y_true = np.hstack(f["y_true"])
                    f_y_pred = np.hstack(f["y_pred"])
                else:
                    f_y_true = f["y_true"]
                    f_y_pred = f["y_pred"]
            else:
                f_y_true = f["y_true"].flatten()
                f_y_pred = f["y_pred"].flatten()

            y_true.extend(f_y_true)
            y_pred.extend(f_y_pred)

            print("File: {}".format(fpath))
            cm = confusion_matrix(f_y_true, f_y_pred, labels=[0, 1, 2, 3, 4])
            print_performance(cm)
    print(" ")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    total = np.sum(cm, axis=1)

    print("DeepSleepNet (current)")
    print_performance(cm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/akara/Workspace/deepsleep_output/results/outputs",
                        help="Directory where to load prediction outputs")
    args = parser.parse_args()

    if args.data_dir is not None:
        perf_overall(data_dir=args.data_dir)

    sharman2017 = np.asarray([
        [7944, 11, 12, 6, 30],
        [183, 113, 123, 4, 181],
        [48, 4, 3334, 149, 86],
        [13, 0, 198, 1088, 0],
        [52, 11, 207, 0, 1339]
    ], dtype=np.int)

    hassan2017 = np.asarray([
        [3971, 28, 6, 0, 23],
        [53, 117, 43, 0, 89],
        [70, 5, 1641, 54, 41],
        [33, 0, 104, 513, 0],
        [41, 24, 84, 1, 655]
    ], dtype=np.int)

    tsinalis2016 = np.asarray([
        [2744, 441, 34, 23, 138],
        [472, 1654, 262, 8, 366],
        [621, 1270, 13696, 1231, 760],
        [143, 7, 469, 4966, 6],
        [308, 899, 340, 0, 6164]
    ], dtype=np.int)

    dong2016 = np.asarray([
        [5022, 577, 188, 19, 395],
        [407, 2468, 989, 4, 965],
        [130, 630, 27254, 1021, 763],
        [13, 0, 1236, 6399, 5],
        [103, 258, 609, 0, 9611]
    ], dtype=np.int)

    hsu2013 = np.asarray([
        [34, 2, 7, 2, 3],
        [0, 20, 23, 3, 9],
        [3, 4, 574, 8, 1],
        [0, 0, 3, 26, 0],
        [3, 5, 13, 4, 213]
    ], dtype=np.int)

    liang2012 = np.asarray([
        [195, 24, 4, 0, 3],
        [61, 72, 48, 3, 69],
        [12, 103, 4078, 216, 220],
        [1, 4, 196, 1309, 0],
        [8, 8, 22, 6, 1818]
    ], dtype=np.int)

    fraiwan2012 = np.asarray([
        [2407, 89, 111, 38, 40],
        [56, 185, 52, 8, 48],
        [69, 85, 1897, 174, 131],
        [14, 9, 86, 482, 3],
        [33, 60, 92, 3, 719]
    ], dtype=np.int)

    print(" ")
    print("Sharma (2017)")
    print_performance(sharman2017)
    print(" ")
    print("Hassan (2017)")
    print_performance(hassan2017)
    print(" ")
    print("Tsinalis (2016)")
    print_performance(tsinalis2016)
    print(" ")
    print("Dong (2016)")
    print_performance(dong2016)
    print(" ")
    print("Hsu (2013)")
    print_performance(hsu2013)
    print(" ")
    print("Liang (2012)")
    print_performance(liang2012)
    print(" ")
    print("Fraiwan (2012)")
    print_performance(fraiwan2012)


if __name__ == "__main__":
    main()
