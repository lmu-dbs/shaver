import pickle
from datetime import datetime
import itertools
import logging
import os
import sys
import time

import numpy as np
import pm4py
from spmf import Spmf

import storage
import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
rootLogger = logging.getLogger()


def create_successive_std(traces_with_timestamps):
    acts = dict()
    for tid, trace in enumerate(traces_with_timestamps):
        for e in trace:
            if str(e[0]) not in acts.keys():
                acts[str(e[0])] = []
        if len(trace) > 1:
            for e1 in trace:
                a1 = e1[0]
                if str(a1) in acts.keys():
                    acts[str(a1)].append((e1[2] - e1[1]).total_seconds())
                else:
                    acts[str(a1)] = [(e1[2] - e1[1]).total_seconds()]
        elif len(trace) == 1:
            a1 = trace[0][0]
            if str(a1) in acts.keys():
                acts[str(a1)].append(0)
            else:
                acts[str(a1)] = [0]
        else:
            print("ERROR: Trace is empty!")

    avg = {}
    std = {}
    ext = {}
    mysum = {}
    mymax = {}

    for rel, values in acts.items():
        if not values:
            values = [0]
        avg[rel] = np.mean(values)
        std[rel] = np.std(values)
        ext[rel] = np.max(np.abs(values))
        mysum[rel] = np.sum(values)
        mymax[rel] = np.max(values)

    return avg, std, ext, mysum, mymax


def import_bpic11(f):
    log = pm4py.read_xes('data/bpic11/Hospital_log.xes')  # if preprocessing is necessary make sure that the corresponding log exists
    log, storage.TRACES_EVENTS, storage.TRACES_TIMESTAMPS, traces_end_timestamps, mapping_dict = utils.convert_log_to_traces_bpic11(
        log, f)
    rootLogger.info(f"Mapping dict: {mapping_dict}")

    traces_with_timestamps = []
    for i in range(len(storage.TRACES_EVENTS)):
        traces_with_timestamps.append(
            list(zip(storage.TRACES_EVENTS[i], storage.TRACES_TIMESTAMPS[i], traces_end_timestamps[i])))

    return log, traces_with_timestamps, mapping_dict

