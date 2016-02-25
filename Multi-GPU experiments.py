
# coding: utf-8

# In[1]:

import glob
import gzip
import json
from collections import defaultdict
from itertools import cycle

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

def smooth(window, x, y=None):
    def convolve(window, z):
        return np.convolve(z, np.ones((window,)) / window, mode='same')[window // 2:-window // 2]
    if y is None:
        return convolve(window, x)
    else:
        return x[window // 2:-window // 2], convolve(window, y)


# In[93]:

# Plotting settings
plt.figure(figsize=(12,8))
plt.rc('text', usetex=True)
colors = iter(sns.color_palette("Paired", 10))
smoothing = 1000

# Multi-GPU experiments
experiments = list(sorted(set(map(lambda filename: filename.split('.')[0], glob.glob('*.gz')))))
experiments.remove('6b8a3e')
experiments = experiments + ['6b8a3e']

with gzip.open('6b8a3e.log.jsonl.gz', mode='rt') as f:
    _times, _costs = [], []
    start_time = json.loads(f.readline())['train_time']
    f.seek(0)
    for entry in f:
        entry = json.loads(entry)
        if 'cost' not in entry:
            continue
        _times.append(entry['train_time'] - start_time)
        _costs.append(entry['cost'] / entry['average_target_length'])
    _times, _costs = map(np.asarray, smooth(smoothing, _times, _costs))
    i = np.argsort(_costs)
    _times, _costs = _times[i], _costs[i]

for experiment in sorted(experiments):
    with open('{}.config.json'.format(experiment)) as f:
        config = json.loads(f.read())
    if experiment == '6b8a3e' or config['multi'].get('algorithm', 'easgd') == 'asgd':
        continue
    with gzip.open('{}.log.jsonl.gz'.format(experiment), mode='rt') as f:
        workers = set()
        times, costs = defaultdict(list), defaultdict(list)
        start_time = json.loads(f.readline())['train_time']
        f.seek(0)
        for entry in f:
            entry = json.loads(entry)
            if 'cost' not in entry:
                continue
            workers.add(entry['remote_log'])
            costs[entry['remote_log']].append(entry['cost'] / entry['average_target_length'])
            times[entry['remote_log']].append(entry['train_time'] - start_time)
    color = next(colors)
    for worker in costs.keys():
        _t, _c = smooth(smoothing, times[worker], costs[worker])
        _t_div = np.interp(_c, _costs, _times)
        plt.plot(_t, _t_div / _t,
                 color=color, alpha=0.8,
                 label='{} GPUs, $\tau={}$, $\\beta={}$'
                   .format(len(workers),
                           config['multi']['train_len'],
                           config['multi']['beta']) if worker == 1 else None)

# Plotting
plt.legend()
plt.xlabel('Training time (s)')
plt.ylabel('Speedup (smoothed over {} entries)'.format(smoothing))
plt.ylim([0.5, 2])
plt.show()


# In[91]:

# Plotting settings
plt.figure(figsize=(12,8))
plt.rc('text', usetex=True)
colors = iter(sns.color_palette("Paired", 10))
smoothing = 1000

# Multi-GPU experiments
experiments = list(sorted(set(map(lambda filename: filename.split('.')[0], glob.glob('*.gz')))))
experiments.remove('6b8a3e')
experiments = experiments + ['6b8a3e']

for experiment in sorted(experiments):
    with open('{}.config.json'.format(experiment)) as f:
        config = json.loads(f.read())
    if config['multi'].get('algorithm', 'easgd') == 'asgd':
        continue
    with gzip.open('{}.log.jsonl.gz'.format(experiment), mode='rt') as f:
        workers = set()
        times, costs = defaultdict(list), defaultdict(list)
        start_time = json.loads(f.readline())['train_time']
        f.seek(0)
        for entry in f:
            entry = json.loads(entry)
            if 'cost' not in entry:
                continue
            workers.add(entry['remote_log'])
            costs[entry['remote_log']].append(entry['cost'] / entry['average_target_length'])
            times[entry['remote_log']].append(entry['train_time'] - start_time)
    color = next(colors)
    for worker in costs.keys():
        plt.plot(*smooth(smoothing, times[worker], costs[worker]),
                 color=color, alpha=0.8,
                 label='{} GPUs, $\tau={}$, $\\beta={}$'
                   .format(len(workers),
                           config['multi']['train_len'],
                           config['multi']['beta']) if worker == 1 else None)

# Plotting
plt.legend()
plt.xlabel('Training time (s)')
plt.ylabel('Training cost per token (smoothed over {} entries)'.format(smoothing))
plt.show()


# In[94]:

# Plotting settings
plt.figure(figsize=(12,8))
plt.rc('text', usetex=True)
colors = iter(sns.color_palette("Paired", 10))
smoothing = 1000

# Multi-GPU experiments
experiments = set(map(lambda filename: filename.split('.')[0], glob.glob('*.gz')))

for experiment in sorted(experiments):
    with open('{}.config.json'.format(experiment)) as f:
        config = json.loads(f.read())
    if config['multi'].get('algorithm', 'easgd') == 'easgd':
        continue
    with gzip.open('{}.log.jsonl.gz'.format(experiment), mode='rt') as f:
        workers = set()
        times, costs = defaultdict(list), defaultdict(list)
        start_time = json.loads(f.readline())['train_time']
        f.seek(0)
        for entry in f:
            entry = json.loads(entry)
            if 'cost' not in entry:
                continue
            workers.add(entry['remote_log'])
            costs[entry['remote_log']].append(entry['cost'] / entry['average_target_length'])
            times[entry['remote_log']].append(entry['train_time'] - start_time)
    color = next(colors)
    for worker in costs.keys():
        plt.plot(*smooth(smoothing, times[worker], costs[worker]),
                 color=color, alpha=0.8,
                 label='{} GPUs, communication period {}'
                   .format(len(workers),
                           config['multi']['train_len']) if worker == 1 else None)

# Plotting
plt.legend()
plt.xlabel('Training time (s)')
plt.ylabel('Training cost per token (smoothed over {} entries)'.format(smoothing))
plt.show()


# In[ ]:



