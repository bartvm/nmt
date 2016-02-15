from __future__ import print_function
import binascii
import io
import json
import os
import shutil
import sys

import numpy
from mimir import ServerLogger

from platoon.channel import Controller


class NMTController(Controller):
    """
    This multi-process controller implements patience-based early-stopping SGD
    """

    def __init__(self, experiment_id, config, control_port, max_mb):
        """
        Initialize the NMTController

        Parameters
        ----------
        experiment_id : str
            A string that uniquely identifies this run.
        config : dict
            The deserialized JSON configuration file
        control_port : int
            The control port
        max_mb : int
            Max number of minibatches to train on.
        patience: : int
            Training stops when this many minibatches have been trained on
            without any reported improvement.
        valid_freq : int
            Number of minibatches to train on between every monitoring step.
            Should be a multiple of train_len!
        """
        self.config = config
        super(NMTController, self).__init__(control_port)
        self.patience = config['training']['patience']
        self.max_mb = int(max_mb)

        self.valid_freq = config['management']['valid_freq']
        self.uidx = 0
        self.bad_counter = 0
        self.min_valid_cost = numpy.inf

        self.valid = False

        self.experiment_id = experiment_id
        ServerLogger(filename='{}.log.jsonl.gz'.format(self.experiment_id),
                     threaded=True)

    def handle_control(self, req, worker_id):
        """
        Handles a control_request received from a worker

        Parameters
        ----------
        req : str or dict
            Control request received from a worker.
            The control request can be one of the following
            1) "next" : request by a worker to be informed of its next action
               to perform. The answers from the server can be 'train' (the
               worker should keep training on its training data), 'valid' (the
               worker should perform monitoring on its validation set and test
               set) or 'stop' (the worker should stop training).
            2) dict of format {"done":N} : used by a worker to inform the
                server that is has performed N more training iterations and
                synced its parameters. The server will respond 'stop' if the
                maximum number of training minibatches has been reached.
            3) dict of format {"valid_err":x, "test_err":x2} : used by a worker
                to inform the server that it has performed a monitoring step
                and obtained the included errors on the monitoring datasets.
                The server will respond "best" if this is the best reported
                validation error so far, otherwise it will respond 'stop' if
                the patience has been exceeded.
        """
        control_response = ""

        if req == 'config':
            control_response = self.config
        elif req == 'experiment_id':
            control_response = self.experiment_id
        elif req == 'next':
            if self.valid:
                self.valid = False
                control_response = 'valid'
            else:
                control_response = 'train'
        elif 'done' in req:
            self.uidx += req['done']

            if numpy.mod(self.uidx, self.valid_freq) == 0:
                self.valid = True
        elif 'valid_err' in req:
            valid_err = req['valid_err']

            if valid_err <= self.min_valid_cost:
                self.bad_counter = 0
                self.min_valid_cost = valid_err
                control_response = 'best'
            else:
                self.bad_counter += 1

        if self.uidx > self.max_mb or self.bad_counter > self.patience:
            control_response = 'stop'
            self.worker_is_done(worker_id)

        return control_response


if __name__ == '__main__':
    # Load the configuration file
    with io.open(sys.argv[1]) as f:
        config = json.load(f)
    # Create unique experiment ID and backup config file
    experiment_id = binascii.hexlify(os.urandom(3)).decode()
    shutil.copyfile(sys.argv[1], '{}.config.json'.format(experiment_id))
    # Start controller
    l = NMTController(experiment_id, config, control_port=5567,
                      max_mb=(5000 * 1998) / 10)
    l.serve()
