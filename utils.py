import theano
from theano import tensor
import warnings
import six

import numpy
from collections import OrderedDict


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in six.iteritems(params):
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in six.iteritems(zipped):
        new_params[kk] = vv.get_value()
    return new_params


# Turn list of objects with .name attribute into dict
def name_dict(lst):
    d = OrderedDict()
    for obj in lst:
        d[obj.name] = obj
    return d


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in six.iteritems(tparams)]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         state_before *
                         trng.binomial(state_before.shape,
                                       p=0.5,
                                       n=1,
                                       dtype=state_before.dtype),
                         state_before * 0.5)
    return proj


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in six.iteritems(params):
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in six.iteritems(params):
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def uniform_weight(nin, nout, scale=None):
    if scale is None:
        scale = numpy.sqrt(6. / (nin + nout))

    W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')
