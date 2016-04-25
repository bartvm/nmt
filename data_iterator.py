import io
import logging
from itertools import count

import numpy
import six

from fuel.datasets.text import TextFile
from fuel.transformers import Merge
from fuel.schemes import ConstantScheme
from fuel.transformers import (Batch, Cache, Mapping, SortMapping, Padding,
                               Filter, Transformer)

LOGGER = logging.getLogger(__name__)

EOS_TOKEN = '<EOS>'  # 0
UNK_TOKEN = '<UNK>'  # 1
EOW_TOKEN = ' '     # 2, only for characters


class Shuffle(Transformer):
    def __init__(self, data_stream, buffer_size, **kwargs):
        if kwargs.get('iteration_scheme') is not None:
            raise ValueError
        super(Shuffle, self).__init__(
            data_stream, produces_examples=data_stream.produces_examples,
            **kwargs)
        self.buffer_size = buffer_size
        self.cache = [[] for _ in self.sources]

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        if not self.cache[0]:
            self._cache()
        return tuple(cache.pop() for cache in self.cache)

    def _cache(self):
        temp_caches = [[] for _ in self.sources]
        for i in range(self.buffer_size):
            try:
                for temp_cache, data in zip(temp_caches,
                                            next(self.child_epoch_iterator)):
                    temp_cache.append(data)
            except StopIteration:
                if i:
                    pass
                else:
                    raise
        shuffled_indices = numpy.random.permutation(len(temp_caches[0]))
        for i in shuffled_indices:
            for temp_cache, cache in zip(temp_caches, self.cache):
                cache.append(temp_cache[i])


def _source_length(sentence_pair):
    """Returns the length of the second element of a sequence.

    This function is used to sort sentence pairs by the length of the
    target sentence.

    """
    return len(sentence_pair[3])


def load_dict(filename, dict_size=0):
    """Load vocab from TSV with words in last column."""
    dict_ = {EOS_TOKEN: 0, UNK_TOKEN: 1}
    with io.open(filename, encoding='utf8') as f:
        if dict_size > 0:
            indices = range(len(dict_), dict_size)
        else:
            indices = count(len(dict_))
        dict_.update(zip(map(lambda x: x.rstrip('\n').split('\t')[-1], f),
                     indices))
    return dict_


def get_stream(source, target, source_char_dict, source_word_dict,
               target_char_dict, target_word_dict, batch_size,
               buffer_multiplier=100, n_chars_source=0, n_words_source=0,
               n_chars_target=0, n_words_target=0,
               max_src_word_length=None, max_trg_word_length=None,
               max_src_char_length=None, max_trg_char_length=None):
    """Returns a stream over sentence pairs.

    Parameters
    ----------
    source : list
        A list of files to read source languages from.
    target : list
        A list of corresponding files in the target language.
    source_char_dict : str
        Path to a tab-delimited text file whose last column contains the
        vocabulary.
    source_word_dict : str
        See `source_char_dict`.
    target_char_dict : str
        See `source_char_dict`.
    target_word_dict : str
        See `source_char_dict`.
    batch_size : int
        The minibatch size.
    buffer_multiplier : int
        The number of batches to load, concatenate, sort by length of
        source sentence, and split again; this makes batches more uniform
        in their sentence length and hence more computationally efficient.
    n_chars_source : int
        The number of characters in the source vocabulary. Pass 0 (default) to
        use the entire vocabulary.
    n_words_source : int
        The number of words in the source vocabulary. Pass 0 (default) to
        use the entire vocabulary.
    n_chars_target : int
        See `n_chars_source`.
    n_words_target : int
        See `n_words_source`.

    """
    if len(source) != len(target):
        raise ValueError("number of source and target files don't match")

    # Read the dictionaries
    dicts = [load_dict(source_char_dict, dict_size=n_chars_source),
             load_dict(source_word_dict, dict_size=n_words_source),
             load_dict(target_char_dict, dict_size=n_chars_target),
             load_dict(target_word_dict, dict_size=n_words_target)]

    # Open the two sets of files and merge them
    streams = [
        TextFile(source, dicts[0], level='character', bos_token=None,
                 eos_token=EOW_TOKEN, encoding='utf-8').get_example_stream(),
        TextFile(source, dicts[1], bos_token=None,
                 eos_token=EOS_TOKEN, encoding='utf-8').get_example_stream(),
        TextFile(target, dicts[2], level='character', bos_token=None,
                 eos_token=EOW_TOKEN, encoding='utf-8').get_example_stream(),
        TextFile(target, dicts[3], bos_token=None,
                 eos_token=EOS_TOKEN, encoding='utf-8').get_example_stream()
    ]
    merged = Merge(streams, ('source_chars', 'source_words',
                             'target_chars', 'target_words'))

    # Filter sentence lengths
    if max_src_word_length or max_trg_word_length:
        def filter_pair(pair):
            src_chars, src_words, \
                trg_chars, trg_words = pair
            src_word_ok = (not max_src_word_length) or \
                len(src_words) < max_src_word_length
            trg_word_ok = (not max_trg_word_length) or \
                len(trg_words) < max_trg_word_length
            src_char_ok = (not max_src_char_length) or \
                len(src_chars) < (max_src_char_length + max_src_word_length)
            trg_char_ok = (not max_trg_char_length) or \
                len(trg_chars) < (max_trg_char_length + max_trg_word_length)

            return src_word_ok and trg_word_ok and src_char_ok and trg_char_ok

        merged = Filter(merged, filter_pair)

    # Batches of approximately uniform size
    large_batches = Batch(
        merged,
        iteration_scheme=ConstantScheme(batch_size * buffer_multiplier)
    )
    sorted_batches = Mapping(large_batches, SortMapping(_source_length))
    batches = Cache(sorted_batches, ConstantScheme(batch_size))
    shuffled_batches = Shuffle(batches, buffer_multiplier)
    masked_batches = Padding(shuffled_batches,
                             mask_sources=('source_words', 'target_words'))

    return masked_batches


def load_data(src, trg,
              valid_src, valid_trg,
              src_char_vocab, src_word_vocab,
              trg_char_vocab, trg_word_vocab,
              n_chars_src, n_words_src,
              n_chars_trg, n_words_trg,
              batch_size, valid_batch_size,
              max_src_word_length, max_trg_word_length,
              max_src_char_length, max_trg_char_length):
    LOGGER.info('Loading data')

    dictionaries = [src_char_vocab, src_word_vocab,
                    trg_char_vocab, trg_word_vocab]
    datasets = [src, trg]
    valid_datasets = [valid_src, valid_trg]

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in six.iteritems(worddicts[ii]):
            worddicts_r[ii][vv] = kk

    train_stream = get_stream([datasets[0]],
                              [datasets[1]],
                              dictionaries[0],
                              dictionaries[1],
                              dictionaries[2],
                              dictionaries[3],
                              n_chars_source=n_chars_src,
                              n_words_source=n_words_src,
                              n_chars_target=n_chars_trg,
                              n_words_target=n_words_trg,
                              batch_size=batch_size,
                              max_src_word_length=max_src_word_length,
                              max_trg_word_length=max_trg_word_length,
                              max_src_char_length=max_src_char_length,
                              max_trg_char_length=max_trg_char_length)
    valid_stream = get_stream([valid_datasets[0]],
                              [valid_datasets[1]],
                              dictionaries[0],
                              dictionaries[1],
                              dictionaries[2],
                              dictionaries[3],
                              n_chars_source=n_chars_src,
                              n_words_source=n_words_src,
                              n_chars_target=n_chars_trg,
                              n_words_target=n_words_trg,
                              batch_size=valid_batch_size)

    return worddicts_r, train_stream, valid_stream
