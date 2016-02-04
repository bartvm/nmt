from itertools import count

from fuel.datasets.text import TextFile
from fuel.transformers import Merge
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Cache, Mapping, SortMapping


def _source_length(sentence_pair):
    """Returns the length of the first element of a sequence.

    This function is used to sort sentence pairs by the length of the
    source sentence.

    """
    return len(sentence_pair[0])


def load_dict(filename, n_words=0):
    """Load vocab from TSV with words in last column."""
    dict_ = {'<UNK>': 0}
    with open(filename) as f:
        if n_words > 0:
            indices = range(1, n_words)
        else:
            indices = count(1)
        dict_.update(zip(map(lambda x: x.split()[-1], f), indices))
    return dict_


def get_stream(source, target, source_dict, target_dict, batch_size=128,
               buffer_multiplier=100, n_words_source=0, n_words_target=0):
    """Returns a stream over sentence pairs.

    Parameters
    ----------
    source : list
        A list of files to read source languages from.
    target : list
        A list of corresponding files in the target language.
    source_dict : str
        Path to a tab-delimited text file whose last column contains the
        vocabulary.
    target_dict : str
        See `source_dict`.
    batch_size : int
        The minibatch size.
    buffer_multiplier : int
        The number of batches to load, concatenate, sort by length of
        source sentence, and split again; this makes batches more uniform
        in their sentence length and hence more computationally efficient.
    n_words_source : int
        The number of words in the source vocabulary. Pass 0 (default) to
        use the entire vocabulary.
    n_words_target : int
        See `n_words_source`.

    """
    if len(source) != len(target):
        raise ValueError("number of source and target files don't match")

    # Read the dictionaries
    dicts = []
    for dict_file, dict_size in zip([source_dict, target_dict],
                                    [n_words_source, n_words_target]):
        dicts.append(load_dict(dict_file, dict_size))

    # Open the two sets of files and merge them
    streams = []
    for lang_files, dictionary in zip([source, target], dicts):
        dataset = TextFile(lang_files, dictionary, bos_token=None,
                           eos_token=None)
        streams.append(dataset.get_example_stream())
    merged = Merge(streams, ('source', 'target'))

    # Batches of approximately uniform size
    large_batches = Batch(
        merged,
        iteration_scheme=ConstantScheme(batch_size * buffer_multiplier)
    )
    sorted_batches = Mapping(large_batches, SortMapping(_source_length))
    batches = Cache(sorted_batches, ConstantScheme(batch_size))
    # masked_batches = Padding(batches)

    return batches
