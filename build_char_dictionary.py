import numpy
import cPickle as pkl
import codecs

import sys

from collections import OrderedDict


def main():
    for filename in sys.argv[1:]:
        print 'Processing', filename
        char_freqs = OrderedDict()
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                char_seq = line.strip()
                for c in char_seq:
                    if c not in char_freqs:
                        char_freqs[c] = 0
                    char_freqs[c] += 1
        chars = char_freqs.keys()
        freqs = char_freqs.values()

        sorted_idx = numpy.argsort(freqs)
        sorted_chars = [chars[ii] for ii in sorted_idx[::-1]]

        chardict = OrderedDict()
        n_special_tokens = 3
        chardict['eos'] = 0
        chardict['eow'] = 1
        chardict['UNK'] = 2
        for ii, ww in enumerate(sorted_chars):
            chardict[ww] = ii + n_special_tokens

        with open('%s.pkl' % filename, 'wb') as f:
            pkl.dump(chardict, f)

        print 'Done'


if __name__ == '__main__':
    main()
