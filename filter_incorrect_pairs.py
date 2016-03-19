from __future__ import print_function

import argparse
import io
import itertools

from langdetect import detect_langs, lang_detect_exception

LANG_DECISION_PROB = 0.9


def run_lang_detection(src_input, trg_input,
                       src_lang, trg_lang,
                       src_output, trg_output):

    with io.open(src_input, 'r') as fin_src, \
            io.open(trg_input, 'r') as fin_trg, \
            io.open(src_output, 'w') as fout_src, \
            io.open(trg_output, 'w') as fout_trg:
        for src_line, trg_line in itertools.izip(fin_src, fin_trg):
            src_line = src_line.strip()
            trg_line = trg_line.strip()

            # Skip a pair of sentences if either source or target is empty
            if (not src_line) or (not trg_line):
                continue

            try:
                src_languages = detect_langs(src_line)
                trg_languages = detect_langs(trg_line)
            except lang_detect_exception.LangDetectException:
                print(("Failed to detect the languages for the following pair"
                      "\n\tSRC: %s\n\tTRG: %s") % (src_line, trg_line))
                continue

            most_probable_src_lang = sorted(src_languages, reverse=True)[0]
            most_probable_trg_lang = sorted(trg_languages, reverse=True)[0]

            if most_probable_src_lang.lang == src_lang and \
               most_probable_src_lang.prob > LANG_DECISION_PROB and \
               most_probable_trg_lang.lang == trg_lang and \
               most_probable_trg_lang.prob > LANG_DECISION_PROB:

                print(src_line, file=fout_src)
                print(trg_line, file=fout_trg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_corpus', type=str)
    parser.add_argument('--target_corpus', type=str)

    args = parser.parse_args()

    src_base, src_lang = args.source_corpus.rsplit('.', 1)
    trg_base, trg_lang = args.target_corpus.rsplit('.', 1)

    assert(src_base == trg_base)

    src_output = src_base + '.removed.' + src_lang
    trg_output = trg_base + '.removed.' + trg_lang

    run_lang_detection(args.source_corpus, args.target_corpus,
                       src_lang, trg_lang,
                       src_output, trg_output)
