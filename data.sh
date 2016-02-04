# Bash functions to help with preprocessing data
# File naming scheme used: base.type.lang

function check_moses {
  : "${MOSES:?Need to set MOSES}"
}

function check_langs {
  : "${LANG1:?Need to set LANG1}"
  : "${LANG2:?Need to set LANG2}"
}

function min {
  if [ "$1" -lt "$2" ]; then
    echo $1
  else
    echo $2
  fi
}

function tokenize {
  [ $# -lt 1 ] && { echo "Usage: $0 filename"; return 1; }
  check_moses
  base="${1%.*}"
  lang="${1##*.}"
  $MOSES/scripts//tokenizer/tokenizer.perl -threads $(min $(nproc --all) 4) -l $lang < $1 > $base.tok.$lang
}

function truecase {
  [ $# -lt 1 ] && { echo "Usage: $0 filename"; return 1; }
  base="${1%.*}"
  lang="${1##*.}"
  $MOSES/scripts/recaser/train-truecaser.perl --model truecase-model.$lang --corpus $1
  $MOSES/scripts/recaser/truecase.perl --model truecase-model.$lang < $1 > $base.true.$lang
}

function clean {
  [ $# -lt 1 ] && { echo "Usage: $0 basename"; return 1; }
  check_moses
  check_langs
  $MOSES/scripts/training/clean-corpus-n.perl $1 $LANG1 $LANG2 $1.clean 1 80
}

function shuffle {
  [ $# -lt 1 ] && { echo "Usage: $0 basename"; return 1; }
  check_langs
  dd if=/dev/urandom of=rand count=$((128*1024)) status=none
  shuf --random-source=rand $1.$LANG1 > $1.shuf.$LANG1
  shuf --random-source=rand $1.$LANG2 > $1.shuf.$LANG2
  rm rand
}

function create-vocabulary {
  # Pipe to cut -f 2 in order to just get words
  [ $# -lt 1 ] && { echo "Usage: $0 filename"; return 1; }
  tr "[:blank:]" "\n" < $1 | \
    sort -S 8G --compress-program=gzip --batch-size 128 --parallel=$(min $(nproc --all) 4) | \
    uniq -c | sort -k 1nr | awk -v OFS='\t' '{print $1, $2}'
}
