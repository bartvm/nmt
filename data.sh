# Bash functions to help with preprocessing data
# File naming scheme used: base.type.lang

function check_moses {
  : "${MOSES:?Need to set MOSES}"
}

function check_langs {
  : "${LANG1:?Need to set LANG1}"
  : "${LANG2:?Need to set LANG2}"
  if [[ "$LANG1" == "$LANG2" ]]
  then
    echo "LANG1 and LANG2 cannot be the same"
    return 1
  fi
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
  [ $# -lt 1 ] && { echo "Usage: $0 filename [truecase-model]"; return 1; }
  check_moses
  base="${1%.*}"
  lang="${1##*.}"
  if [[ -z "$2" ]]
  then
    model="${base%.*}.truecase-model.$lang"
    $MOSES/scripts/recaser/train-truecaser.perl --model $model --corpus $1
  else
    model="$2"
  fi
  $MOSES/scripts/recaser/truecase.perl --model $model < $1 > $base.true.$lang
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

# Pipe vocabularies to cut -f 2 to just get words

function create-vocabulary {
  [ $# -lt 1 ] && { echo "Usage: $0 filename"; return 1; }
  awk -v OFS="\t" '{ for(i=1; i<=NF; i++) w[$i]++ } END { for(i in w) print w[i], i }' $1 | sort -k 1nr
}

function create-char-vocabulary {
  [ $# -lt 1 ] && { echo "Usage: $0 filename"; return 1; }
  awk -v OFS="\t" -v FS="" '{ for(i=1; i<=NF; i++) w[$i]++ } END { for(i in w) print w[i], i }' $1 | sort -k 1nr
}
