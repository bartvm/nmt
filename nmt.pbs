#!/bin/bash
#PBS -A jvb-000-ag
#PBS -l signal=SIGTERM@300
#PBS -m ae

# Invocation: msub nmt.pbs -F "\"config.json\"" -l nodes=1:gpus=2 -l walltime=1:00:00 -l feature=k80

echo "Using config file $1"

# Kill this job if any of these commands fail
set -e

cd "${PBS_O_WORKDIR}"

# This should load CUDA as well as cuDNN
module load cuda/7.5.18 libs/cuDNN/4

# Use own Python installation
export PATH=$HOME/miniconda3/bin:$PATH

# This is where we store e.g. jq
export PATH=${RAP}nmt/bin:$PATH

# Use the following Theano settings
declare -A theano_flags
theano_flags["floatX"]="float32"
theano_flags["force_device"]="true"
theano_flags["base_compiledir"]="/rap/jvb-000-aa/${USER}/theano_compiledir"
theano_flags["lib.cnmem"]="0.9"
theano_flags["dnn.enabled"]="True"

function join { local IFS="$1"; shift; echo "$*"; }
function merge {
  for i in $(seq 1 $(($# / 2)))
  do
    eval "k=\${$i}"
    eval "v=\${$(($i + $# / 2))}"
    rval[$i]="$k=$v"
  done
  echo $(join , ${rval[@]})
}

export THEANO_FLAGS=$(merge ${!theano_flags[@]} ${theano_flags[@]})
echo "THEANO_FLAGS=$THEANO_FLAGS"

# Make sure we pick a port that isn't in use already
control_port=$(($(((RANDOM<<15)|RANDOM)) % 16383 + 49152))
batch_port=$(($(((RANDOM<<15)|RANDOM)) % 16383 + 49152))
log_port=$(($(((RANDOM<<15)|RANDOM)) % 16383 + 49152))

# Try to connect to ports to see if they are taken
# type nc &>/dev/null
# until ! (nc -z localhost $control_port || false)
# do
#   echo "Trying another control port!"
#   control_port=$((control_port+1))
# done
# until ! (nc -z localhost $batch_port || false) && (( batch_port != control_port ))
# do
#   echo "Trying another batch port!"
#   batch_port=$((batch_port+1))
# done

# Write ports to config
_1="$(mktemp)"
cat "$1" | jq ".multi.control_port |= $control_port | .multi.batch_port |= $batch_port | .multi.log_port |= $log_port" > "$_1"

# Read data from Luster parallel file system
echo "Working from ${RAP}nmt"
files=(src trg src_vocab trg_vocab valid_src valid_trg)
for file in "${files[@]}"
do
  filename="${RAP}nmt/$(cat "$_1" | jq -r ".data.$file")"
  test -e "$filename" || (echo "$filename doesn't exist" && exit 1)
  FILTERS[$((${#FILTERS[@]} + 1))]=".data.$file |= \"${RAP}nmt/\" + ."
done
_2="$(mktemp)"
cat "$_1" | jq  "$(join '|' "${FILTERS[@]}")" > "$_2"

# Print final config
cat "$_2" | jq '.'

# The following GPUs are available
for id in $(nvidia-smi --query-gpu=index --format=csv,noheader)
do
  GPUS[${#GPUS[@]}]="gpu$id"
done
echo "Using GPUs ${GPUS[*]}"

# Make sure the GPU and cuDNN work
THEANO_FLAGS=device=${GPUS[0]} python -c "import theano.sandbox.cuda; theano.sandbox.cuda.dnn_available()"

# For some strange reason this is set to C (ANSI_X3.4-1968)
export LANG=en_US.UTF-8

# Let's run this thing
set +e
platoon-launcher nmt ${GPUS[@]} -d -c "$_2 ${#GPUS[@]}" -w "$control_port"
