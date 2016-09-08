#!/usr/bin/env bash

set -e
set -o nounset

basedir=$(cd $(dirname $0)/..;pwd)

function fat_echo() {
    echo "############################################"
    echo "########## $1"
}

function wget_or_curl() {
  [ $# -eq 2 ] || { echo "Usage: wget_or_curl <url> <fpath>" && exit 1; }
  if type wget &> /dev/null; then
    local download_cmd="wget -T 10 -t 3 -O"
  else
    local download_cmd="curl -L -o"
  fi
  $download_cmd "$2" "$1"
}

if [ ! -e "$basedir/rnnlm" ]; then
    fat_echo "Downloading RNNLM Toolkit from rnnlm.org"
    # rm -rf $basedir/rnnlm-tk
    (
        wget_or_curl https://f25ea9ccb7d3346ce6891573d543960492b92c30.googledrive.com/host/0ByxdPXuxLPS5RFM5dVNvWVhTd0U/rnnlm-0.4b.tgz $basedir/rnnlm-0.4b.tgz
        tar -xf $basedir/rnnlm-0.4b.tgz
        mv $basedir/rnnlm-0.4b $basedir/rnnlm
        echo -e '\n#ifndef _GNU_SOURCE\ndouble exp10(double x)\n{\n    return pow((double) 10, x);\n}\n#endif' >> $basedir/rnnlm/rnnlmlib.h
        rm $basedir/rnnlm-0.4b.tgz
    )
fi

if [ ! -e "$basedir/rnnlm.py" ]; then
    ln -s $basedir/rnnlm-python/rnnlm.py $basedir/rnnlm.py
fi

