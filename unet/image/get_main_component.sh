#!/bin/bash

rootdir=$1
newdir=$2

for f in $rootdir/*; do
  echo $f
  filename=$(basename "$f")
  c3d \
  $f -popas S \
  -push S -thresh 1 inf 1 0 -comp -popas C \
  -push C -thresh 1 1 1 0 \
  -push S -multiply \
  -o $newdir/$filename
done
