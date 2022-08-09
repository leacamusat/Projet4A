#!/bin/sh

#script sh to remove comas from test.data, median_test.data ...

python DIMLP_setup.py

list=`ls *.data`
for i in $list
do
  echo $i
  a="virg"
  sed 'y/,/ /' $i > $a${i}
done

