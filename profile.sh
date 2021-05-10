#!/bin/bash
python_file=$1
nsys_cmd="nsys profile --sample=none --backtrace=none --cpuctxsw=none -f true"
# echo ${python_file}

for idx in {0..6}
do
    prefix=${python_file%%/*}
    profile_file="profile/${prefix}-${idx}"
    # echo "${nsys_cmd} -o ${profile_file} python ${python_file} ${idx}"
    ${nsys_cmd} -o ${profile_file} python ${python_file} ${idx}
done