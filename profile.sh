#!/bin/bash
python_file=$1
nsys_cmd="nsys profile --sample=none --backtrace=none --cpuctxsw=none"
# echo ${python_file}

for idx in {0..6}
do
    profile_file="profile/ex-${idx}"
    # echo "${nsys_cmd} -o ${profile_file} python ${python_file} ${idx}"
    ${nsys_cmd} -o ${profile_file} python ${python_file} ${idx}
done