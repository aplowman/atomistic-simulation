#!/bin/bash

host=$1
src_dir=$2
dst_dir=$3

echo "Connecting to host: $host"
echo "Copying from source directory: $src_dir"
echo "...to destination directory: $dst_dir"

if (ssh $host "[ -d $dst_dir ]"); then
    echo "Directory: $dst_dir already exists on remote scratch. `
        `Nothing was copied. Exiting."
    exit 1
fi

# Copy sims directory and set default permissions expected on Unix
rsync -az --chmod=Du=rwx,Dgo=rx,Fu=rw,Fog=r $src_dir $host:$dst_dir
