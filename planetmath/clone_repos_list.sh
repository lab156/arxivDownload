#!/bin/bash

# Download files from TOPIC_LIST


TOPIC_LIST=$1
CLONE_DIR=$2
URL_PREFIX="https://github.com/planetmath/"
URL_SUFFIX=".git"

while IFS= read -r line
do
  GITHUB_URL="$URL_PREFIX""$line""$URL_SUFFIX"
  echo "Cloning dir:" $GITHUB_URL
  git clone $GITHUB_URL $CLONE_DIR$line
done < "$TOPIC_LIST"
