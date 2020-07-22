#!/bin/bash

rm -f TODO.md
# look for TODOs in the code but filter out some specific files from the
# search because they just add noise.
echo "THIS IS A PROGRMATICALLY GENERATED FILE. DO NOT EDIT BY HAND." > TODO.md
grep -r TODO * | \
  grep -vi "run_update_todos.sh" | \
  grep -vi "README.md" \
  >> TODO.md
