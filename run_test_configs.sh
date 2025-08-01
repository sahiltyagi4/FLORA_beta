#!/bin/bash

clear

configs=($(ls ./conf/test_*.yaml | sed 's|./conf/||; s|\.yaml$||'))

echo "Running tests with the following configurations:"
for config in "${configs[@]}"; do
    echo "- $config"
done

echo

for i in {5..1}; do
    echo "Starting tests in $i seconds..."
    sleep 1
done

echo

for config in "${configs[@]}"; do

    ./main.sh --config-name "$config"

    echo
    echo
done
