#!/bin/bash

# List of directories to search for .rst files
directories=("docs" "examples" "radarx" "tests" ".")

# Loop through each directory and convert .rst to .md
for dir in "${directories[@]}"; do
    find "$dir" -type f -name "*.rst" | while read file; do
        # Convert each .rst file to a corresponding .md file
        pandoc "$file" -f rst -t markdown -o "${file%.rst}.md"
        echo "Converted $file to ${file%.rst}.md"
    done
done
