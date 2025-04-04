#!/bin/bash
# Script to test both Python Cog and direct Rust CLI calls

# Make script executable
chmod +x predict.py

echo "=== Building Rust binary ==="
cargo build --release

echo -e "\n=== Testing via Python Cog ==="
cog predict \
  -i base="11111111111111111111111111111111" \
  -i owner="BPFLoaderUpgradeab1e11111111111111111111111" \
  -i target="auto" \
  -i case_insensitive=false \
  -i target_type="suffix" \
  -i num_results=10