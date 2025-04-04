# Solana Vanity Address Generator for Replicate

This project provides a Solana vanity address generator that can be deployed on Replicate to leverage cloud GPU resources for faster address generation.

# To Test

`cargo build && bash grind.sh`

## Features

- Generate Solana vanity addresses with custom prefixes or suffixes
- Support for case-insensitive matching
- Parallel processing with multiple threads
- Option to generate multiple addresses in a single request
- Compatible with Replicate's prediction API
- GPU acceleration in Replicate environment, CPU acceleration on local Mac

## Request Format

```json
{
  "base": "Base58EncodedPublicKey",
  "owner": "Base58EncodedOwnerKey",
  "target": "desiredPattern",
  "case_insensitive": true,
  "num_results": 1,
  "target_type": "prefix" // or "suffix"
}
```

## Response Format

```json
{
  "results": [
    {
      "pubkey": "Base58EncodedPublicKey",
      "seed": "GeneratedSeed",
      "attempts": 12345,
      "time_secs": 1.23
    }
  ]
}
```

## Local Development (Mac)

For Mac users, you can run the CPU implementation locally:

```bash
# Build without CUDA support
cargo build --release 

# Run locally with test input
cargo run --release -- predict -i base="11111111111111111111111111111111" -i owner="BPFLoaderUpgradeab1e11111111111111111111111" -i target="auto" -i case_insensitive=true -i target_type="suffix"
```

The CPU implementation will utilize all available cores on your Mac for parallel processing.

## Deployment to Replicate (with GPU)

1. Install the Cog CLI: `pip install cog`
2. Push to Replicate: `cog push your-username/vanity-generator`

When deployed to Replicate, the model will automatically use GPU acceleration if available.

## Testing with Node.js

See the `/test` directory for a Node.js script that demonstrates how to interact with the deployed model. 