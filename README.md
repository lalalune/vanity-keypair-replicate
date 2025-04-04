# Solana Vanity Address Generator for Replicate

This project provides a Solana vanity address generator that can be deployed on Replicate to leverage cloud GPU resources for faster address generation. It uses Python Cog to interface with the Replicate API while leveraging performant Rust code for the core generation logic.

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

## Local Development & Testing (Mac/Linux with Cog)

The primary way to build and test the model locally is using the Cog CLI.

1.  **Install Cog:**
    ```bash
    pip install cog
    ```
2.  **Build Rust & Run Cog Predict:**
    The `test-cog.sh` script handles building the underlying Rust code and running a local prediction using Cog.
    ```bash
    bash test-cog.sh
    ```
    This script essentially performs:
    ```bash
    # Build the Rust binary needed by the Python Cog wrapper
    cargo build --release

    # Run the prediction via Cog (which uses predict.py)
    cog predict -i target="auto" -i target_type="suffix"
    ```
    The Rust code utilizes available CPU cores for parallel processing during local execution.

## Deployment to Replicate (with GPU)

Use the Cog CLI to deploy the model to Replicate:

1.  **Login to Cog (if needed):**
    ```bash
    cog login
    ```
2.  **Push to Replicate:**
    Replace `your-username/vanity-generator` with your desired Replicate model name.
    ```bash
    cog push r8.im/your-username/vanity-generator
    ```

When deployed to Replicate, the model will automatically attempt to use GPU acceleration if available in the environment.

## Testing with Node.js

See the `/test` directory for a Node.js script that demonstrates how to interact with the deployed model on Replicate. 