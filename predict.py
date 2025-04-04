#!/usr/bin/env python
from typing import List, Dict, Any
import subprocess
import json
import os
import time
from loguru import logger
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """
        Compile the Rust code with CUDA support for GPU acceleration
        """
        logger.info("Building Rust binary with CUDA support...")
        # Build with CUDA feature
        os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:{os.environ['PATH']}"
        
        # Check that Rust is installed properly
        try:
            subprocess.run(["rustc", "--version"], check=True, capture_output=True)
            logger.info("Rust is installed correctly")
        except Exception as e:
            logger.error(f"Rust installation issue: {e}")
            
        # Build with CUDA feature
        try:
            subprocess.run(
                ["cargo", "build", "--release", "--features", "cuda"],
                check=True,
                capture_output=True
            )
            logger.info("Rust binary built successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error building Rust binary: {e}")
            logger.error(f"STDOUT: {e.stdout.decode('utf-8')}")
            logger.error(f"STDERR: {e.stderr.decode('utf-8')}")
            raise e

    def predict(
        self, 
        base: str = Input(description="Base public key in base58 format", default="11111111111111111111111111111111"),
        owner: str = Input(description="Owner public key in base58 format", default="BPFLoaderUpgradeab1e11111111111111111111111"),
        target: str = Input(description="Target prefix/suffix for the generated address", default="auto"),
        case_insensitive: bool = Input(description="Whether to match case-insensitively", default=True),
        num_results: int = Input(description="Number of results to generate", default=1, ge=1, le=100),
        target_type: str = Input(description="Target type: 'prefix' or 'suffix'", default="prefix", choices=["prefix", "suffix"])
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate Solana vanity address(es) with the specified parameters
        """
        logger.info(f"Generating vanity address with target: {target}, type: {target_type}")
        
        # Prepare command arguments for CLI
        cmd = [
            "./target/release/vanity-keypairs",
            "predict",
            "--base", base,
            "--owner", owner,
            "--target", target,
            "--case-insensitive", str(case_insensitive).lower(),
            "--num-results", str(num_results),
            "--target-type", target_type
        ]
        
        # Run the Rust binary
        start_time = time.time()
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run the process with CLI arguments
            process = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=True
            )
            
            # Parse the JSON output
            output = process.stdout
            elapsed = time.time() - start_time
            logger.info(f"Vanity address generation completed in {elapsed:.2f} seconds")
            
            # The output should already be JSON, but we validate it
            try:
                result = json.loads(output)
                return result
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON output: {output}")
                return {"results": []}
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running Rust binary: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return {"results": []} 