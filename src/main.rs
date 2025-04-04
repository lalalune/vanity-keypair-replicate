use std::pin::Pin;
use std::future::Future;
use anyhow::Result;
use cog_rust::Cog;
use schemars::JsonSchema;
use sha2::{Digest, Sha256};
use rand::{distributions::Alphanumeric, Rng};
use serde::{Deserialize, Serialize};
use std::{array, sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}};
use rayon::prelude::*;
use std::time::Instant;
use async_trait::async_trait;

#[derive(Deserialize, JsonSchema)]
pub struct VanityRequest {
	/// Base public key in base58 format
	base: String,
	
	/// Owner public key in base58 format 
	owner: String,
	
	/// Target prefix/suffix for the generated address
	target: String,
	
	/// Whether to match case-insensitively
	case_insensitive: String,
	
	/// Number of results to generate (default: 1)
	num_results: String,
	
	/// Target type: prefix or suffix
	#[serde(default)]
	target_type: TargetType,
}

#[derive(Deserialize, JsonSchema, Clone, Copy, Debug)]
#[serde(rename_all = "lowercase")]
enum TargetType {
	Prefix,
	Suffix,
}

impl Default for TargetType {
	fn default() -> Self {
		TargetType::Prefix
	}
}

fn default_num_results() -> usize {
	1
}

#[derive(Serialize, Debug, JsonSchema)]
struct VanityResult {
	pubkey: String,
	seed: String,
	attempts: u64,
	time_secs: f64,
}

#[derive(Serialize, Debug, JsonSchema)]
pub struct VanityResponse {
	results: Vec<VanityResult>,
}

struct VanityModel;

#[async_trait]
impl Cog for VanityModel {
	type Request = VanityRequest;
	type Response = VanityResponse;

	async fn setup() -> Result<Self> {
		// Initialize rayon thread pool for parallel processing
		rayon::ThreadPoolBuilder::new().build_global()?;
		Ok(Self)
	}

	fn predict(&self, input: Self::Request) -> Result<Self::Response> {
		// Parse base and owner keys
		let base = parse_pubkey(&input.base)?;
		let owner = parse_pubkey(&input.owner)?;
		
		// Validate target
		let target = get_validated_target(&input.target, input.case_insensitive.parse::<bool>()?);
		
		// parse case_insensitive from string to bool
		let case_insensitive = input.case_insensitive.parse::<bool>()?;

		// parse num_results from string to usize
		let num_results = input.num_results.parse::<usize>()?;
		
		// Track results
		let mut results = Vec::with_capacity(num_results);

		// For each requested result
		for _ in 0..num_results {
			// Generate a vanity address
			let result = grind_single_vanity(
				&base,
				&owner,
				&target,
				case_insensitive,
				input.target_type,
			)?;
			
			results.push(result);
		}
		
		Ok(VanityResponse { results })
	}
}

fn grind_single_vanity(
	base: &[u8; 32],
	owner: &[u8; 32],
	target: &str,
	case_insensitive: bool,
	target_type: TargetType,
) -> Result<VanityResult> {
	#[cfg(feature = "cuda")]
	{
		// Try GPU first if CUDA is available
		match grind_gpu(base, owner, target, case_insensitive, target_type) {
			Ok(result) => return Ok(result),
			Err(e) => {
				eprintln!("GPU grinding failed: {}, falling back to CPU", e);
				// Fall back to CPU if GPU fails
			}
		}
	}
	
	// Always available CPU implementation
	grind_cpu(base, owner, target, case_insensitive, target_type)
}

#[cfg(feature = "cuda")]
fn grind_gpu(
	base: &[u8; 32],
	owner: &[u8; 32],
	target: &str,
	case_insensitive: bool,
	target_type: TargetType,
) -> Result<VanityResult> {
	use std::time::Instant;
	
	let start_time = Instant::now();
	
	// Generate a random seed for the GPU
	let seed = new_gpu_seed(0, 0);
	
	// Target type handling - convert to bytes
	let target_bytes = target.as_bytes();
	
	// Prepare output buffer: 16 bytes for seed + 8 bytes for count
	let mut out = [0u8; 24];
	
	// Call the GPU implementation
	unsafe {
		// Adjust the is_prefix parameter based on target_type
		let is_prefix = match target_type {
			TargetType::Prefix => true,
			TargetType::Suffix => false,
		};
		
		vanity_round(
			0, // gpu index - only one GPU in Replicate
			seed.as_ptr(),
			base.as_ptr(),
			owner.as_ptr(),
			target_bytes.as_ptr(),
			target_bytes.len() as u64,
			out.as_mut_ptr(),
			case_insensitive,
		);
	}
	
	// Extract the seed and count from the output
	let seed_bytes = &out[0..16];
	let count_bytes = &out[16..24];
	
	// Compute the pubkey from the returned seed
	let pubkey_bytes: [u8; 32] = Sha256::new()
		.chain_update(base)
		.chain_update(seed_bytes)
		.chain_update(owner)
		.finalize()
		.into();
	
	let pubkey = bs58::encode(pubkey_bytes).into_string();
	let seed_str = core::str::from_utf8(seed_bytes).unwrap_or("invalid-utf8").to_string();
	let attempts = u64::from_le_bytes(array::from_fn(|i| count_bytes[i]));
	
	Ok(VanityResult {
		pubkey,
		seed: seed_str,
		attempts,
		time_secs: start_time.elapsed().as_secs_f64(),
	})
}

#[cfg(feature = "cuda")]
fn new_gpu_seed(gpu_id: u32, iteration: u64) -> [u8; 32] {
	Sha256::new()
		.chain_update(rand::random::<[u8; 32]>())
		.chain_update(gpu_id.to_le_bytes())
		.chain_update(iteration.to_le_bytes())
		.finalize()
		.into()
}

fn grind_cpu(
	base: &[u8; 32],
	owner: &[u8; 32],
	target: &str,
	case_insensitive: bool,
	target_type: TargetType,
) -> Result<VanityResult> {
	let start_time = Instant::now();
	
	let exit = Arc::new(AtomicBool::new(false));
	let exit_clone = exit.clone();
	
	// Use rayon to parallelize the search
	let result = Arc::new(Mutex::new(None));
	let result_clone = result.clone();
	
	// Get the number of threads to use - on Mac, use all available cores
	let num_threads = rayon::current_num_threads();
	println!("Using {} CPU threads for grinding", num_threads);
	
	(0..num_threads).into_par_iter().for_each(|_| {
		let mut local_count = 0_u64;
		
		loop {
			if exit_clone.load(Ordering::Relaxed) {
				return;
			}
			
			// Generate a random seed
			let mut seed_iter = rand::thread_rng().sample_iter(&Alphanumeric).take(16);
			let seed: [u8; 16] = array::from_fn(|_| seed_iter.next().unwrap());
			
			// Hash to generate the pubkey
			let pubkey_bytes: [u8; 32] = Sha256::new()
				.chain_update(base)
				.chain_update(&seed)
				.chain_update(owner)
				.finalize()
				.into();
			
			let pubkey = bs58::encode(pubkey_bytes).into_string();
			let check_str = if case_insensitive {
				maybe_bs58_aware_lowercase(&pubkey)
			} else {
				pubkey.clone()
			};
			
			local_count += 1;
			
			// Check if this matches our target
			let matches = match target_type {
				TargetType::Prefix => check_str.starts_with(target),
				TargetType::Suffix => check_str.ends_with(target),
			};
			
			if matches {
				let time_secs = start_time.elapsed().as_secs_f64();
				let seed_str = core::str::from_utf8(&seed).unwrap_or("invalid-utf8").to_string();
				
				// Store the result and signal other threads to exit
				let mut r = result_clone.lock().unwrap();
				*r = Some(VanityResult {
					pubkey,
					seed: seed_str,
					attempts: local_count,
					time_secs,
				});
				
				exit_clone.store(true, Ordering::Relaxed);
				return;
			}
			
			// Periodically check if we should yield
			if local_count % 1000 == 0 {
				std::thread::yield_now();
			}
		}
	});
	
	// Retrieve the result
	let result = result.lock().unwrap().take().ok_or_else(|| anyhow::anyhow!("Failed to find a matching address"))?;
	Ok(result)
}

// Helper functions for key handling and target validation
fn parse_pubkey(input: &str) -> Result<[u8; 32]> {
	if input.len() < 32 {
		return Err(anyhow::anyhow!("Public key too short"));
	}
	
	match bs58::decode(input).into_vec() {
		Ok(bytes) => {
			if bytes.len() != 32 {
				Err(anyhow::anyhow!("Invalid pubkey length: {}", bytes.len()))
			} else {
				let mut array = [0u8; 32];
				array.copy_from_slice(&bytes);
				Ok(array)
			}
		},
		Err(e) => Err(anyhow::anyhow!("Failed to decode pubkey: {}", e)),
	}
}

fn get_validated_target(target: &str, case_insensitive: bool) -> String {
	// Static string of BS58 characters
	const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

	// Validate target (i.e. does it include 0, O, I, l)
	for c in target.chars() {
		if !BS58_CHARS.contains(c) {
			panic!("Invalid character in target: {}", c);
		}
	}

	// Return bs58-aware lowercase if needed
	if case_insensitive {
		maybe_bs58_aware_lowercase(target)
	} else {
		target.to_string()
	}
}

fn maybe_bs58_aware_lowercase(target: &str) -> String {
	// L is only char that shouldn't be converted to lowercase in case-insensitivity case
	const LOWERCASE_EXCEPTIONS: &str = "L";

	target
		.chars()
		.map(|c| {
			if LOWERCASE_EXCEPTIONS.contains(c) {
				c
			} else {
				c.to_ascii_lowercase()
			}
		})
		.collect::<String>()
}

#[cfg(feature = "cuda")]
extern "C" {
	fn vanity_round(
		gpus: u32,
		seed: *const u8,
		base: *const u8,
		owner: *const u8,
		target: *const u8,
		target_len: u64,
		out: *mut u8,
		case_insensitive: bool,
	);
}

cog_rust::start!(VanityModel);
