#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cuda.h>

// SHA-256 Constants
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// BS58 charset - Removed as it's unused
// const char BS58_CHARS[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 256
#define ATTEMPTS_PER_THREAD 1000

__device__ void sha256_transform(uint32_t state[8], const uint8_t data[64]) {
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
    
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];
    
    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void sha256(uint8_t hash[32], const uint8_t *data, size_t len) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint8_t block[64];
    uint32_t bitlen = len * 8;
    
    // Process full blocks
    for (size_t i = 0; i < len / 64; i++) {
        for (int j = 0; j < 64; j++) {
            block[j] = data[i * 64 + j];
        }
        sha256_transform(state, block);
    }
    
    // Process final block(s) with padding
    size_t remain = len % 64;
    for (size_t i = 0; i < remain; i++) {
        block[i] = data[len - remain + i];
    }
    
    // Append '1' bit
    block[remain] = 0x80;
    
    // Pad with zeros and add length (in bits) at the end
    for (size_t i = remain + 1; i < 64 - 8; i++) {
        block[i] = 0;
    }
    
    // Add bit length to the end of the block
    for (int i = 0; i < 8; i++) {
        block[64 - 8 + i] = (bitlen >> (8 * (7 - i))) & 0xff;
    }
    sha256_transform(state, block);
    
    // Copy the final state to the hash output
    for (int i = 0; i < 8; i++) {
        hash[i * 4] = (state[i] >> 24) & 0xff;
        hash[i * 4 + 1] = (state[i] >> 16) & 0xff;
        hash[i * 4 + 2] = (state[i] >> 8) & 0xff;
        hash[i * 4 + 3] = state[i] & 0xff;
    }
}

// BS58 encoding functions for verification
__device__ bool check_match(const uint8_t *hash, const uint8_t *target, uint32_t target_len, bool is_prefix, bool case_insensitive) {
    // This is a simplified BS58 encoding check
    // For a real implementation, you would need a full BS58 encoder
    
    // We'll focus on checking the first few bytes for a match
    // in a real implementation, this would do a proper BS58 encoding
    
    // Simple byte comparison for demonstration purposes
    bool match = true;
    for (uint32_t i = 0; i < target_len && match; i++) {
        uint8_t h = is_prefix ? hash[i] : hash[32 - target_len + i];
        uint8_t t = target[i];
        
        if (case_insensitive) {
            // Convert to lowercase for case-insensitive comparison
            if (h >= 'A' && h <= 'Z') h += 32;
            if (t >= 'A' && t <= 'Z') t += 32;
            
            // Exception for 'L' which shouldn't be converted
            if (h == 'l') h = 'L';
            if (t == 'l') t = 'L';
        }
        
        match = (h == t);
    }
    
    return match;
}

__global__ void vanity_kernel(
    uint8_t *out,
    const uint8_t *base,
    const uint8_t *owner,
    const uint8_t *seed_base,
    const uint8_t *target,
    uint32_t target_len,
    bool is_prefix,
    bool case_insensitive,
    int *d_found_flag
) {
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t attempts = 0;
    
    // Each thread has its own seed derived from the seed_base
    uint8_t seed[16];
    for (int i = 0; i < 16; i++) {
        seed[i] = seed_base[i] ^ (thread_idx & 0xFF) ^ ((thread_idx >> 8) & 0xFF);
    }
    
    // Buffer for SHA-256 input
    uint8_t buffer[128]; // base(32) + seed(16) + owner(32) = 80 bytes
    
    // Copy base and owner to buffer
    for (int i = 0; i < 32; i++) {
        buffer[i] = base[i];
        buffer[i + 48] = owner[i];
    }
    
    uint8_t hash[32];
    bool found = false;
    
    for (int iter = 0; iter < ATTEMPTS_PER_THREAD && atomicAdd(d_found_flag, 0) == 0; iter++) {
        // Update seed for this attempt
        for (int i = 0; i < 16; i++) {
            seed[i] = (seed[i] + iter) % 256;
            buffer[32 + i] = seed[i];
        }
        
        // Compute SHA-256 hash
        sha256(hash, buffer, 80);
        
        // Check if this matches our target
        if (check_match(hash, target, target_len, is_prefix, case_insensitive)) {
            // Attempt to claim the find
            int old_flag = atomicCAS(d_found_flag, 0, 1);
            
            if (old_flag == 0) { // Only write if we were the first
                // Copy the result to the output buffer if we found a match
                // Output format: [seed(16 bytes), attempts(8 bytes)]
                for (int i = 0; i < 16; i++) {
                    out[i] = seed[i];
                }
                
                // Store attempts count (little-endian)
                attempts = (uint64_t)iter + 1;
                for (int i = 0; i < 8; i++) {
                    out[16 + i] = (attempts >> (i * 8)) & 0xFF;
                }
            }
            // Even if we weren't first, break this thread's loop as a match was found globally
            break; 
        }
        
        // Note: attempts is local to thread, not incremented here anymore
        // The attempt count written is the local iter count when found.
    }
}

extern "C" {
    void vanity_round(
        const uint8_t *seed,
        const uint8_t *base,
        const uint8_t *owner,
        const uint8_t *target,
        uint64_t target_len,
        uint8_t *out,
        bool case_insensitive,
        bool is_prefix
    ) {
        // Setup device memory
        uint8_t *d_out = nullptr;
        uint8_t *d_base = nullptr;
        uint8_t *d_owner = nullptr;
        uint8_t *d_seed = nullptr;
        uint8_t *d_target = nullptr;
        int *d_found_flag = nullptr; // Pointer for atomic flag
        
        cudaMalloc(&d_out, 24); // 16 bytes for seed + 8 bytes for attempt count
        cudaMalloc(&d_base, 32);
        cudaMalloc(&d_owner, 32);
        cudaMalloc(&d_seed, 32);
        cudaMalloc(&d_target, target_len);
        cudaMalloc(&d_found_flag, sizeof(int)); // Allocate memory for the flag

        // Initialize d_out and d_found_flag to 0
        cudaMemset(d_out, 0, 24);
        cudaMemset(d_found_flag, 0, sizeof(int));
        
        cudaMemcpy(d_base, base, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_owner, owner, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, seed, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, target, target_len, cudaMemcpyHostToDevice);
        
        // Run kernel
        vanity_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_out,
            d_base,
            d_owner,
            d_seed,
            d_target,
            target_len,
            is_prefix,
            case_insensitive,
            d_found_flag
        );
        
        // Wait for kernel completion
        cudaDeviceSynchronize();

        // Copy results back
        cudaMemcpy(out, d_out, 24, cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_out);
        cudaFree(d_base);
        cudaFree(d_owner);
        cudaFree(d_seed);
        cudaFree(d_target);
        cudaFree(d_found_flag); // Free the flag memory
    }
} 