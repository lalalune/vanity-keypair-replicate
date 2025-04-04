fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
        
        cc::Build::new()
            .cuda(true)
            .flag("-gencode")
            .flag("arch=compute_75,code=sm_75")
            .flag("-gencode")
            .flag("arch=compute_80,code=sm_80")
            .flag("-gencode")
            .flag("arch=compute_86,code=sm_86")
            .file("src/cuda/vanity.cu")
            .compile("vanity_cuda");
            
        println!("cargo:rerun-if-changed=src/cuda/vanity.cu");
    }
} 