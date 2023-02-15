#![allow(dead_code)]

use std::{env, path::PathBuf};

#[cfg(not(feature = "disable-sys-build-script"))]
fn main() {
    let libort_install_dir = PathBuf::from(env::var("ORT_LIB_LOCATION").unwrap());
    let include_dir = libort_install_dir.join("include");
    let lib_dir = libort_install_dir.join("lib");
    let generated_file = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("bindings.rs");

    // Tell cargo to tell rustc to link onnxruntime shared library.
    println!("cargo:rustc-link-lib=onnxruntime");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=ORT_LIB_LOCATION");
    println!("cargo:rerun-if-changed={}", generated_file.display());

    #[cfg(feature = "cuda")]
    let bindings = bindgen::Builder::default()
        .header("wrapper_cuda.h")
        .clang_args(&[format!("-I{}", include_dir.display())])
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .size_t_is_usize(true)
        .rustfmt_bindings(true)
        .rustified_enum(".*")
        .generate()
        .unwrap();

    #[cfg(not(feature = "cuda"))]
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&[format!("-I{}", include_dir.display())])
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .size_t_is_usize(true)
        .rustfmt_bindings(true)
        .rustified_enum(".*")
        .generate()
        .unwrap();

    // Write the bindings to (source controlled) src/generated/bindings.rs
    bindings.write_to_file(generated_file).unwrap();
}
