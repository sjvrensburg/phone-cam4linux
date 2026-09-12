fn main() {
    // ONNX Runtime's WebGPU provider lives in a separate `libwebgpu_dawn.so` that
    // `ort` copies next to the binary; look there first so a release tarball
    // (binary + models/) runs from any directory without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN");
}
