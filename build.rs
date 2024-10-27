use std::{
    env, fs,
    path::{Path, PathBuf},
    println,
};

/// Link vcpkg package.
fn link_vcpkg(mut path: PathBuf, name: &str) -> PathBuf {
    let target = "x64-windows-static";
    println!("cargo:info={}", target);
    path.push("installed");
    path.push(target);
    println!(
        "{}",
        format!(
            "cargo:rustc-link-lib=static={}",
            name.trim_start_matches("lib")
        )
    );
    println!(
        "{}",
        format!(
            "cargo:rustc-link-search={}",
            path.join("lib").to_str().unwrap()
        )
    );
    let include = path.join("include");
    println!("{}", format!("cargo:include={}", include.to_str().unwrap()));
    include
}


fn find_package(name: &str) -> Vec<PathBuf> {
    if let Ok(vcpkg_root) = std::env::var("VCPKG_ROOT") {
        vec![link_vcpkg(vcpkg_root.into(), name)]
    } else {
        panic!("VCPKG_ROOT environment variable not set.");
    }
}

fn generate_bindings(
    ffi_header: &Path,
    include_paths: &[PathBuf],
    ffi_rs: &Path,
    exact_file: &Path,
    regex: &str,
) {
    let mut b = bindgen::builder()
        .header(ffi_header.to_str().unwrap())
        .allowlist_type(regex)
        .allowlist_var(regex)
        .allowlist_function(regex)
        .rustified_enum(regex)
        .trust_clang_mangling(false)
        .layout_tests(false) // breaks 32/64-bit compat
        .generate_comments(false); // comments have prefix /*!\

    for dir in include_paths {
        b = b.clang_arg(format!("-I{}", dir.display()));
    }

    b.generate().unwrap().write_to_file(ffi_rs).unwrap();
    fs::copy(ffi_rs, exact_file).ok(); // ignore failure
}

fn gen_vcpkg_package(package: &str, ffi_header: &str, generated: &str, regex: &str) {
    let includes = find_package(package);
    let src_dir = env::var_os("CARGO_MANIFEST_DIR").unwrap();
    let src_dir = Path::new(&src_dir);
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    let ffi_header = src_dir.join("src").join("bindings").join(ffi_header);
    println!("rerun-if-changed={}", ffi_header.display());
    for dir in &includes {
        println!("rerun-if-changed={}", dir.display());
    }

    let ffi_rs = out_dir.join(generated);
    let exact_file = src_dir.join("generated").join(generated);
    generate_bindings(&ffi_header, &includes, &ffi_rs, &exact_file, regex);
}



fn main() {
    //注意：所有x86（32位）的链接符号名称都以“_”为前缀。
    //运行“rustup show”显示当前默认的工具链，如果是stable-x86-pc-windows-msvc
    //请使用“rustup toolchain install stable-x86_64-pc-windows-msvc”安装x64工具链
    //通过“rustup default stable-x86_64-pc-windows-msvc”将x64设置为默认值
    let target = target_build_utils::TargetInfo::new();
    if target.unwrap().target_pointer_width() != "64" {
        panic!("Only support 64bit system");
    }
    env::remove_var("CARGO_CFG_TARGET_FEATURE");
    env::set_var("CARGO_CFG_TARGET_FEATURE", "crt-static");

    find_package("libyuv");
    gen_vcpkg_package("libvpx", "vpx_ffi.h", "vpx_ffi.rs", "^[vV].*");
    gen_vcpkg_package("aom", "aom_ffi.h", "aom_ffi.rs", "^(aom|AOM|OBU|AV1).*");
    gen_vcpkg_package("libyuv", "yuv_ffi.h", "yuv_ffi.rs", ".*");
    println!("cargo:rustc-cfg=dxgi");
}
