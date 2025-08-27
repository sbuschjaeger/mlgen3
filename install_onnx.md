# Install ONNX Runtime (C/C++) on Linux — Step‑by‑Step (Command Line)

This guide shows a clean, repeatable CLI install of **ONNX Runtime** for **C and C++**, with both **CPU** and **CUDA GPU** options, plus minimal C/C++ samples and CMake.

---

## Plan (pseudocode)
1. Detect distro & install build tools: compiler, CMake, `curl`, `jq`, `tar`.
2. Choose flavor: `cpu` (default) or `cuda` (GPU). Detect `x86_64`/`aarch64`.
3. Fetch latest ONNX Runtime release from GitHub; pick the correct Linux archive by flavor & arch.
4. Extract to `$HOME/.local/onnxruntime/<version>/<flavor>`; create `current` symlink.
5. Export `ORT_HOME` & update `LD_LIBRARY_PATH`. Best to add this to `~/.bashrc`
6. Build & run tiny C and C++ programs that print the ONNX Runtime version; verify linking at runtime.
7. (Optional) Build with CMake; set rpath for portable binaries.
8. Uninstall by removing the folder/symlink.

---

## Quickstart (CPU, x86_64)
> One‑liners; see the full script below for robust handling and GPU.

```bash
# 1) Prereqs (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y build-essential cmake curl jq

# 2) Get latest ONNX Runtime CPU (x64) and install under ~/.local/onnxruntime
LATEST=$(curl -s https://api.github.com/repos/microsoft/onnxruntime/releases/latest | jq -r '.tag_name')
ASSET_URL=$(curl -s https://api.github.com/repos/microsoft/onnxruntime/releases/latest \
  | jq -r '.assets[].browser_download_url | select(test("onnxruntime-linux-x64-[0-9].*\\.tgz$"))' | head -n1)
mkdir -p "$HOME/.local/onnxruntime" && cd "$HOME/.local/onnxruntime"
curl -L "$ASSET_URL" -o ort.tgz && tar -xzf ort.tgz && rm ort.tgz
# The tarball usually unpacks as onnxruntime-linux-x64-<ver>
VER_DIR=$(find . -maxdepth 1 -type d -name 'onnxruntime-linux-*' | head -n1)
ln -snf "$VER_DIR" current
# 3) Env vars
cat > "$HOME/.local/onnxruntime/activate_ort.sh" << 'EOF'
export ORT_HOME="$HOME/.local/onnxruntime/current"
export LD_LIBRARY_PATH="$ORT_HOME/lib:${LD_LIBRARY_PATH:-}"
EOF
. "$HOME/.local/onnxruntime/activate_ort.sh"

# 4) Verify
ls "$ORT_HOME/include" | grep onnxruntime_c
ls "$ORT_HOME/lib" | grep onnxruntime
```

---

## Full, repeatable setup (script + samples + CMake)
> Copy the whole block to a file, or run chunks as needed.
> All files should be created under ~/.local/onnxruntime
> After installation and activation, run the commands from ~ HOME

```bash
# =============================
# file: scripts/install_onnxruntime.sh
# =============================
#!/usr/bin/env bash
set -euo pipefail

# --- Config (edit as needed) ---
FLAVOR="cpu"        # "cpu" or "cuda"
INSTALL_ROOT="$HOME/.local/onnxruntime"
GITHUB_API="https://api.github.com/repos/microsoft/onnxruntime/releases/latest"
# You may pin a specific tag (e.g. v1.18.1) by exporting ORT_TAG before running.
: "${ORT_TAG:=}"

# --- Helpers ---
msg() { printf "\033[1;32m[+]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[!]\033[0m %s\n" "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { err "Missing '$1'"; return 1; }; }

# Detect package manager
pm=""
if command -v apt-get >/dev/null 2>&1; then pm="apt"; fi
if command -v dnf >/dev/null 2>&1; then pm="dnf"; fi
if command -v yum >/dev/null 2>&1 && [ -z "$pm" ]; then pm="yum"; fi
if command -v pacman >/dev/null 2>&1; then pm="pacman"; fi

msg "Installing prerequisites (${pm:-manual})"
case "$pm" in
  apt)
    sudo apt-get update
    sudo apt-get install -y build-essential cmake curl jq tar ca-certificates
    ;;
  dnf)
    sudo dnf install -y gcc gcc-c++ make cmake curl jq tar ca-certificates
    ;;
  yum)
    sudo yum install -y gcc gcc-c++ make cmake curl jq tar ca-certificates
    ;;
  pacman)
    sudo pacman -Sy --noconfirm base-devel cmake curl jq tar ca-certificates
    ;;
  *)
    need gcc && need g++ && need make && need cmake && need curl && need jq && need tar || {
      err "Install the listed tools manually, then re-run."; exit 1; }
    ;;
esac

# Arch mapping for asset names
uname_m=$(uname -m)
case "$uname_m" in
  x86_64) ARCH="x64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *) err "Unsupported arch: $uname_m"; exit 1 ;;
esac

msg "Arch: $uname_m -> $ARCH | Flavor: $FLAVOR"

# Pick release tag and asset URL
if [ -n "$ORT_TAG" ]; then
  TAG="$ORT_TAG"
  API_URL="https://api.github.com/repos/microsoft/onnxruntime/releases/tags/${TAG}"
else
  TAG=$(curl -s "$GITHUB_API" | jq -r '.tag_name')
  API_URL="$GITHUB_API"
fi
[ -n "$TAG" ] || { err "Failed to resolve release tag"; exit 1; }
msg "Using release: $TAG"

# Choose asset pattern by flavor
if [ "$FLAVOR" = "cpu" ]; then
  PATTERN="onnxruntime-linux-${ARCH}-[0-9].*\\.tgz$"
else
  # CUDA builds often include cuda version in the name (e.g., -cuda12-)
  PATTERN="onnxruntime-linux-${ARCH}-cuda[0-9.\-]*[0-9]-[0-9].*\\.tgz$|onnxruntime-linux-${ARCH}-cuda[0-9.]*-[0-9].*\\.tgz$"
fi

ASSET_URL=$(curl -s "$API_URL" \
  | jq -r --arg re "$PATTERN" '.assets[].browser_download_url | select(test($re))' \
  | head -n1)

if [ -z "$ASSET_URL" ]; then
  err "No asset matched pattern for ${FLAVOR}/${ARCH}. Check available assets on GitHub."; exit 1
fi
msg "Asset URL: $ASSET_URL"

# Download & install
mkdir -p "$INSTALL_ROOT" && cd "$INSTALL_ROOT"
rm -f ort.tgz
msg "Downloading…"
curl -L "$ASSET_URL" -o ort.tgz
msg "Extracting…"
tar -xzf ort.tgz && rm -f ort.tgz
VER_DIR=$(find . -maxdepth 1 -type d -name "onnxruntime-linux-*" | sort | tail -n1)
[ -d "$VER_DIR" ] || { err "Extracted dir not found"; exit 1; }
ln -snf "$VER_DIR" current

# Activate script
cat > "$INSTALL_ROOT/activate_ort.sh" << 'EOF'
# source this file to use ONNX Runtime in this shell
export ORT_HOME="$HOME/.local/onnxruntime/current"
export LD_LIBRARY_PATH="$ORT_HOME/lib:${LD_LIBRARY_PATH:-}"
EOF

msg "Installed to: $INSTALL_ROOT/$VER_DIR"
msg "Usage: source $INSTALL_ROOT/activate_ort.sh"

# Basic sanity checks
. "$INSTALL_ROOT/activate_ort.sh"
[ -f "$ORT_HOME/lib/libonnxruntime.so" ] || { err "libonnxruntime.so not found"; exit 1; }
msg "libonnxruntime located. Setup complete."

# =============================
# file: scripts/activate_onnxruntime.sh (optional standalone)
# =============================
# source this file after editing ORT_HOME if you prefer a fixed path
# export ORT_HOME="$HOME/.local/onnxruntime/current"
# export LD_LIBRARY_PATH="$ORT_HOME/lib:${LD_LIBRARY_PATH:-}"

# =============================
# file: examples/c/min.c
# =============================
cat > /dev/null <<'C_EOF'
#include <onnxruntime_c_api.h>
#include <stdio.h>

int main(void) {
    const OrtApiBase* base = OrtGetApiBase();
    if (!base) { fprintf(stderr, "Failed to get OrtApiBase\n"); return 1; }
    printf("ORT C API version: %s\n", base->GetVersionString());
    // Why: creating an Env validates dynamic linking succeeds
    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    OrtEnv* env = NULL;
    if (api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "c_min", &env) != 0) {
        fprintf(stderr, "CreateEnv failed\n");
        return 2;
    }
    api->ReleaseEnv(env);
    return 0;
}
C_EOF

# =============================
# file: examples/cpp/min.cpp
# =============================
cat > /dev/null <<'CPP_EOF'
#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    std::cout << "ORT C API version: " << OrtGetApiBase()->GetVersionString() << "\n";
    // Why: Env ensures C++ wrapper links and loads provider libs
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "cpp_min"};
    return 0;
}
CPP_EOF

# =============================
# file: CMakeLists.txt
# =============================
cat > /dev/null <<'CMAKE_EOF'
cmake_minimum_required(VERSION 3.15)
project(ort_min C CXX)
set(CMAKE_CXX_STANDARD 17)

# Expect ORT_HOME in the environment or -DORT_HOME=/path
if(NOT DEFINED ORT_HOME)
  if(DEFINED ENV{ORT_HOME})
    set(ORT_HOME $ENV{ORT_HOME})
  else()
    message(FATAL_ERROR "Set ORT_HOME to the ONNX Runtime install root (contains include/ and lib/)")
  endif()
endif()

# Imported target for libonnxruntime.so
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ORT_HOME}/lib/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_HOME}/include"
)

add_executable(min_c examples/c/min.c)
add_executable(min_cpp examples/cpp/min.cpp)

target_link_libraries(min_c PRIVATE onnxruntime)
target_link_libraries(min_cpp PRIVATE onnxruntime)

# Why: embed rpath so binaries run without setting LD_LIBRARY_PATH
set_target_properties(min_c min_cpp PROPERTIES BUILD_RPATH "${ORT_HOME}/lib")
CMAKE_EOF

# =============================
# Build commands (after installation & activation)
# =============================
# Source env for this shell (best to add to .bashrc)
. "$HOME/.local/onnxruntime/activate_ort.sh"

# Direct compile (no CMake)
mkdir -p bin
cc examples/c/min.c -I"$ORT_HOME/include" -L"$ORT_HOME/lib" -lonnxruntime -Wl,-rpath,"$ORT_HOME/lib" -o bin/min_c
c++ examples/cpp/min.cpp -std=c++17 -I"$ORT_HOME/include" -L"$ORT_HOME/lib" -lonnxruntime -Wl,-rpath,"$ORT_HOME/lib" -o bin/min_cpp

# With CMake
cmake -S . -B build -DORT_HOME="$ORT_HOME"
cmake --build build --config Release --parallel

# Run
./bin/min_c || ./build/min_c
./bin/min_cpp || ./build/min_cpp

# =============================
# Optional: GPU (CUDA) notes
# =============================
# 1) Ensure a working NVIDIA driver & CUDA matching the ONNX Runtime CUDA build.
# 2) Reinstall with FLAVOR=cuda
#      env FLAVOR=cuda bash scripts/install_onnxruntime.sh
# 3) When running, provider libraries (e.g., libonnxruntime_providers_cuda.so) must be discoverable via rpath or LD_LIBRARY_PATH (activation script already handles $ORT_HOME/lib).

# =============================
# Uninstall
# =============================
# Remove the install and symlink (adjust if pinned)
rm -rf "$HOME/.local/onnxruntime/current" "$HOME/.local/onnxruntime/onnxruntime-linux-*"
```

---

## Usage in your project (summary)
> First run the source ` . "$HOME/.local/onnxruntime/activate_ort.sh" `
> or, separately run
> `export ORT_HOME="$HOME/.local/onnxruntime/current"`
> `export LD_LIBRARY_PATH="$ORT_HOME/lib:${LD_LIBRARY_PATH:-}"`

- **Include paths**: `-I$(ORT_HOME)/include`
- **Linker**: `-L$(ORT_HOME)/lib -lonnxruntime`
- **Runtime search path**: embed `rpath=$(ORT_HOME)/lib` or export `LD_LIBRARY_PATH`.
- **Headers**: `onnxruntime_c_api.h` (C) and `onnxruntime_cxx_api.h` (C++ wrapper).

---

## Troubleshooting
- *`undefined reference` when linking*: ensure `-lonnxruntime` appears **after** your object files.
- *`error while loading shared libraries: libonnxruntime.so: cannot open shared object file`*: set `LD_LIBRARY_PATH="$ORT_HOME/lib"` or embed rpath.
- *Old glibc on legacy distros*: consider building from source to match your system toolchain.
- *CUDA provider not found*: verify CUDA toolkit & driver versions compatible with the downloaded ONNX Runtime build; prefer matching the `cudaXX` suffix in the asset name.

---

## Build from source (fallback)
```bash
sudo apt-get update && sudo apt-get install -y git python3 python3-pip cmake build-essential
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
# CPU shared lib
./build.sh --config Release --build_shared_lib --parallel
# CUDA example (ensure CUDA installed)
# ./build.sh --config Release --use_cuda --cuda_version=12.2 --cuda_home=/usr/local/cuda --cudnn_home=/usr/lib/x86_64-linux-gnu --build_shared_lib --parallel
# After build, headers are under include/, libs under build/Linux/Release/
```

