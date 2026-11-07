#!/usr/bin/env bash
# Le kit CUDA (nvcc, runtime, en-têtes) et TensorRT pour construire ajp_backend_tensorrt sous Linux,
# aux versions de tools/release/versions.env, dans le dossier donné. Le lancer, c'est accepter la
# licence du kit CUDA (https://docs.nvidia.com/cuda/eula/index.html) et celle de TensorRT
# (https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html). Rien de ce dossier
# n'entre dans un paquet : la CI compile et jette (P1.17), le paquet ne garde que notre module.
#
# Le kit vient des archives redist de NVIDIA, chacune vérifiée par l'empreinte de son manifeste,
# fusionnées en un seul dossier ; TensorRT, de la roue tensorrt-cu13-libs de pypi.nvidia.com
# (bibliothèques, empreinte épinglée) et des en-têtes de NVIDIA/TensorRT au commit épinglé. Avec
# --trtexec, trtexec en plus, sorti de l'archive Linux de TensorRT (empreinte épinglée, lue en flux
# sans être gardée sur le disque) : le paquet TensorRT le porte à côté du module (P1.19).
# Sous GitHub Actions, CUDACXX, CUDAToolkit_ROOT et AJP_TENSORRT_ROOT passent aux étapes suivantes.
#
#   tools/deps/linux-tensorrt.sh [--trtexec] ~/ajp-nvidia
set -euo pipefail

with_trtexec=0
if [ "${1:-}" = "--trtexec" ]; then with_trtexec=1; shift; fi
dir="${1:?usage : linux-tensorrt.sh [--trtexec] <dossier>}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=/dev/null
source "$root/tools/release/versions.env"
: "${CUDA_REDIST_SERIES:?}" "${TENSORRT_LIBS_VERSION:?}" "${TENSORRT_LIBS_SHA256:?}" "${TENSORRT_HEADERS_COMMIT:?}"

mkdir -p "$dir/downloads"
dir="$(cd "$dir" && pwd)"
downloads="$dir/downloads"

# --- Le kit CUDA : le dernier manifeste redistrib_<série>.<n>.json. ---
redist=https://developer.download.nvidia.com/compute/cuda/redist
cuda="$(curl -fsSL "$redist/" | grep -o "redistrib_${CUDA_REDIST_SERIES//./\\.}\.[0-9]*\.json" |
  sed 's/^redistrib_//; s/\.json$//' | sort -u -V | tail -1)"
[ -n "$cuda" ] || { echo "aucun manifeste CUDA $CUDA_REDIST_SERIES dans $redist" >&2; exit 1; }
manifest="$downloads/redistrib_$cuda.json"
curl -fsSL "$redist/redistrib_$cuda.json" -o "$manifest"
echo "CUDA $cuda (manifeste redistrib_$cuda.json)"

cuda_root="$dir/cuda"
rm -rf "${cuda_root:?}" && mkdir -p "$cuda_root"
# nvcc et son frontal, le runtime (libcudart_static.a, stubs/libcuda.so), CCCL, NVTX ; un composant
# absent d'une version est passé, un composant renommé garde ses deux noms (cuda_cccl puis cccl).
for names in cuda_nvcc cuda_crt libnvvm cuda_cudart "cuda_cccl cccl" cuda_nvtx; do
  line="$(python3 -I - "$manifest" $names <<'EOF'
import json, sys
manifest = json.load(open(sys.argv[1]))
for name in sys.argv[2:]:
    entry = manifest.get(name, {}).get("linux-x86_64")
    if entry:
        print(name, entry["relative_path"], entry["sha256"])
        break
EOF
)"
  if [ -z "$line" ]; then
    echo "  $names : absent de CUDA $cuda"
    continue
  fi
  read -r component path sha <<<"$line"
  file="$downloads/$(basename "$path")"
  [ -f "$file" ] || curl -fsSL "$redist/$path" -o "$file"
  got="$(sha256sum "$file" | cut -d' ' -f1)"
  [ "$got" = "$sha" ] || { rm -f "${file:?}"; echo "$component : empreinte $got, attendue $sha" >&2; exit 1; }
  tar -xf "$file" -C "$cuda_root" --strip-components=1
  echo "  $component"
done
[ -x "$cuda_root/bin/nvcc" ] || { echo "nvcc absent de $cuda_root" >&2; exit 1; }
# CMake (FindCUDAToolkit) cherche lib64 ; les archives redist posent lib.
[ -e "$cuda_root/lib64" ] || ln -s lib "$cuda_root/lib64"

# --- TensorRT : bibliothèques de la roue, en-têtes du dépôt de NVIDIA. ---
trt_root="$dir/tensorrt"
rm -rf "${trt_root:?}" && mkdir -p "$trt_root/lib" "$trt_root/include"
wheel="$downloads/tensorrt_cu13_libs-$TENSORRT_LIBS_VERSION-py3-none-manylinux_2_28_x86_64.whl"
curl -fsSL "https://pypi.nvidia.com/tensorrt-cu13-libs/$(basename "$wheel")" -o "$wheel"
got="$(sha256sum "$wheel" | cut -d' ' -f1)"
[ "$got" = "$TENSORRT_LIBS_SHA256" ] || { echo "$wheel : empreinte $got, attendue $TENSORRT_LIBS_SHA256" >&2; exit 1; }
# La roue fait près de 4 Go : seulement libnvinfer et libnvonnxparser en sortent, puis elle s'efface.
python3 -I - "$wheel" "$trt_root/lib" <<'EOF'
import os, re, shutil, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    for name in archive.namelist():
        base = os.path.basename(name)
        if re.fullmatch(r"lib(nvinfer|nvonnxparser)\.so\.\d+", base):
            with archive.open(name) as src, open(os.path.join(sys.argv[2], base), "wb") as dst:
                shutil.copyfileobj(src, dst)
            print("  " + base)
EOF
truncate -s 0 "$wheel"
# Le nom sans version que l'éditeur de liens cherche.
for lib in "$trt_root"/lib/lib*.so.*; do
  base="$(basename "$lib")"
  ln -sf "$base" "$trt_root/lib/${base%%.so.*}.so"
done
[ -e "$trt_root/lib/libnvinfer.so" ] || { echo "libnvinfer absent de $wheel" >&2; exit 1; }

headers="$downloads/tensorrt-headers"
rm -rf "${headers:?}"
git init -q "$headers"
git -C "$headers" fetch -q --depth 1 --filter=blob:none https://github.com/NVIDIA/TensorRT "$TENSORRT_HEADERS_COMMIT"
git -C "$headers" sparse-checkout set include
git -C "$headers" checkout -q FETCH_HEAD
[ "$(git -C "$headers" rev-parse HEAD)" = "$TENSORRT_HEADERS_COMMIT" ] || { echo "en-têtes : mauvais commit" >&2; exit 1; }
cp -r "$headers/include/." "$trt_root/include/"
grep -q "TRT_MINOR_ENTERPRISE $(echo "$TENSORRT_LIBS_VERSION" | cut -d. -f2)\b" "$trt_root/include/NvInferVersion.h" ||
  { echo "en-têtes et bibliothèques de TensorRT de versions différentes" >&2; exit 1; }

if (( with_trtexec )); then
  : "${TENSORRT_LINUX_URL:?}" "${TENSORRT_LINUX_SHA256:?}"
  # L'archive fait près de 4 Go : lue en flux, seul bin/trtexec en sort ; son empreinte est celle du
  # flux entier, comparée avant de garder trtexec.
  mkdir -p "$trt_root/bin" "$downloads/trtexec"
  rm -f "${downloads:?}/trt.fifo" && mkfifo "$downloads/trt.fifo"
  sha256sum <"$downloads/trt.fifo" | cut -d' ' -f1 >"$downloads/trt.sha" &
  hasher=$!
  curl -fsSL "$TENSORRT_LINUX_URL" | tee "$downloads/trt.fifo" | zstd -dc |
    tar -x -C "$downloads/trtexec" --wildcards '*/bin/trtexec'
  wait "$hasher"
  got="$(cat "$downloads/trt.sha")"
  rm -f "${downloads:?}/trt.fifo" "${downloads:?}/trt.sha"
  found="$(find "$downloads/trtexec" -path '*/bin/trtexec' -type f | head -1)"
  if [ "$got" != "$TENSORRT_LINUX_SHA256" ] || [ -z "$found" ]; then
    rm -rf "${downloads:?}/trtexec"
    echo "archive TensorRT Linux : empreinte $got, attendue $TENSORRT_LINUX_SHA256 (ou trtexec absent)" >&2
    exit 1
  fi
  install -m 755 "$found" "$trt_root/bin/trtexec"
  rm -rf "${downloads:?}/trtexec"
  echo "  trtexec"
fi

if [ -n "${GITHUB_ENV:-}" ]; then
  {
    echo "CUDACXX=$cuda_root/bin/nvcc"
    echo "CUDAToolkit_ROOT=$cuda_root"
    echo "AJP_TENSORRT_ROOT=$trt_root"
  } >>"$GITHUB_ENV"
fi
echo "CUDA : $cuda_root ; TensorRT : $trt_root"
