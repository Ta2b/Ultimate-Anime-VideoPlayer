#!/usr/bin/env bash
# Essaie les scripts de publication sans réseau ni GPU, sur des fichiers fabriqués (B51-01, B51-02, B51-04, B51-07,
# B51-10, B51-16) : chacun doit refuser le défaut qu'il garde, et laisser passer le cas sain.
set -euo pipefail
R="$(cd "$(dirname "$0")/../release" && pwd)"
CI="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"
fail=0
expect() {  # attendu (0|1) description commande…
  local want=$1 what=$2; shift 2
  local got=0
  "$@" >/dev/null 2>&1 || got=1
  if [ "$got" != "$want" ]; then echo "ÉCHEC : $what (code attendu $want)" >&2; fail=1; else echo "ok : $what"; fi
}
engine() { printf '\n[[engine]]\nid = "tensorrt"\nplatform = "%s"\nurl = "https://github.com/o/r/releases/download/%s/m.tar.gz"\nsha256 = "%s"\nsize = %s\n\n[[vendor_file]]\nid = "tensorrt"\nplatform = "%s"\nurl = "https://pypi.nvidia.com/x"\nname = "n"\n' "$1" "$2" "$3" "$4" "$1"; }

# B51-01 : l'index du jour nomme les paquets embarqués ; un ancien index joint ne passe plus devant.
{ echo 'version = 1'; engine linux trt-NEW aaa 1; } > linux.lock
{ echo 'version = 1'; engine windows trt-win-NEW bbb 2; } > windows.lock
{ echo 'version = 1'; engine linux trt-OLD aaa 1; } > joint.lock
AJP_RELEASE_TAG=v1.0.0 "$R/nvidia-index.sh" index.lock linux.lock windows.lock >/dev/null
expect 0 "index d'après les paquets embarqués" "$R/check-nvidia-index.sh" index.lock linux.lock windows.lock
grep -qx 'tag = "v1.0.0"' index.lock && echo "ok : l'index porte son étiquette" || { echo "ÉCHEC : étiquette absente de l'index" >&2; fail=1; }
"$R/nvidia-index.sh" stale.lock joint.lock linux.lock windows.lock >/dev/null
expect 1 "un ancien index joint qui gagnerait est refusé" "$R/check-nvidia-index.sh" stale.lock linux.lock windows.lock

# B51-16 : la lecture par plage des roues de NVIDIA, sans réseau.
expect 0 "surveillance des roues de NVIDIA (essai)" python3 -I "$R/check-vendor.py" --selftest

# B51-10 : liste blanche des archives de moteur.
mkdir good evil
touch good/ajp_backend_tensorrt.so good/trtexec evil/ajp_backend_tensorrt.so evil/trtexec evil/libnvJitLink.so.13
tar -C good -czf good.tar.gz . && tar -C evil -czf evil.tar.gz .
expect 0 "archive de moteur : module et trtexec seulement" "$R/check-engine-archive.sh" good.tar.gz
expect 1 "archive de moteur : un fichier de plus est refusé" "$R/check-engine-archive.sh" evil.tar.gz

# B51-07 : les archives téléchargées sont celles de la construction.
mkdir rel && echo archive > rel/ajp-linux-x64.tar.gz
sum="$(sha256sum rel/ajp-linux-x64.tar.gz | cut -d' ' -f1)"
expect 0 "archive identique à la construction" "$R/check-built-sums.sh" rel "ajp-linux-x64.tar.gz=$sum" "ajp-macos-arm64.zip="
expect 1 "archive remplacée dans la release" "$R/check-built-sums.sh" rel "ajp-linux-x64.tar.gz=$(printf 'x%.0s' {1..64})"
expect 1 "archive non construite mais présente" "$R/check-built-sums.sh" rel "ajp-linux-x64.tar.gz="

# B51-02, B51-16 : l'archive de chaque moteur existe avec la taille et le SHA-256 de l'index.
mkdir bin
printf 'payload\n' > payload.tar.gz
psha="$(sha256sum payload.tar.gz | cut -d' ' -f1)" psize="$(stat -c %s payload.tar.gz)"
printf '#!/bin/bash\nout=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done; cp "%s/payload.tar.gz" "$out"\n' "$work" > bin/curl
chmod +x bin/curl
check_dir() {  # dossier d'une release factice
  mkdir -p "$1" "$1.stage/ajp-linux-x64"
  { echo 'version = 1'; engine linux trt-A "$2" "$3"; } > "$1/engines.lock"
  cp "$1/engines.lock" "$1.stage/ajp-linux-x64/engines.lock"
  tar -C "$1.stage" -czf "$1/ajp-linux-x64.tar.gz" ajp-linux-x64
  { echo 'version = 1'; engine windows trt-win-A "$2" "$3"; } > w.lock
  python3 -I -c "import sys, zipfile; z = zipfile.ZipFile(sys.argv[1], 'w'); z.write('w.lock', 'ajp/engines.lock')" "$1/ajp-windows-x64.zip"
}
check_dir good_rel "$psha" "$psize"
check_dir bad_rel "$(printf 'f%.0s' {1..64})" "$psize"
expect 0 "release complète, archives des moteurs vérifiées" env PATH="$work/bin:$PATH" AJP_RELEASE_REPO=o/r "$R/check-release.sh" v1.0.0 good_rel
expect 1 "archive de moteur réécrite après coup" env PATH="$work/bin:$PATH" AJP_RELEASE_REPO=o/r "$R/check-release.sh" v1.0.0 bad_rel
# C51-01 : l'engines.lock joint à part n'est pas cru ; celui de l'archive vérifiée prime, et une différence est refusée.
cp -r good_rel swapped_rel
{ echo 'version = 1'; engine linux trt-EVIL "$psha" "$psize"; } > swapped_rel/engines.lock
expect 1 "engines.lock Linux remplacé dans la release" env PATH="$work/bin:$PATH" AJP_RELEASE_REPO=o/r "$R/check-release.sh" v1.0.0 swapped_rel
sed -i '/^\[\[vendor_file\]\]/,$d' good_rel/engines.lock
cp good_rel/engines.lock good_rel.stage/ajp-linux-x64/engines.lock
tar -C good_rel.stage -czf good_rel/ajp-linux-x64.tar.gz ajp-linux-x64
python3 -I -c "import zipfile; z = zipfile.ZipFile('good_rel/ajp-windows-x64.zip', 'w'); z.writestr('ajp/engines.lock', 'version = 1\n')"
expect 1 "release finale sans [[vendor_file]] ni TensorRT Windows" env PATH="$work/bin:$PATH" AJP_RELEASE_REPO=o/r "$R/check-release.sh" v1.0.0 good_rel

# C51-04 : pas de release finale sans clé de secours compilée.
printf 'set(AJP_UPDATE_PUBKEY "%s" CACHE STRING "")\nset(AJP_UPDATE_PUBKEY_BACKUP "" CACHE STRING "")\n' "$(printf 'a%.0s' {1..64})" > cmake-vide.txt
printf 'set(AJP_UPDATE_PUBKEY "%s" CACHE STRING "")\nset(AJP_UPDATE_PUBKEY_BACKUP "%s" CACHE STRING "")\n' "$(printf 'a%.0s' {1..64})" "$(printf 'b%.0s' {1..64})" > cmake-secours.txt
printf 'set(AJP_UPDATE_PUBKEY "%s" CACHE STRING "")\nset(AJP_UPDATE_PUBKEY_BACKUP "%s" CACHE STRING "")\n' "$(printf 'a%.0s' {1..64})" "$(printf 'A%.0s' {1..64})" > cmake-meme.txt
expect 1 "release finale sans clé de secours" "$CI/check-backup-key.sh" v1.0.0 cmake-vide.txt
expect 0 "préversion sans clé de secours" "$CI/check-backup-key.sh" v1.0.0-rc.1 cmake-vide.txt
expect 0 "release finale avec clé de secours" "$CI/check-backup-key.sh" v1.0.0 cmake-secours.txt
expect 1 "clé de secours égale à la clé active" "$CI/check-backup-key.sh" v1.0.0 cmake-meme.txt
exit "$fail"
