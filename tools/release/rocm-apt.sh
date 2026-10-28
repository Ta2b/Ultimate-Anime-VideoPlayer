#!/usr/bin/env bash
# Le dépôt apt de ROCm (noble) à la version de tools/release/versions.env, épinglé devant Ubuntu :
# release-linux.yml et l'image du préflight installent ensuite migraphx-dev et hip-dev.
#
#   tools/release/rocm-apt.sh           ajoute le dépôt et met apt à jour
#   tools/release/rocm-apt.sh size      affiche la taille du téléchargement et de l'installation
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=versions.env
. "$here/versions.env"

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' | sudo tee /etc/apt/preferences.d/rocm-pin-600 >/dev/null
sudo apt-get update -qq
if [ "${1:-}" = size ]; then
  sudo apt-get install --assume-no --no-install-recommends migraphx-dev hip-dev \
    | grep -E "^(Need to get|After this operation)" || true
fi
