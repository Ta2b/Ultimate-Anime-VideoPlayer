#!/usr/bin/env bash
# Une seule version (P5.35) : celle de project() dans CMakeLists.txt, recopiée dans vcpkg.json.
#
#   tools/ci/check-version.sh            # CMakeLists.txt et vcpkg.json d'accord
#   tools/ci/check-version.sh v0.11.0    # et l'étiquette d'une release dit la même version (A28-04) :
#                                        # v<VERSION>, ou v<VERSION>-<suffixe> pour une préversion
set -euo pipefail
cd "$(dirname "$0")/../.."
cmake_version="$(sed -n '/^project(ajp/,/)/s/^ *VERSION \([0-9.]*\).*/\1/p' CMakeLists.txt)"
vcpkg_version="$(sed -n 's/^ *"version-semver": "\([0-9.]*\)".*/\1/p' vcpkg.json)"
if [[ -z "$cmake_version" || "$cmake_version" != "$vcpkg_version" ]]; then
  echo "version : CMakeLists.txt dit « $cmake_version », vcpkg.json « $vcpkg_version »" >&2
  exit 1
fi
if [ $# -ge 1 ]; then
  tag="$1"
  if [[ "$tag" != "v$cmake_version" && ! "$tag" =~ ^v${cmake_version//./\\.}-[0-9A-Za-z.]+$ ]]; then
    echo "::error::l'étiquette $tag ne dit pas la version de CMakeLists.txt ($cmake_version) : attendu v$cmake_version ou v$cmake_version-<suffixe>" >&2
    exit 1
  fi
fi
echo "version $cmake_version${1:+ (étiquette $1)}"
