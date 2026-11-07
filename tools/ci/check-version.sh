#!/usr/bin/env bash
# Une seule version (P5.35) : celle de project() dans CMakeLists.txt, recopiée dans vcpkg.json.
set -euo pipefail
cd "$(dirname "$0")/../.."
cmake_version="$(sed -n '/^project(ajp/,/)/s/^ *VERSION \([0-9.]*\).*/\1/p' CMakeLists.txt)"
vcpkg_version="$(sed -n 's/^ *"version-semver": "\([0-9.]*\)".*/\1/p' vcpkg.json)"
if [[ -z "$cmake_version" || "$cmake_version" != "$vcpkg_version" ]]; then
  echo "version : CMakeLists.txt dit « $cmake_version », vcpkg.json « $vcpkg_version »" >&2
  exit 1
fi
echo "version $cmake_version"
