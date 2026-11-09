#!/usr/bin/env bash
# Le banc de qualité en une commande, sans carte graphique : télécharge les modèles livrés s'ils manquent
# (tools/release/fetch-models.sh, vérifiés par SHA-256), puis mesure chaque réseau contre sa référence FP32 et
# contre la vraie image (tools/bench/qualite/banc.py, docs/banc-qualite.md).
#
#   tools/bench/qualite.sh [options de banc.py]
#   tools/bench/qualite.sh --video episode.mkv --md rapport.md
#
# Il faut python3 avec numpy, onnx et onnxruntime, ffmpeg, et 7z pour le premier téléchargement.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/ajp-release-models"
MODELS="$CACHE/modeles"
if [ ! -f "$MODELS/rife/distil_drba_v2_lite.onnx" ] || [ ! -f "$MODELS/rife/rife_v4.25.onnx" ]; then
  "$ROOT/tools/release/fetch-models.sh" "$MODELS" "$CACHE"
fi
exec python3 -I "$ROOT/tools/bench/qualite/banc.py" --modeles "$MODELS" "$@"
