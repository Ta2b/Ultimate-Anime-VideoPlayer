# Les moteurs TensorRT des bancs de GMFlow natif (gmflow_mlp_banc.sh, gmflow_attention_banc.sh), à sourcer.
#
# Deux recettes, telles que le lecteur construit ses moteurs (prepare/trtexec_command.cpp, prepare/builder.cpp) :
#  - lecteur : celle de la paire GMFSS (gpu::ModelSpec::tuned faux, app/play/chain_gmfss.cpp) : --stronglyTyped,
#    niveau d'optimisation par défaut (3), sans pavage L2 (--tilingOptimizationLevel=0) ;
#  - reglee : celle des modèles réglés (Recipe::Fp32IslandsTuned et le pavage du lecteur) : niveau 5, pavage 3.
# Sans cuDNN ni cuBLAS tant que trtexec prend --tacticSources (TensorRT 10 ; le lecteur le retire sous 11). Le
# natif se compare au meilleur des deux : TensorRT ne s'enlève pas là où il suffisait de le régler.
#
# Chaque moteur se garde dans le dossier de travail : relancer un banc ne reconstruit rien.

TRTEXEC="${TRTEXEC:-$(command -v trtexec || echo "$HOME/kirari-sdk/tensorrt/bin/trtexec")}"
export LD_LIBRARY_PATH="${TRTEXEC%/bin/trtexec}/lib:${LD_LIBRARY_PATH:-}"
# L'aide entière dans une variable : un « trtexec --help | grep -q » perd la course au SIGPIPE sous pipefail.
TRT_HELP="$("$TRTEXEC" --help 2>&1 || true)"
TACTICS=""
case "$TRT_HELP" in *tacticSources*) TACTICS="--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT" ;; esac
RECIPES=(lecteur reglee)

recipe_flags() { # recette -> drapeaux de construction
  case "$1" in
    lecteur) echo "--stronglyTyped $TACTICS --tilingOptimizationLevel=0" ;;
    reglee) echo "--stronglyTyped $TACTICS --builderOptimizationLevel=5 --tilingOptimizationLevel=3" ;;
    *) return 1 ;;
  esac
}

# trt_time ONNX FORMES ÉTIQUETTE RECETTE DOSSIER -> médiane du « GPU Compute Time » (ms), en graphe CUDA.
# FORMES : « x:2x3840x128,message:… » pour un graphe à formes libres, vide pour un graphe à taille fixe.
trt_time() {
  local onnx="$1" shapes="$2" tag="$3-$4" out="$5" build=() run=() flags median
  flags="$(recipe_flags "$4")"
  if [ -n "$shapes" ]; then
    build=(--minShapes="$shapes" --optShapes="$shapes" --maxShapes="$shapes")
    run=(--shapes="$shapes")
  fi
  if [ ! -f "$out/$tag.engine" ]; then
    # shellcheck disable=SC2086
    "$TRTEXEC" --onnx="$onnx" "${build[@]}" $flags --saveEngine="$out/$tag.engine" >"$out/$tag.build.log" 2>&1 ||
      { echo "construction de $tag échouée ($out/$tag.build.log)" >&2; rm -f "$out/$tag.engine"; return 1; }
  fi
  "$TRTEXEC" --loadEngine="$out/$tag.engine" "${run[@]}" --useCudaGraph --noDataTransfers --useSpinWait \
    --warmUp=500 --iterations=300 --dumpProfile --separateProfileRun --exportProfile="$out/$tag.profile.json" \
    >"$out/$tag.run.log" 2>&1 || { echo "mesure de $tag échouée ($out/$tag.run.log)" >&2; return 1; }
  median="$(sed -n '/GPU Compute Time: min/{s/.*median = \([0-9.]*\) ms.*/\1/p;q;}' "$out/$tag.run.log")"
  [ -n "$median" ] || { echo "pas de médiane dans $out/$tag.run.log" >&2; return 1; }
  echo "$median"
}

mean2() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.4f", (a + b) / 2 }'; }
min2() { awk -v a="$1" -v b="$2" 'BEGIN { print (b < a ? b : a) }'; }
ratio() { awk -v n="$1" -v t="$2" 'BEGIN { printf "%.3f", n / t }'; }
held() { awk -v r="$1" -v s="$2" 'BEGIN { print (r <= s ? "oui" : "non") }'; }
bench_header() { # la carte, le pilote et TensorRT, pour l'en-tête du rapport
  echo "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo carte inconnue)," \
    "$(sed -n 's/.*\(TensorRT v[0-9]*\).*/\1/p;T;q' <<<"$TRT_HELP")"
}
