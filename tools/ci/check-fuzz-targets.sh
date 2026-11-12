#!/usr/bin/env bash
# La campagne libFuzzer (fuzz.yml) lance toutes les cibles de tests/fuzz (C28-01) : les fonctions de
# fuzz_targets.h sont exactement celles que fuzz-targets.py extrait de CMakeLists.txt, et chacune a son corpus.
# Une cible oubliée de la campagne n'était fuzzée que par le rejeu court de ctest.
set -euo pipefail
cd "$(dirname "$0")/../.."
extracted="$(python3 -I tools/ci/fuzz-targets.py | sort)"
declared="$(sed -n 's/^int \([a-z_0-9]*\)(const std::uint8_t\* data.*/\1/p' tests/fuzz/fuzz_targets.h | sort)"
bad=0
if ! diff <(echo "$declared") <(echo "$extracted") >&2; then
  echo "check-fuzz-targets : '<' déclarée dans fuzz_targets.h, '>' extraite pour la campagne : elles doivent être les mêmes" >&2
  bad=1
fi
for name in $extracted; do
  [ -d "tests/fuzz/corpus/$name" ] || { echo "check-fuzz-targets : $name n'a pas de corpus" >&2; bad=1; }
done
echo "fuzz : $(echo "$extracted" | wc -l) cibles pour la campagne"
exit "$bad"
