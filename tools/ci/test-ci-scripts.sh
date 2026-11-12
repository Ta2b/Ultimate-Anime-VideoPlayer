#!/usr/bin/env bash
# Essaie les contrôles de tools/ci/ sur des dépôts jetables (C28-07, C28-09, C28-10) : chacun doit refuser le défaut
# qu'il garde, pour sa raison, et laisser passer le cas sain. Un contrôle qui n'échoue jamais ne garde rien.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail=0
expect() {  # attendu (0|1) description message-attendu commande…
  local want=$1 what=$2 message=$3; shift 3
  local got=0 output
  output="$("$@" 2>&1)" || got=1
  if [ "$got" != "$want" ]; then
    echo "ÉCHEC : $what (code attendu $want)" >&2; printf '%s\n' "$output" >&2; fail=1
  elif [ "$want" = 1 ] && ! grep -qF -- "$message" <<<"$output"; then
    echo "ÉCHEC : $what : refusé, mais pas pour « $message »" >&2; printf '%s\n' "$output" >&2; fail=1
  else
    echo "ok : $what"
  fi
}
# Un dépôt factice qui reçoit une copie du script et un src/ de la forme voulue.
repo() {  # nom script contenu-de-src/x.cpp
  local dir="$work/$1"
  mkdir -p "$dir/tools/ci" "$dir/src"
  cp "$here/$2" "$dir/tools/ci/$2"
  printf '%s\n' "$3" > "$dir/src/x.cpp"
  git -C "$dir" init -q
  git -C "$dir" add -A
  echo "$dir/tools/ci/$2"
}

# C28-07 : un rapport JUnit absent ou vide n'est pas un succès.
expect 1 "check-skips : rapport absent" "absent ou vide" "$here/check-skips.sh" "$work/nul.xml" core-tsan
: > "$work/vide.xml"
expect 1 "check-skips : rapport vide" "absent ou vide" "$here/check-skips.sh" "$work/vide.xml" core-tsan
{ echo '<testsuite>'; for i in $(seq 1 20); do echo "<testcase name=\"t$i\" status=\"run\"/>"; done; echo '</testsuite>'; } > "$work/ok.xml"
expect 0 "check-skips : rapport complet" "" "$here/check-skips.sh" "$work/ok.xml" core-tsan
printf '<testsuite><testcase name="a" status="run"/></testsuite>' > "$work/court.xml"
expect 1 "check-skips : rapport amputé" "moins que le plancher" "$here/check-skips.sh" "$work/court.xml" core-tsan

# C28-09 : generic_string() et deux appels sur une même ligne comptent.
script="$(repo ps check-path-string.sh 'auto a = p.string(); auto b = q.string();')"
expect 1 "check-path-string : deux appels sur une ligne comptent deux" "appels dans src/" bash -c \
  "sed -i 's/^readonly kMax=.*/readonly kMax=1/' '$script' && '$script'"
script="$(repo pg check-path-string.sh 'auto a = p.generic_string();')"
expect 1 "check-path-string : generic_string() compte" "appels dans src/" bash -c \
  "sed -i 's/^readonly kMax=.*/readonly kMax=0/' '$script' && '$script'"
script="$(repo pn check-path-string.sh 'auto a = core::to_utf8(p);')"
expect 0 "check-path-string : to_utf8 passe" "" bash -c "sed -i 's/^readonly kMax=.*/readonly kMax=0/' '$script' && '$script'"

# C28-10 : un mois en toutes lettres suivi d'une année est une date.
script="$(repo cm check-comments.sh '// exporté avant novembre 2026')"
expect 1 "check-comments : « novembre 2026 » refusé" "commentaire avec un nom ou une date" "$script"
script="$(repo cd check-comments.sh '// le 12/11/2026')"
expect 1 "check-comments : jj/mm/aaaa refusé" "commentaire avec un nom ou une date" "$script"
script="$(repo cn check-comments.sh '// exemple de format : 2027-02-27')"
expect 0 "check-comments : un exemple de format passe" "" "$script"
exit "$fail"
