#!/usr/bin/env bash
# Un journal ne dit pas où sont les vidéos (A39-03) : ajp.log se joint à un rapport public. Un appel de
# journal (info, warn, error, critical, debug, trace ; par ->, . ou ::) de src/ qui écrit un chemin par core::to_utf8 ou utf8_path doit
# passer par filename() ou stem(). Les appels qui restent (dossiers de l'application, chemins d'export
# ou de moteurs, déjà réduits à leur nom par le rapport) ne peuvent que diminuer : baisser kMax.
set -euo pipefail
cd "$(dirname "$0")/../.."
readonly kMax=6
hits="$(git ls-files -z -- 'src/*.cpp' | xargs -0 perl -0777 -ne '
  while (/(?:->|\.|::)(?:info|warn|error|critical|debug|trace)\((?:[^;"]|"(?:\\.|[^"\\])*")*;/g) {
    my $call = $&;
    my $line = 1 + (() = substr($_, 0, $-[0]) =~ /\n/g);
    next unless $call =~ /(?:utf8_path|to_utf8)\(/;
    next if $call =~ /(?:filename|stem)\(\)/;
    print "$ARGV:$line\n";
  }
  ')"
count=0
[[ -n "$hits" ]] && count="$(printf '%s\n' "$hits" | wc -l)"
if ((count > kMax)); then
  echo "journal : $count appels écrivent un chemin complet, $kMax au plus ; journaliser filename() (A39-03)" >&2
  printf '%s\n' "$hits" >&2
  exit 1
fi
echo "journal : $count appels avec un chemin complet (au plus $kMax)"
