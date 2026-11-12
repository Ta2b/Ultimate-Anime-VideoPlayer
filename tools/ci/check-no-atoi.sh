#!/usr/bin/env bash
# Les nombres se lisent par from_chars, qui dit l'échec et la fin du texte lu (P2.35) : atoi et consorts
# rendent 0 sur « abc » sans rien dire, sscanf lit au-delà de ce qu'on croit. Interdits dans src/, hors
# commentaires et chaînes.
set -euo pipefail
names='atoi|atol|atoll|atof|sscanf|swscanf|vsscanf|strtod|strtof|strtold|strtol|strtoll|strtoul|strtoull|stoi|stol|stoll|stoul|stoull|stof|stod|stold'
hits=$(git ls-files -z -- 'src/*' | AJP_NAMES="$names" xargs -0 perl -ne '
  BEGIN { $re = qr/\b(?:$ENV{AJP_NAMES})\s*\(/ }
  (my $code = $_) =~ s{\x27(?:\\.|[^\x27\\])\x27}{}g;
  $code =~ s{"(?:\\.|[^"\\])*"}{""}g;
  $code =~ s{//.*}{};
  print "$ARGV:$.: $_" if $code =~ $re;
  close ARGV if eof;')
if [[ -n "$hits" ]]; then
  printf '%s\n' "$hits"
  echo "::error::un nombre se lit par from_chars dans src/ (atoi, atof et sscanf interdits)"
  exit 1
fi
