#!/usr/bin/env bash
# Aucune option ne vient de l'environnement : profil ou DebugFlags seulement. Interdit dans src/,
# hors commentaires et chaînes : lire une variable (getenv et ses variantes POSIX, Windows et SDL, appelées,
# prises par adresse ou importées par using) ou tout le bloc (environ, _wenviron,
# GetEnvironmentStrings, _NSGetEnviron). Seule exception, la ligne marquée « env: hérité » : le bloc
# transmis tel quel à un processus enfant, sans qu'aucune valeur n'en soit lue.
set -euo pipefail
names='secure_getenv|getenv|_wgetenv|getenv_s|_wgetenv_s|_dupenv_s|_wdupenv_s|SDL_getenv|SDL_getenv_unsafe'
names+='|SDL_GetEnvironment\w*|GetEnvironmentVariable[AW]?|GetEnvironmentStrings[AW]?|_NSGetEnviron'
names+='|environ|_environ|__environ|_wenviron'
hits=$(git ls-files -z -- 'src/*' | AJP_ENV_NAMES="$names" xargs -0 perl -ne '
  BEGIN { $re = qr/\b(?:$ENV{AJP_ENV_NAMES})\b/ }
  (my $code = $_) =~ s{\x27(?:\\.|[^\x27\\])\x27}{}g;
  $code =~ s{"(?:\\.|[^"\\])*"}{""}g;
  $code =~ s{//.*}{};
  print "$ARGV:$.: $_" if $code =~ $re && !/env: hérité/;
  close ARGV if eof;')
if [[ -n "$hits" ]]; then
  printf '%s\n' "$hits"
  echo "::error::l'environnement ne se lit pas dans src/ (règle « aucune bascule cachée »)"
  exit 1
fi
