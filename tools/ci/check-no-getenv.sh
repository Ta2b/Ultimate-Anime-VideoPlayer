#!/usr/bin/env bash
# Aucune option ne vient de l'environnement : play.toml ou la ligne de commande seulement. Interdit dans src/,
# hors commentaires et chaînes : lire une variable (getenv et ses variantes POSIX, Windows et SDL, appelées,
# prises par adresse ou importées par using) ou tout le bloc (environ, _wenviron,
# GetEnvironmentStrings, _NSGetEnviron). Seule exception, la ligne marquée « env: hérité » : le bloc
# transmis tel quel à un processus enfant, sans qu'aucune valeur n'en soit lue ; la ligne marquée
# « env: lanceur » : un fait posé par le lanceur d'un paquet (AppImage) ; et la ligne marquée « env: système » :
# une variable de la norme XDG que la session ou le lanceur pose (XDG_CACHE_HOME, C34-13 ;
# XDG_ACTIVATION_TOKEN, C34-15). Le réglage Smooth Motion transmet l'état du pilote NVIDIA au redémarrage.
# Les essais A/B documentés utilisent une variable KIRARI_AB_ et la marque « env: banc ».
set -euo pipefail
names='secure_getenv|getenv|_wgetenv|getenv_s|_wgetenv_s|_dupenv_s|_wdupenv_s|SDL_getenv|SDL_getenv_unsafe'
names+='|SDL_GetEnvironment\w*|GetEnvironmentVariable[AW]?|GetEnvironmentStrings[AW]?|ExpandEnvironmentStrings[AW]?|_NSGetEnviron'
names+='|environ|_environ|__environ|_wenviron'
hits=$(git ls-files -z -- 'src/*' | KIRARI_ENV_NAMES="$names" xargs -0 perl -ne '
  BEGIN { $re = qr/\b(?:$ENV{KIRARI_ENV_NAMES})\b/ }
  my $scan = $_;
  if (/\/\/ env: banc\s*$/) {
    $scan =~ s{\bstd::getenv\(\s*"KIRARI_AB_[A-Z0-9_]+"\s*\)}{""};
  }
  my $smooth = $ARGV =~ m{\Asrc/app/(?:main\.cpp|play/placebo_open\.cpp)\z}
            && /\/\/ env: pilote NVIDIA\s*$/;
  (my $code = $scan) =~ s{\x27(?:\\.|[^\x27\\])\x27}{}g;
  $code =~ s{"(?:\\.|[^"\\])*"}{""}g;
  $code =~ s{//.*}{};
  print "$ARGV:$.: $_" if $code =~ $re && !/env: (hérité|lanceur|système)/ && !$smooth;
  close ARGV if eof;')
if [[ -n "$hits" ]]; then
  printf '%s\n' "$hits"
  echo "::error::l'environnement ne se lit pas dans src/ (règle « aucune bascule cachée »)"
  exit 1
fi
