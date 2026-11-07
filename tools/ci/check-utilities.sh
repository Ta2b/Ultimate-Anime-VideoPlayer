#!/usr/bin/env bash
# Les petits utilitaires n'existent qu'une fois (P5.13) : refuse dans core/types.h, trim, ascii_lower et
# iequal dans core/text.h, le texte d'un AVERROR dans core/av_error.h, le demi-flottant dans
# kernels/ref/half.h. Une copie locale dans src/ est refusée, hors commentaires et chaînes.
set -euo pipefail
cd "$(dirname "$0")/../.."
# motif perl ;; fichiers qui seuls peuvent le contenir (motif) ;; ce qu'il faut écrire à la place. vt_pixels.cpp
# garde son _Float16 matériel et retombe sur kernels/ref/half.h sans lui.
rules=(
  'std::unexpected<\w*:*Refusal>\s+refuse\(\s*RefusalCode\s+\w+\s*,\s*std::string\s+\w+\s*\)\s*\{ ;; src/core/types.h ;; core/types.h : ajp::refuse'
  'std::(?:tolower|toupper)\s*\( ;; src/core/text.h ;; core::ascii_lower (core/text.h)'
  'std::string_view\s+(?:trim|trimmed)\s*\(\s*std::string_view ;; src/core/text.h ;; core::trim (core/text.h)'
  '\bav_strerror\s*\( ;; src/core/av_error.h ;; core::av_error_text (core/av_error.h)'
  'std::uint16_t\s+(?:to_half|float_to_half)\s*\(\s*float ;; src/kernels/ref/half.h|src/gpu/coreml/vt_pixels.cpp ;; kernels::ref::float_to_half'
)
status=0
for rule in "${rules[@]}"; do
  pattern=${rule%% ;; *}
  rest=${rule#* ;; }
  owner=${rest%% ;; *}
  instead=${rest#* ;; }
  hits=$(git ls-files -z -- 'src/*.cpp' 'src/*.h' 'src/*.mm' 'src/*.cu' ':!src/kernels/ref/half.cpp' |
    AJP_RE="$pattern" AJP_OWNER="$owner" xargs -0 perl -ne '
      BEGIN { $re = qr/$ENV{AJP_RE}/ }
      next if $ARGV =~ /^(?:$ENV{AJP_OWNER})$/;
      (my $code = $_) =~ s{"(?:\\.|[^"\\])*"}{""}g;
      $code =~ s{//.*}{};
      print "$ARGV:$.: $_" if $code =~ $re;
      close ARGV if eof;')
  if [[ -n "$hits" ]]; then
    printf '%s\n' "$hits"
    echo "::error::copie locale d'un utilitaire partagé : utiliser $instead"
    status=1
  fi
done
exit "$status"
