// Les lecteurs de fichiers et de flux venus de l'extérieur (P1.29) : un modèle ONNX (le seul lecteur
// protobuf du dépôt), un SRT et son encodage, une LUT .cube, un flux deflate (roues et zip
// téléchargés), le manifeste de mise à jour, l'import d'un .ini d'AnimeJaNai, la ligne de commande.
// Chacun lit des octets quelconques sans plantage ni lecture hors du tampon ; certains vérifient en
// plus une propriété (fuzz_check.h).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/options.h"
#include "core/inflate.h"
#include "core/update_manifest.h"
#include "fuzz_check.h"
#include "fuzz_targets.h"
#include "gpu/ir/import.h"
#include "render/cube_lut.h"
#include "subs/srt.h"
#include "subs/text_encoding.h"

namespace {

// Une copie à la taille exacte : ASan voit toute lecture au-delà.
std::string text_of(const std::uint8_t* data, std::size_t size) {
  return {reinterpret_cast<const char*>(data), size};
}

}  // namespace

int ajp::fuzz::onnx(const std::uint8_t* data, std::size_t size) {
  const std::vector<std::byte> bytes(reinterpret_cast<const std::byte*>(data),
                                     reinterpret_cast<const std::byte*>(data) + size);
  (void)ajp::gpu::ir::load_onnx(bytes);
  return 0;
}

int ajp::fuzz::srt(const std::uint8_t* data, std::size_t size) {
  const std::string utf8 = ajp::subs::to_utf8(text_of(data, size));
  std::vector<std::string> skipped;
  if (const auto cues = ajp::subs::parse_srt(utf8, &skipped)) {
    for (const auto& cue : *cues) (void)ajp::subs::srt_text_to_ass(cue.text);
  }
  return 0;
}

int ajp::fuzz::cube(const std::uint8_t* data, std::size_t size) {
  if (auto lut = ajp::render::parse_cube(text_of(data, size))) {
    // Une LUT lue se réécrit et se relit.
    ajp::fuzz::require(ajp::render::parse_cube(ajp::render::write_cube(*lut)).has_value(),
                       "une LUT écrite se relit");
  }
  return 0;
}

int ajp::fuzz::inflate(const std::uint8_t* data, std::size_t size) {
  // Lue par morceaux de 7 octets, pour éprouver les reprises ; la sortie bornée à 64 Mio (une bombe
  // de décompression n'est pas un défaut ici).
  std::size_t at = 0;
  std::uint64_t out = 0;
  (void)ajp::core::inflate_raw(
      [&](std::span<std::uint8_t> buffer) {
        const std::size_t n = std::min({buffer.size(), size - at, std::size_t{7}});
        std::copy_n(data + at, n, buffer.begin());
        at += n;
        return n;
      },
      [&](std::span<const std::uint8_t> bytes) {
        out += bytes.size();
        return out < (std::uint64_t{64} << 20U);
      });
  return 0;
}

int ajp::fuzz::manifest(const std::uint8_t* data, std::size_t size) {
  (void)ajp::core::parse_update_manifest(text_of(data, size));
  return 0;
}

int ajp::fuzz::options(const std::uint8_t* data, std::size_t size) {
  // Un argument par ligne.
  const std::string text = text_of(data, size);
  std::vector<std::string_view> args;
  std::string_view rest = text;
  while (!rest.empty()) {
    const std::size_t end = rest.find('\n');
    args.push_back(rest.substr(0, end));
    rest = end == std::string_view::npos ? std::string_view{} : rest.substr(end + 1);
  }
  (void)ajp::app::play::parse_play_options(args);
  return 0;
}
