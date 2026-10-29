// Des images fixes tirées d'une vidéo, en RGB 8 bits : le corpus de la dérivation du SPAN FP8
// (app/play/span_derive.h, docs/span-fp8-cutlass.md). Le lecteur calibre sur la vidéo que l'utilisateur
// regarde, pas sur un dossier d'images : douze images espacées sur la durée, les noires et les quasi
// uniformes écartées. Rien n'est gardé : les images ne vivent que dans la mémoire de l'appelant.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <stop_token>
#include <vector>

#include "core/frame.h"
#include "core/types.h"

namespace ajp::media {

// Une image découpée au centre de la source, RGB 8 bits planaire (R, G, B de height × width octets l'un après
// l'autre, comme SpanStill de gpu/backend.h), avec de quoi juger si elle sert.
struct Still {
  int width = 0;
  int height = 0;
  std::int64_t pts_us = 0;
  std::vector<std::uint8_t> rgb;
  double mean = 0.0;    // luminance moyenne, de 0 à 1
  double spread = 0.0;  // écart type de la luminance, de 0 à 1
};

struct StillsSpec {
  int count = 12;       // images gardées
  int candidates = 24;  // instants essayés, de 8 % à 92 % de la durée
  int minimum = 8;      // en dessous, la vidéo ne donne pas de corpus : refus
  // La découpe : au plus max_width × max_height, au centre, ramenée à un multiple de align_width ×
  // align_height (ce que les noyaux du SPAN FP8 prennent : 128 × 4).
  int max_width = 1920;
  int max_height = 1088;
  int align_width = 128;
  int align_height = 4;
  double min_mean = 0.06;    // plus sombre : une image noire, écartée
  double min_spread = 0.03;  // moins contrastée : un aplat, écarté
};

// Le RGB 8 bits de la découpe centrale d'une image décodée en mémoire hôte (Nv12, Yuv420p, P010, Yuv420p10 ;
// matrice BT.601 ou BT.709, plage limitée ou pleine ; chroma répétée au plus proche). Refus FormatUnsupported
// pour une image HDR (PQ, HLG, BT.2020), d'un autre format ou hors de la mémoire hôte, et pour une image trop
// petite pour la découpe.
std::expected<Still, Refusal> still_from_frame(const DecodedFrame& frame, const StillsSpec& spec);

// Les indices des images à garder parmi celles qui sont assez claires et assez contrastées : les count plus
// contrastées, dans l'ordre du temps. Moins que minimum à garder : vide.
std::vector<std::size_t> pick_stills(std::span<const Still> stills, const StillsSpec& spec);

// Décode spec.candidates images de la première piste vidéo de file (décodage logiciel, aucun GPU) et garde
// les spec.count meilleures. progress reçoit la fraction faite. Refus : fichier illisible, pas de piste
// vidéo, durée inconnue, vidéo HDR, Cancelled, ou moins de spec.minimum images exploitables.
std::expected<std::vector<Still>, Refusal> extract_stills(const std::filesystem::path& file,
                                                          const StillsSpec& spec, const std::stop_token& stop,
                                                          const std::function<void(float)>& progress = {});

}  // namespace ajp::media
