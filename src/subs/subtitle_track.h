// Piste de sous-titres libass : en-tête ASS d'un conteneur suivi de ses
// paquets, piste texte (SRT, UTF-8) d'un conteneur, ou fichier externe.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <tuple>

#include "core/types.h"
#include "subs/ass_library.h"
#include "subs/timebase.h"

struct ass_track;

namespace ajp::subs {

// Taille maximale d'un script lu en entier (fichier externe, en-tête).
inline constexpr std::size_t kMaxScriptBytes = std::size_t{64} << 20;
// Taille maximale d'un paquet.
inline constexpr std::size_t kMaxPacketBytes = std::size_t{1} << 20;

enum class TrackKind : std::uint8_t {
  Ass,   // paquets Matroska « ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text »
  Text,  // paquets de texte SRT en UTF-8, balises <i>, <font>…
};

struct SubtitlePacket {
  std::int64_t start_us = 0;
  std::int64_t duration_us = 0;
  std::string data;
};

class SubtitleTrack {
 public:
  // Piste ASS d'un conteneur, depuis son CodecPrivate. Refus si l'en-tête est
  // vide, trop gros ou sans section ASS reconnue.
  static std::expected<SubtitleTrack, Refusal> from_header(const AssLibrary& library,
                                                           std::string_view header);
  // Piste texte d'un conteneur, sur l'en-tête text_track_header().
  static std::expected<SubtitleTrack, Refusal> text(const AssLibrary& library);
  // Script ASS ou SSA complet, en mémoire.
  static std::expected<SubtitleTrack, Refusal> from_ass(const AssLibrary& library, std::string_view script);
  // Fichier SRT complet, en mémoire.
  static std::expected<SubtitleTrack, Refusal> from_srt(const AssLibrary& library, std::string_view file);
  // Fichier externe .ass, .ssa ou .srt, d'au plus kMaxScriptBytes.
  static std::expected<SubtitleTrack, Refusal> load(const AssLibrary& library,
                                                    const std::filesystem::path& path);

  // Ajoute un paquet ; un doublon, revu après un seek, est ignoré.
  // Refus : horodatage, durée ou taille hors de leurs bornes.
  std::expected<void, Refusal> add_packet(const SubtitlePacket& packet);

  TrackKind kind() const { return kind_; }
  int event_count() const;
  ass_track* get() const { return track_.get(); }

 private:
  struct Free {
    void operator()(ass_track* track) const;
  };
  SubtitleTrack(std::unique_ptr<ass_track, Free> track, TrackKind kind);
  static std::expected<SubtitleTrack, Refusal> create(const AssLibrary& library, TrackKind kind,
                                                      std::string_view header);

  std::unique_ptr<ass_track, Free> track_;
  TrackKind kind_;
  // Paquets texte déjà vus (début, durée, texte) : libass ne déduplique que par
  // ReadOrder, que le texte n'a pas.
  std::set<std::tuple<std::int64_t, std::int64_t, std::string>> seen_text_;
};

}  // namespace ajp::subs
