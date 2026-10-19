// Texte de l'OSD rendu par libass, comme les sous-titres : un script d'une seule ligne de
// dialogue, en haut à gauche, rendu à la taille de sortie dans un atlas R8. L'écran le
// dessine par le même chemin que les sous-titres.
//
// Provisoire : l'OSD du lecteur sera ImGui, dessiné par render (pl_dispatch_vertex).
#pragma once

#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "app/play/links.h"
#include "core/types.h"
#include "subs/ass_library.h"
#include "subs/atlas_pool.h"
#include "subs/subtitle_frame.h"
#include "subs/subtitle_renderer.h"
#include "subs/subtitle_track.h"

namespace ajp::app::play {

class OsdText {
 public:
  static std::expected<std::unique_ptr<OsdText>, Refusal> create();
  OsdText(const OsdText&) = delete;
  OsdText& operator=(const OsdText&) = delete;
  ~OsdText();

  // L'image des lignes à la taille de sortie ; recalculée seulement si le texte ou la
  // taille changent. Vide : rien à dessiner.
  std::expected<const subs::SubtitleFrame*, Refusal> render(std::span<const std::string> lines,
                                                            OutputSize size);

 private:
  OsdText(subs::AssLibrary library, subs::SubtitleRenderer renderer);

  subs::AssLibrary library_;
  subs::SubtitleRenderer renderer_;
  std::unique_ptr<subs::AtlasPool> pool_;
  std::optional<subs::SubtitleTrack> track_;
  std::vector<std::string> lines_;
  OutputSize size_;
  subs::SubtitleFrame frame_;
};

// Le script ASS de l'OSD : une ligne de dialogue de durée illimitée, en haut à gauche, les
// caractères spéciaux d'ASS neutralisés.
std::string osd_script(std::span<const std::string> lines, OutputSize size);

}  // namespace ajp::app::play
