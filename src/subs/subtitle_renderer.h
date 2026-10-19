// Moteur de rendu libass : ass_render_frame à l'horodatage d'une image de
// sortie, à la résolution de sortie, puis la liste ASS_Image en atlas R8.
// Un contenu inchangé réutilise l'atlas précédent, sans nouveau téléversement.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "core/types.h"
#include "subs/ass_library.h"
#include "subs/atlas_pool.h"
#include "subs/subtitle_frame.h"
#include "subs/subtitle_track.h"
#include "subs/vsfilter_color.h"

struct ass_renderer;
struct ass_image;
struct ass_library;

namespace ajp::subs {

inline constexpr int kMaxTargetSide = 16384;
inline constexpr int kMaxRenderThreads = 64;

enum class FontProvider : std::uint8_t {
  System,  // fontconfig, CoreText ou DirectWrite, selon le système
  None,    // seulement les polices jointes et default_font
};

struct RendererOptions {
  FontProvider fonts = FontProvider::System;
  std::string default_family = "sans-serif";
  std::filesystem::path default_font;  // facultatif : police de repli explicite
  int render_threads = 1;              // threads de libass : 1 aucun, 0 automatique, au plus 64
};

// Résolution de sortie, taille de stockage de la vidéo source pour l'aspect et
// l'échelle du flou (ass_set_storage_size), et matrice de la vidéo pour les
// couleurs VSFilter (vsfilter_color.h).
struct RenderTarget {
  int width = 0;
  int height = 0;
  int video_width = 0;  // 0 : inconnue, la sortie sert de stockage
  int video_height = 0;
  std::optional<VideoMatrix> video_matrix;  // nullopt : inconnue, couleurs du script telles quelles
  // Échelle des polices (ass_set_font_scale, comme sub-scale de mpv) : kMinFontScale à kMaxFontScale.
  float font_scale = 1.0F;
  friend bool operator==(const RenderTarget&, const RenderTarget&) = default;
};

inline constexpr float kMinFontScale = 0.25F;
inline constexpr float kMaxFontScale = 4.0F;

struct RenderRequest {
  std::int64_t pts_us = 0;
  std::uint32_t epoch = 0;
  RenderTarget target;
};

// Contenu neuf, et tous les atlas sont en vol : rappeler render() après un retour.
struct WaitForAtlas {};
using RenderOutcome = std::variant<SubtitleFrame, WaitForAtlas>;

class SubtitleRenderer {
 public:
  static std::expected<SubtitleRenderer, Refusal> create(const AssLibrary& library,
                                                         const RendererOptions& options);

  // Refus : cible ou horodatage hors plage (FieldOutOfRange), contenu au-delà
  // de l'atlas borné ou de kMaxRectsPerFrame (SubtitleAtlasFull). Un contenu
  // refusé l'est une fois : tant que libass le rend inchangé ou seulement
  // déplacé, les images suivantes sont vides, sans nouveau refus.
  std::expected<RenderOutcome, Refusal> render(const SubtitleTrack& track, const RenderRequest& request,
                                               AtlasPool& pool);

  // Oublie le dernier contenu, au changement de piste : l'atlas courant est relâché.
  void forget();

  // Une police vient d'être jointe. libass garde en cache chaque famille déjà
  // cherchée, introuvable ou remplacée comprise : si le moteur a déjà rendu,
  // il est recréé avant le rendu suivant.
  void fonts_changed();

 private:
  struct Free {
    void operator()(ass_renderer* renderer) const;
  };
  using Handle = std::unique_ptr<ass_renderer, Free>;
  static std::expected<Handle, Refusal> make(ass_library* library, const RendererOptions& options);
  SubtitleRenderer(Handle renderer, ass_library* library, RendererOptions options);
  std::expected<void, Refusal> refresh_fonts();
  void configure(const RenderTarget& target);
  std::expected<RenderOutcome, Refusal> build(SubtitleFrame frame, AtlasPool& pool);
  std::unexpected<Refusal> refuse_content(Refusal refusal);

  Handle renderer_;
  ass_library* library_;  // la bibliothèque survit au moteur
  RendererOptions options_;
  bool rendered_ = false;     // ass_render_frame appelé depuis la création du moteur
  bool stale_fonts_ = false;  // à recréer avant le prochain rendu
  std::optional<RenderTarget> target_;
  std::vector<const ass_image*> pieces_;  // images visibles du dernier ass_render_frame
  // Dernier contenu bâti. in_sync_ : il correspond au dernier ass_render_frame,
  // dont libass compare l'image suivante. refused_ : ce contenu a été refusé.
  std::shared_ptr<const SubtitleAtlas> current_;
  std::vector<SubtitleRect> rects_;
  bool in_sync_ = false;
  bool refused_ = false;
  bool vsfilter_ = false;  // couleurs converties comme VSFilter, pour la piste et la cible courantes
  std::uint64_t generation_ = 0;
};

}  // namespace ajp::subs
