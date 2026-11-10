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
  double sample_aspect = 1.0;  // forme du pixel de la vidéo (SAR) : l'écran l'étire à l'affichage
  std::optional<VideoMatrix> video_matrix;  // nullopt : inconnue, couleurs du script telles quelles
  // Échelle des polices (ass_set_font_scale, comme sub-scale de mpv) : kMinFontScale à kMaxFontScale.
  float font_scale = 1.0F;
  // Part de la hauteur (0 à kMaxLineLift) dont les dialogues du bas montent, pour passer au-dessus de la
  // barre du lecteur (ass_set_line_position, comme sub-pos de mpv) : libass ne déplace que les événements
  // sans position explicite, chacun en entier. Un panneau \pos ou \move reste sur son enseigne (A24-19).
  float line_lift = 0.0F;
  friend bool operator==(const RenderTarget&, const RenderTarget&) = default;
};

inline constexpr float kMinFontScale = 0.25F;
inline constexpr float kMaxFontScale = 4.0F;
inline constexpr float kMaxLineLift = 0.5F;

struct RenderRequest {
  std::int64_t pts_us = 0;
  std::uint32_t epoch = 0;
  // Décalage des sous-titres (P3.72) : libass lit l'instant pts_us − delay_us ; > 0, ils viennent plus tard.
  std::int64_t delay_us = 0;
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

  // Appels à libass depuis la création (P3.39 : un écran sans événement n'en coûte aucun).
  std::uint64_t libass_calls() const { return libass_calls_; }

 private:
  struct Free {
    void operator()(ass_renderer* renderer) const;
  };
  using Handle = std::unique_ptr<ass_renderer, Free>;
  static std::expected<Handle, Refusal> make(ass_library* library, const RendererOptions& options);
  SubtitleRenderer(Handle renderer, ass_library* library, RendererOptions options);
  std::expected<void, Refusal> refresh_fonts();
  void configure(const RenderTarget& target, bool use_margins);
  std::expected<RenderOutcome, Refusal> render_at(const SubtitleTrack& track, const RenderRequest& request,
                                                  AtlasPool& pool, bool use_margins);
  static RenderTarget shrunk(RenderTarget target, int shrink);
  static constexpr int kMaxShrink = 4;
  std::expected<RenderOutcome, Refusal> build(SubtitleFrame frame, AtlasPool& pool);
  std::unexpected<Refusal> refuse_content(Refusal refusal);

  Handle renderer_;
  ass_library* library_;  // la bibliothèque survit au moteur
  RendererOptions options_;
  bool rendered_ = false;               // ass_render_frame appelé depuis la création du moteur
  bool stale_fonts_ = false;            // à recréer avant le prochain rendu
  std::optional<RenderTarget> target_;  // configuré dans libass : la demande, réduite de shrink_
  // Réduction du rendu (1, 2, 4) : un instant qui ne tient pas dans l'atlas (dessins plein écran en
  // 4K) est rendu plus petit, puis étiré à l'affichage, plutôt que jeté. Revient à 1 quand l'écran
  // se vide ou que la demande change.
  int shrink_ = 1;
  bool use_margins_ = false;              // ass_set_use_margins de la dernière configuration
  std::vector<const ass_image*> pieces_;  // images visibles du dernier ass_render_frame
  // Dernier contenu bâti. in_sync_ : il correspond au dernier ass_render_frame,
  // dont libass compare l'image suivante. refused_ : ce contenu a été refusé.
  std::shared_ptr<const SubtitleAtlas> current_;
  std::vector<SubtitleRect> rects_;
  bool in_sync_ = false;
  bool refused_ = false;
  bool vsfilter_ = false;  // couleurs converties comme VSFilter, pour la piste et la cible courantes
  std::uint64_t generation_ = 0;
  // P3.39 : après un rendu vide, aucun événement avant quiet_until_ms_ (piste de quiet_events_ événements) :
  // les instants de [quiet_from_ms_, quiet_until_ms_) sont vides sans appeler libass.
  long long quiet_from_ms_ = 0;
  long long quiet_until_ms_ = -1;
  int quiet_events_ = -1;
  std::uint64_t libass_calls_ = 0;
};

}  // namespace ajp::subs
