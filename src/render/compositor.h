// Compositor : un slot du ring devient l'image finale d'une cible, sur le thread main.
//
//   slot RGBA16 (couleur de la source ou PQ de TrueHDR)
//     → pl_render_image : mise à l'échelle avec bandes, conversion vers l'espace de la cible,
//       grade sur l'étape OUTPUT (sortie PQ seulement), puis sous-titres en pl_overlay ;
//     → OSD ImGui par pl_dispatch_vertex, en dernier.
// La cible est une pl_frame fournie par l'appelant : celle de la swapchain de present, ou une
// texture hors écran (OffscreenTarget). render ne touche jamais à la swapchain ; app relie les
// deux. Une cible dont l'espace est inconnu prend celui de DisplayInfo.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "core/profile.h"
#include "core/resolve.h"
#include "core/types.h"
#include "render/color.h"
#include "render/grade_params.h"
#include "subs/subtitle_frame.h"

struct pl_frame;
struct pl_tex_t;
struct ImDrawData;
struct ImGuiIO;
struct ImGuiPlatformIO;

namespace ajp::render {

class RenderDevice;

struct CompositionSettings {
  core::DisplayInfo display;
  GradeSettings grade;
  core::HdrMode route = core::HdrMode::Off;  // la route HDR du Plan
};

// Ce qui se pose sur l'image, dans l'ordre : sous-titres (après le grade), puis OSD.
struct ComposeLayers {
  const subs::SubtitleFrame* subtitles = nullptr;
  ImDrawData* osd = nullptr;  // celui d'ImGui::GetDrawData(), dont les textures se mettent à jour
  bool calibration = false;   // mire d'étalonnage à la place de l'image, sortie PQ seulement
};

class Compositor {
 public:
  static std::expected<Compositor, Refusal> create(const RenderDevice& device);

  Compositor(Compositor&&) noexcept;
  Compositor& operator=(Compositor&&) noexcept;
  Compositor(const Compositor&) = delete;
  Compositor& operator=(const Compositor&) = delete;
  ~Compositor();

  // FieldOutOfRange : réglage du grade ou pic de l'écran hors plage ; l'ancien réglage reste.
  std::expected<void, Refusal> configure(const CompositionSettings& settings);
  const CompositionSettings& settings() const;

  // Bascule A/B du grade (cmd::ToggleGrade) : prise en compte au prochain compose().
  void set_grade_enabled(bool enabled);
  bool grade_enabled() const;
  // Le grade appliqué par le dernier compose().
  GradeMode grade_mode() const;

  // slot : la texture rendue par OutputRing::acquire, ou toute pl_tex RGBA échantillonnable.
  // RenderFailed : libplacebo refuse le rendu ; InputMismatch : sous-titres incohérents.
  std::expected<void, Refusal> compose(const pl_tex_t* slot, const SlotColor& color, const pl_frame& target,
                                       const ComposeLayers& layers = {});

  // Oublie l'état temporel (historique AJG_STATE du grade, caches) : seek, nouvelle source.
  void reset();
  // Rend l'atlas de sous-titres tenu à subs.
  void drop_subtitles();

  // Somme des temps GPU des passes du dernier rendu, en nanosecondes, telle que libplacebo la
  // mesure (avec une ou deux images de retard) ; 0 si l'appareil ne sait pas mesurer.
  std::uint64_t last_gpu_time_ns() const;

  // Backend de l'OSD : voir ImGuiLayer (imgui_layer.h).
  static void configure_imgui(ImGuiIO& io);
  void release_imgui_textures(ImGuiPlatformIO& platform);

 private:
  struct Impl;
  explicit Compositor(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::render
