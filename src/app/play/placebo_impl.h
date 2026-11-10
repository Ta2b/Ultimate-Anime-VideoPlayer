// Détails de PlaceboScreen, partagés par ses fichiers : le ring exporté vers CUDA et les
// calques (sous-titres, OSD). Interne à placebo_*.cpp.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <libplacebo/cache.h>
#include <libplacebo/dispatch.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/vulkan.h>
#include <spdlog/logger.h>

#include "app/play/compose_key.h"
#include "app/play/links.h"
#include "app/play/live_settings.h"
#include "app/play/osd_text.h"
#include "app/play/placebo_feedback.h"
#include "app/play/placebo_imgui.h"
#include "app/play/placebo_screen.h"
#include "app/play/present_level.h"
#include "app/play/ring_watcher.h"
#include "app/play/video_rate.h"
#include "app/play/vulkan_share.h"
#include "core/gpu_device.h"
#include "core/ring.h"
#include "render/grade.h"
#include "render/sharpen.h"
#include "subs/subtitle_frame.h"

namespace ajp::app::play {

// Les slots du ring : textures RGBA16 exportées et leurs sémaphores timeline. Un slot est soit
// tenu par CUDA (gpu), soit rendu à libplacebo le temps que main le lise (core/ring.h).
class PlaceboRing {
 public:
  // external : le consommateur est une autre API (CUDA), à qui les slots sont cédés par
  // VK_QUEUE_FAMILY_EXTERNAL ; sinon le backend écrit sur le même VkDevice, sans transfert de file.
  // wanted : RGB10A2 si l'appareil l'inscrit et le rend en VK_FORMAT_A2B10G10R10_UNORM_PACK32 (étape 5.5a),
  // RGBA16 sinon ; le format pris est dans chaque RingSlotDesc.
  std::expected<std::vector<RingSlotDesc>, Refusal> create(pl_gpu gpu, int width, int height,
                                                           std::uint32_t count, bool external,
                                                           RingFormat wanted = RingFormat::Rgba16Unorm);
  // La texture d'un RingFrame : rendue à libplacebo, qui attend ready_value, à sa première lecture.
  std::expected<pl_tex, Refusal> acquire(pl_gpu gpu, const RingFrame& frame);
  // Cède le slot à CUDA après le dernier rendu qui l'a lu (signal de ready_value + 1).
  SlotFree retire(pl_gpu gpu, const RingFrame& frame);
  // Le sémaphore du slot a atteint ready_value : l'image est écrite. Lu sans attendre.
  // A31-13 : l'appareil perdu à la lecture du sémaphore rend faux (l'image n'est pas prise) et le retient.
  bool ready(const RingFrame& frame) const;
  bool lost() const { return lost_; }
  void clear(pl_gpu gpu);
  // Le sémaphore du slot d'un RingFrame du ring courant, sinon VK_NULL_HANDLE.
  VkSemaphore semaphore_of(const RingFrame& frame) const;
  // Le veilleur d'écriture (ring_watcher.h) qui attend ces sémaphores : oublié avant leur destruction.
  void set_watcher(RingWatcher* watcher) { watcher_ = watcher; }
  VkDevice device() const { return device_; }

 private:
  struct Slot {
    pl_tex tex = nullptr;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    // Poignée exportée du sémaphore (descripteur, ou HANDLE sous Windows) ; fermée avec le ring.
    std::optional<std::uintptr_t> semaphore_export;
    std::uint64_t placebo_value = 0;  // ready_value que libplacebo attend ; 0 : tenu par CUDA
  };
  std::expected<pl_tex, Refusal> slot_of(const RingFrame& frame);
  std::uint32_t family() const;

  std::vector<Slot> slots_;
  std::uint32_t generation_ = 0;
  bool external_ = true;
  VkDevice device_ = VK_NULL_HANDLE;
  PFN_vkGetSemaphoreCounterValue counter_ = nullptr;
  RingWatcher* watcher_ = nullptr;
  mutable bool lost_ = false;  // vkGetSemaphoreCounterValue a rendu VK_ERROR_DEVICE_LOST
};

// Sous-titres et OSD en calques monochromes, au-dessus de l'image, en pixels de la sortie.
class Overlays {
 public:
  explicit Overlays(pl_gpu gpu) : gpu_(gpu) {}
  ~Overlays();
  Overlays(const Overlays&) = delete;
  Overlays& operator=(const Overlays&) = delete;

  // Les calques d'une présentation : tous, puis les sous-titres (dans l'image composée gardée) et l'OSD
  // (posé à chaque présentation) à part.
  struct Built {
    std::span<const pl_overlay> all;
    std::span<const pl_overlay> subtitles;
    std::span<const pl_overlay> osd;
  };
  // ui_bottom, ui_top : pixels pris en bas et en haut par l'interface ; les sous-titres du bas
  // montent au-dessus, l'OSD descend dessous.
  std::expected<Built, Refusal> build(const subs::SubtitleFrame* subtitles,
                                      const std::span<const std::string>* osd, OsdText& text, OutputSize out,
                                      int ui_bottom, int ui_top);

 private:
  struct Layer {
    pl_tex tex = nullptr;
    const subs::SubtitleAtlas* atlas = nullptr;  // celui que tex contient
    std::uint64_t generation = 0;
    std::vector<pl_overlay_part> parts;
  };
  std::expected<bool, Refusal> add(Layer& layer, const subs::SubtitleFrame& frame, OutputSize out,
                                   int ui_bottom = 0);

  pl_gpu gpu_;
  Layer subtitles_;
  Layer osd_;
  std::array<pl_overlay, 2> overlays_{};
  int count_ = 0;
};

// Les poignées du ring que le backend TensorRT importe dans CUDA : HANDLE NT sous Windows, descripteurs
// ailleurs (placebo_ring.cpp les exporte, placebo_open.cpp vérifie que l'appareil les offre).
#ifdef _WIN32
inline constexpr pl_handle_type kRingHandle = PL_HANDLE_WIN32;
#else
inline constexpr pl_handle_type kRingHandle = PL_HANDLE_FD;
#endif

struct PlaceboScreen::Impl {
  ~Impl();
  void open_shader_cache(const std::filesystem::path& file);
  void save_shader_cache();
  std::expected<void, Refusal> compose(const pl_frame* image, const pl_frame& target) const;
  // La composition gardée (ShowRequest::redraws_expected) : vidéo et sous-titres composés dans `composed`
  // quand la clé change, puis recopiés dans la cible. false : la cible ne se prête pas à la copie, rien n'est
  // dessiné et la composition directe s'impose.
  std::expected<bool, Refusal> compose_kept(const pl_frame* image, const pl_frame& target,
                                            const ComposeKey& key, std::span<const pl_overlay> subtitles);
  // Les seuls calques (l'OSD) sur la cible, sans l'image (renderer.h : image NULL).
  std::expected<void, Refusal> draw_overlays(const pl_frame& target,
                                             std::span<const pl_overlay> layers) const;
  pl_tex composed = nullptr;
  std::optional<ComposeKey> composed_key;
  std::uint64_t recompositions = 0;  // compositions gardées refaites, puis recopies sans recomposition
  std::uint64_t reuses = 0;
  // L'interface par-dessus la composition ; un refus est écrit au journal une fois, la lecture
  // continue sans elle.
  void draw_ui(const ImDrawData* ui, const pl_frame& target);
  std::expected<void, Refusal> save_snapshot(const pl_frame* image, const pl_frame& target, OutputSize out,
                                             const ImDrawData* ui, const std::filesystem::path& path);
  // L'instantané demandé (--snapshot), numéroté s'il y en a plusieurs ; un refus est écrit au journal.
  void take_snapshot(const pl_frame* image, const pl_frame& target, OutputSize out,
                     const ImDrawData* drawn_ui);
  std::shared_ptr<spdlog::logger> channel;
  SDL_Window* window = nullptr;
  pl_log log = nullptr;
  pl_vk_inst inst = nullptr;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  pl_vulkan vk = nullptr;
  std::unique_ptr<VulkanShare> share;  // vk prêté à FFmpeg ; détruit avant lui
  pl_swapchain swapchain = nullptr;
  pl_renderer renderer = nullptr;
  pl_cache shader_cache = nullptr;                  // ScreenOptions::shader_cache (placebo_open.cpp)
  PresentLevel present_level = PresentLevel::Open;  // ScreenOptions::present (present_level.h)
  PresentCaps present_caps;
  std::unique_ptr<PresentFeedback> feedback;
  std::optional<render::DisplayContract> contract;  // 8.2 : le contrat d'écran relu
  std::unique_ptr<RingWatcher> watcher;  // --ring-watcher (ring_watcher.h), le temps d'une chaîne  // niveaux
                                         // Timing et Wait (placebo_feedback.h) ; nul sinon
  std::filesystem::path shader_cache_file;
  GpuDevice device;
  PlaceboRing ring;
  std::unique_ptr<Overlays> overlays;
  std::unique_ptr<OsdText> osd;
  std::unique_ptr<ImGuiLayer> ui;
  static constexpr int kMaxUiFailures = 5;
  int ui_failures = 0;     // échecs de suite du dessin de l'interface
  bool ui_failed = false;  // abandonnée après kMaxUiFailures échecs de suite
  pl_color_space source{};
  double sample_aspect = 1.0;  // SAR de la vidéo, pour la forme de l'image à l'écran
  OutputSize swapchain_size;
  // Le rendu hors écran des essais du bench (ScreenOptions::offscreen) : deux images en file, la cadence de
  // l'écran simulée.
  bool offscreen = false;
  bool hdr_wanted = true;   // ScreenOptions::hdr : la sortie hors écran en HDR10
  bool hdr_allowed = true;  // ScreenOptions::hdr (faux avec --sdr) : HDR10 demandable à la swapchain
  bool sdr_forced =
      false;  // sortie SDR (HDR absent du système, ou --sdr) : swapchain en SDR même sur un écran HDR
  OutputSize offscreen_size;
  std::array<pl_tex, 2> offscreen_tex{};
  std::size_t offscreen_index = 0;
  std::int64_t period_us = 16'667;
  std::int64_t next_tick_us = 0;
  // La cible du rendu hors écran (rend une image libre de la file), puis l'attente de la prochaine période.
  std::expected<pl_frame, Refusal> offscreen_target();
  std::int64_t offscreen_present();
  bool fullscreen = false;
  bool hdr_output = false;
  std::optional<std::filesystem::path> snapshot;
  int snapshot_count = 1;  // présentations à écrire, numérotées s'il y en a plusieurs
  int snapshot_index = 0;
  std::optional<render::Grade> grade;  // absent si les hooks ne se lisent pas (dit au journal)
  render::GradeChoice grade_choice;    // posé par set_grade
  // Posé par set_grade_after : le grade qui attend sa première image, et depuis quand.
  std::optional<render::GradeChoice> grade_next;
  std::int64_t grade_next_pts = 0;
  std::uint32_t grade_next_epoch = 0;
  std::int64_t grade_next_since_us = 0;
  VideoRate video_rate;
  // --hdr-metadata (ScreenOptions::kde_sdr_nits) : le maximum annoncé à la swapchain (0 : rien d'annoncé).
  std::optional<double> kde_sdr_nits;
  bool kde_sdr_auto = false;
  double hinted_max_nits = 0.0;
  double hinted_max_fall = 0.0;
  // La compression de KWin dans le grade et les métadonnées de la sortie, selon la cible et le grade.
  bool kwin_metadata() const;    // la passe « kwin » et les métadonnées HDR10 en jeu
  double sdr_reference() const;  // la luminosité SDR que la passe « kwin » imite (--hdr-metadata)
  std::expected<void, Refusal> submit(const pl_vulkan_present* present) const;  // placebo_screen.cpp
  void match_kwin();  // la cadence des images vidéo distinctes, pour les lissages du grade
  FinalScaler final_scaler = FinalScaler::Default;  // posés par set_finish
  bool deband = false;
  bool frame_debanded = false;  // l'image montrée a été débandée avant SPAN (RingFrame::debanded, P3.56)
  int sharpness = 0;            // posé par set_finish, 0 à 100 %
  std::optional<render::Sharpen> sharpen;  // absent si le hook ne se lit pas (dit au journal)
  // Augmenté à chaque changement de ce que le grade applique (app/play/compose_key.h) : paramètres, looks,
  // KWin, couleur de la source, nouvelle scène.
  std::uint64_t grade_serial = 0;
  std::optional<render::Grade::KwinMatch> kwin_applied;
  // Ce que le grade porte : rien, le grade PQ (vrai) ou le grade SDR (faux) de grade_choice.
  std::optional<bool> grade_loaded_pq;
  // set_grade_after : le grade en attente passe dans grade_choice si l'image à présenter vient après la
  // sienne.
  void take_next_grade(const RingFrame* frame);
  // Charge dans le grade le choix qui va à cette cible ; graded() dit s'il s'applique à une cible.
  void select_grade(const pl_color_space& target);
  bool graded(const pl_color_space& target) const;
  // Temps GPU des passes du grade (minuteries de libplacebo), par image composée avec lui ; le
  // résumé va au journal à la fermeture de l'écran.
  pl_color_space target_color{};  // celle de la dernière présentation
  // Mesure de la rampe du vrai noir : un renderer et un grade à part, pour ne toucher ni la
  // détection de pic ni l'état temporel du grade de la lecture.
  pl_renderer ramp_renderer = nullptr;
  std::optional<render::Grade> ramp_grade;
  // La dernière rampe mesurée : mêmes paramètres, mêmes marches, même sortie, la mesure (un rendu et une
  // relecture synchrones) n'est pas refaite.
  struct RampMeasure {
    render::GradeParams params;
    std::vector<float> levels;
    bool hdr = false;
    std::vector<double> nits;
    std::optional<render::CubeLut> looks;  // la pile de LUT mesurée avec la rampe (P3.61)
  };
  std::optional<RampMeasure> last_ramp;
  std::optional<render::CubeLut> ramp_looks;  // la pile posée sur ramp_grade
  mutable std::uint64_t grade_frame_ns = 0;
  mutable std::uint64_t display_frame_ns = 0;  // toutes les passes de l'affichage, de la dernière composition
  mutable std::vector<std::uint32_t> grade_gpu_us;
};

}  // namespace ajp::app::play
