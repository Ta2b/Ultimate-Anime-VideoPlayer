// L'écran provisoire de la tranche verticale : fenêtre SDL3, appareil Vulkan et swapchain
// libplacebo, ring de sortie exporté vers CUDA, composition des slots avec sous-titres et OSD.
//
// Provisoire : render (appareil, ring, composition, grade, ImGui) et present (fenêtre,
// swapchain, cadence, écran) le remplacent dès qu'ils sont fusionnés ; le contrat qu'il tient
// est celui de Screen (links.h) et du ring (core/ring.h), le même qu'eux.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "app/play/final_scaler.h"
#include "app/play/grade_taste.h"
#include "app/play/links.h"
#include "app/play/present_level.h"
#include "core/frame.h"
#include "core/gpu_device.h"
#include "core/ring.h"
#include "core/types.h"
#include "media/hw_api.h"
#include "media/shot_cutter.h"
#include "render/cube_lut.h"
#include "render/display_model.h"
#include "render/grade_params.h"

struct SDL_Window;
struct ImGuiPlatformIO;
struct AVBufferRef;

namespace ajp::app::play {

// Images que la file de l'écran (FIFO) garde au plus : deux suffisent à ne pas manquer de vsync
// quand le rendu tient dans la période, et chacune de plus retarde l'image d'une période.
inline constexpr int kSwapchainDepth = 2;

class LiveSettings;

struct ScreenOptions {
  std::string title = "ajp";
  int width = 1280;
  int height = 720;
  bool fullscreen = false;
  bool hidden = false;  // fenêtre minuscule sans bordure ni focus (--window pour l'évaluation sans --hidden)
  // Les essais du bench (--hidden) : aucune fenêtre visible. Chaque image se rend en entier dans une image
  // hors écran de offscreen_width × offscreen_height (la taille de l'écran), à la cadence de l'écran, avec
  // deux images en file comme la swapchain : la même charge du GPU que la lecture, sans rien montrer.
  bool offscreen = false;
  int offscreen_width = 1920;
  int offscreen_height = 1080;
  bool hdr = true;  // HDR10 (PQ, BT.2020) si l'écran l'offre, SDR sinon
  // --hdr-metadata : la luminosité SDR de KDE (nits). La sortie HDR10 annonce alors son vrai maximum, et le
  // grade fait lui-même la compression que KWin faisait sans métadonnées (render::Grade::set_kwin_match).
  std::optional<double> kde_sdr_nits;
  bool kde_sdr_auto = false;  // la luminosité SDR du contrat d'écran (KDE), kde_sdr_nits en repli
  // --hdr-contract-v2-curve : les métadonnées du contrat d'écran (set_contract) annoncées à la swapchain,
  // sans kde_sdr_nits ; metadata_off : rien d'annoncé (--hdr-metadata=off).
  bool contract_v2 = false;
  bool metadata_off = false;
  // Smooth Motion du pilote NVIDIA : sa couche implicite VK_LAYER_NV_present s'active par
  // NVPRESENT_ENABLE_SMOOTH_MOTION=1, posé avant la création de l'instance Vulkan.
  bool smooth_motion = false;
  bool allow_software = false;  // appareil Vulkan logiciel (lavapipe) accepté : essais sans GPU
  bool hdr_anime = false;       // --hdr-anime : render::GradeBody::Anime
  bool anime_guided = false;    // --anime-guided : render::GradeBody::AnimeGuided (méga plan A3)
  bool legacy_grade = false;    // --legacy-grade : le grade d'avant (render::GradeBody::Legacy)
  bool split_look = false;      // --legacy-grade=split : render::GradeBody::FusedSplit
  bool grade_deband = false;    // --grade-deband : débandage devant le grade (coupé par défaut)
  bool skin_mask = false;       // --skin-mask : le masque peau du grade rend 0 (essai)
  bool grade_ring = false;      // --grade-ring : le grade fait dans l'anneau (set_ring_live)
  // --present : le retour de présentation (chantier 6, present_level.h) ; open, celle d'aujourd'hui.
  PresentWanted present = PresentWanted::Open;
  // Le cache des shaders et pipelines de libplacebo (le grade, ses passes, les LUT) : relu à l'ouverture,
  // gardé à la fermeture. Sans lui, tout se recompile à chaque lancement. Vide : pas de cache.
  std::filesystem::path shader_cache;
  // --stats avec le retour wp_presentation (--present, niveau Wayland) : <stats>.screen.tsv, les horodatages
  // du compositeur (0.7). Vide : rien.
  std::filesystem::path screen_stamps;
};

class PlaceboScreen final : public Screen {
 public:
  // SDL doit être initialisé (vidéo). Refus : fenêtre, instance, appareil ou swapchain
  // impossibles (DeviceMismatch, LibraryUnavailable).
  static std::expected<std::unique_ptr<PlaceboScreen>, Refusal> open(const ScreenOptions& options);
  ~PlaceboScreen() override;
  PlaceboScreen(const PlaceboScreen&) = delete;
  PlaceboScreen& operator=(const PlaceboScreen&) = delete;

  // L'appareil de rendu : son UUID choisit le GPU de CUDA et de NVDEC.
  const GpuDevice& device() const;
  // L'appareil Vulkan prêté à FFmpeg (vulkan_share.h) : le décodeur Vulkan Video y décode des
  // images que le backend Vulkan lit en place ; nullptr si FFmpeg ne l'a pas accepté.
  const media::HwDeviceSource* shared_devices() const;
  // Le même, en AVHWDeviceContext emprunté, pour l'encodeur de l'export ; nullptr sans partage.
  AVBufferRef* vulkan_device() const;
  // Ce que Vulkan dit de l'appareil pour l'écran de bienvenue : sa mémoire locale (le plus grand
  // tas DEVICE_LOCAL) et les matrices coopératives (les cœurs Tensor d'une RTX).
  struct DeviceFacts {
    std::string name;              // VkPhysicalDeviceProperties::deviceName
    std::uint32_t pci_vendor = 0;  // vendorID
    std::string driver;            // driverVersion, en décimal
    std::uint64_t vram_bytes = 0;
    // Le plus grand tas visible de l'hôte parmi ceux de la VRAM : ~256 Mio sans Resizable BAR,
    // la VRAM entière avec ; 0 si aucun type DEVICE_LOCAL | HOST_VISIBLE.
    std::uint64_t host_visible_vram_bytes = 0;
    bool cooperative_matrix = false;
  };
  DeviceFacts device_facts() const;
  SDL_Window* window() const;
  // Ce que l'écran présente : sa sortie (sRGB ou PQ) et son type, pour l'OSD.
  std::string output_description() const;

  // Crée le ring : count slots RGBA16 (ou RGB10A2, si format le demande et que l'appareil le prend) de
  // width × height, exportés (mémoire et sémaphore timeline) et cédés à CUDA en disposition GENERAL,
  // prêts pour la première écriture.
  std::expected<std::vector<RingSlotDesc>, Refusal> create_ring(int width, int height, std::uint32_t count,
                                                                bool external = true,
                                                                RingFormat format = RingFormat::Rgba16Unorm);
  // Couleur des images du ring : celle de la source (l'agrandissement ne la change pas).
  void set_source_color(const ColorInfo& color);
  // Forme du pixel de la vidéo (SAR, largeur / hauteur) : l'image s'affiche étirée de ce rapport.
  void set_sample_aspect(double sar);
  // Grade (render/grade.h) posé sur OUTPUT : celui du PQ sur une sortie HDR10, celui du SDR sur une
  // sortie SDR BT.709 (sRGB, gamma 2,2 ou 2,4) ; rien : sans grade.
  void set_grade(const render::GradeChoice& choice);
  // Le même, à partir de la première image présentée après after (pts et époque de la dernière image reçue
  // du GPU) : le grade adaptatif change avec le plan de l'image que le GPU vient de traiter, et les images
  // déjà prêtes devant elle, encore de la scène d'avant, gardent l'ancien grade. Repli au bout de 250 ms
  // (pause, seek).
  void set_grade_after(const render::GradeChoice& choice, const RingFrame& after);
  // Les LUT du look (render/grade.h : set_looks) ; nullptr : aucune. Refus si libplacebo ne lit pas la
  // texture.
  std::expected<void, Refusal> set_looks(const std::array<const render::CubeLut*, render::kLookSlots>& luts);
  // Composition finale : filtre qui étire vers la fenêtre, débandage ; sans effet sur l'export.
  void set_finish(FinalScaler scaler, bool deband);
  bool hdr_output() const;  // la dernière présentation était en PQ
  // Sortie SDR même sur un écran HDR (la méthode HDR « SDR ») : la swapchain change d'espace de couleur à
  // la présentation suivante, sans réouverture. Faux : HDR10 si l'écran en reçoit et que --sdr ne l'a pas
  // coupé.
  void set_sdr_output(bool sdr);
  // Vrai noir OLED : ces niveaux (0 à 1, dans la couleur des images du ring : SDR, ou PQ après
  // TrueHDR) passent par libplacebo et le grade en cours sans le vrai noir ; rend leurs nits en
  // sortie (PQ, ou SDR à son blanc). Rendu à part (renderer et grade propres), relu tout de suite.
  std::expected<std::vector<double>, Refusal> grade_ramp(std::span<const float> levels);
  // Soumet le travail en attente (rendus, cessions de slots) sans l'attendre.
  void flush();
  // La prochaine présentation est aussi rendue en SDR hors écran et écrite en PPM (P6) : ce
  // que montre la fenêtre, pour le vérifier sans capture d'écran.
  // count > 1 : autant de présentations de suite, « nom-001.pq.ppm », « nom-002.pq.ppm »…
  void snapshot_next(std::filesystem::path path, int count = 1);
  // Détruit les textures de l'interface (placebo_imgui.h) : avant l'écran, contexte ImGui vivant.
  void release_ui(ImGuiPlatformIO& platform);

  OutputSize size() const override;
  std::expected<ScreenTiming, Refusal> show(const ShowRequest& request) override;
  SlotFree retire(const RingFrame& frame) override;
  bool ready(const RingFrame& frame) const override;
  void watch(const RingFrame& frame) override;
  std::optional<std::int64_t> written_at(const RingFrame& frame) override;
  bool wakes_on_write() const override;
  std::optional<core::VblankGrid> vblank_grid() const override;
  // Le VRR tel que present timing le dit (6.10) : armé, non armé, ou rien (niveau sans cette lecture).
  std::optional<bool> vrr_armed() const;
  // Vrai une fois quand cet état vient de changer : main le relit sans attendre son tour.
  bool vrr_changed();
  // Le veilleur d'écriture le temps d'une chaîne (--ring-watcher) : waker réveille main ; nullptr l'arrête.
  void watch_writes(void (*waker)(void*), void* context);
  // Le contrat d'écran relu (8.2, display_contract.h) : gardé pour le grade et la swapchain du contrat v2.
  void set_contract(const render::DisplayContract& contract);
  // Les plans de la pré-analyse, pour le grade v2 dans l'anneau (8.6) ; rien hors de cet essai.
  void set_plan_shots(std::vector<media::Shot> shots);
  // Le goût appris du grade v2 (8.10, grade_taste.h) ; rien hors de l'essai du grade v2 dans l'anneau.
  void set_taste(const GradeTaste& taste);
  const std::optional<render::DisplayContract>& contract() const;
  void new_scene() override;
  // Le grade de l'anneau (chantier 2c, --grade-ring) : les versions du grade sont postées dans `live` quand
  // la sortie est en PQ et le grade en trois passes ; l'image gradée par le GPU du ring est montrée sans
  // hooks. nullptr : rien n'est posté, l'écran grade.
  void set_ring_live(std::shared_ptr<LiveSettings> live);
  bool stale_grade(const RingFrame& frame) const override;
  void set_fullscreen(bool on) override;
  bool fullscreen() const override;

 private:
  friend class FrameExport;  // l'export rend les slots du ring avec l'appareil de l'écran
  struct Impl;
  explicit PlaceboScreen(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Le veilleur d'écriture le temps d'une portée (run_chain) : waker nul, aucun veilleur.
class WriteWatch {
 public:
  WriteWatch(PlaceboScreen& screen, void (*waker)(void*), void* context) : screen_(screen) {
    screen_.watch_writes(waker, context);
  }
  ~WriteWatch() { screen_.watch_writes(nullptr, nullptr); }
  WriteWatch(const WriteWatch&) = delete;
  WriteWatch& operator=(const WriteWatch&) = delete;

 private:
  PlaceboScreen& screen_;
};

}  // namespace ajp::app::play
