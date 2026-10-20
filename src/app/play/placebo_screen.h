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

#include "app/play/links.h"
#include "core/frame.h"
#include "core/gpu_device.h"
#include "core/ring.h"
#include "core/types.h"
#include "render/grade_params.h"

struct SDL_Window;
struct ImGuiPlatformIO;

namespace ajp::app::play {

struct ScreenOptions {
  std::string title = "ajp";
  int width = 1280;
  int height = 720;
  bool fullscreen = false;
  bool hdr = true;  // HDR10 (PQ, BT.2020) si l'écran l'offre, SDR sinon
  // Smooth Motion du pilote NVIDIA : sa couche implicite VK_LAYER_NV_present s'active par
  // NVPRESENT_ENABLE_SMOOTH_MOTION=1, posé avant la création de l'instance Vulkan.
  bool smooth_motion = false;
  bool allow_software = false;  // appareil Vulkan logiciel (lavapipe) accepté : essais sans GPU
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
  // Ce que Vulkan dit de l'appareil pour l'écran de bienvenue : sa mémoire locale (le plus grand
  // tas DEVICE_LOCAL) et les matrices coopératives (les cœurs Tensor d'une RTX).
  struct DeviceFacts {
    std::string name;              // VkPhysicalDeviceProperties::deviceName
    std::uint32_t pci_vendor = 0;  // vendorID
    std::uint64_t vram_bytes = 0;
    bool cooperative_matrix = false;
  };
  DeviceFacts device_facts() const;
  SDL_Window* window() const;
  // Ce que l'écran présente : sa sortie (sRGB ou PQ) et son type, pour l'OSD.
  std::string output_description() const;

  // Crée le ring : count slots RGBA16 de width × height, exportés (mémoire et sémaphore
  // timeline) et cédés à CUDA en disposition GENERAL, prêts pour la première écriture.
  std::expected<std::vector<RingSlotDesc>, Refusal> create_ring(int width, int height, std::uint32_t count);
  // Couleur des images du ring : celle de la source (l'agrandissement ne la change pas).
  void set_source_color(const ColorInfo& color);
  // Grade (render/grade.h) posé sur OUTPUT : celui du PQ sur une sortie HDR10, celui du SDR sur une
  // sortie SDR BT.709 (sRGB, gamma 2,2 ou 2,4) ; rien : sans grade.
  void set_grade(const render::GradeChoice& choice);
  bool hdr_output() const;  // la dernière présentation était en PQ
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
  void set_fullscreen(bool on) override;
  bool fullscreen() const override;

 private:
  friend class FrameExport;  // l'export rend les slots du ring avec l'appareil de l'écran
  struct Impl;
  explicit PlaceboScreen(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::app::play
