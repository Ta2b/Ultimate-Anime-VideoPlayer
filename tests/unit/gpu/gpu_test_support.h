// Aides des tests du module gpu : images décodées hôtes synthétiques, petits modèles ONNX
// construits dans le test, ring en mémoire hôte pour le backend CPU, puits des images.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/frame.h"
#include "core/messages.h"
#include "core/plan.h"
#include "core/ring.h"
#include "gpu/backend.h"
#include "gpu/output_ring.h"
#include "gpu/program.h"
#include "gpu_ir/onnx_builder.h"

namespace ajp::gpu::test {

// YUV420P 8 bits en mémoire hôte, chaque plan d'une seule valeur sauf retouche par plane().
class HostFrame : public FrameData {
 public:
  HostFrame(int width, int height, std::uint8_t luma, std::uint8_t cb = 128, std::uint8_t cr = 128)
      : width_(width), height_(height) {
    planes_[0].assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), std::byte{luma});
    planes_[1].assign(static_cast<std::size_t>(width / 2) * static_cast<std::size_t>(height / 2),
                      std::byte{cb});
    planes_[2].assign(planes_[1].size(), std::byte{cr});
  }
  FrameMemory memory() const override { return FrameMemory::Host; }
  int plane_count() const override { return 3; }
  std::span<const std::byte> host_plane(int plane) const override { return planes_[index(plane)]; }
  std::size_t host_stride(int plane) const override {
    return static_cast<std::size_t>(plane == 0 ? width_ : width_ / 2);
  }
  std::uintptr_t native_plane(int) const override { return 0; }
  std::size_t native_pitch(int) const override { return 0; }
  std::uintptr_t ready_event() const override { return 0; }
  std::uint64_t ready_value() const override { return 0; }
  std::vector<std::byte>& plane(int plane) { return planes_[index(plane)]; }

 private:
  static std::size_t index(int plane) { return static_cast<std::size_t>(plane); }
  int width_;
  int height_;
  std::array<std::vector<std::byte>, 3> planes_;
};

inline constexpr int kWidth = 64;
inline constexpr int kHeight = 36;
inline constexpr std::int64_t kFrameUs = 40'000;  // 25 im/s : des pts entiers à toutes les fractions

inline DecodedFrame make_frame(std::uint32_t id, std::uint8_t luma, std::uint32_t epoch = 0,
                               std::int64_t pts_us = -1, int width = kWidth, int height = kHeight) {
  return DecodedFrame{.id = SourceId{id},
                      .width = width,
                      .height = height,
                      .format = PixelFormat::Yuv420p,
                      .pts_us = pts_us >= 0 ? pts_us : static_cast<std::int64_t>(id) * kFrameUs,
                      .duration_us = kFrameUs,
                      .epoch = epoch,
                      .data = std::make_shared<HostFrame>(width, height, luma)};
}

// Dossier temporaire propre au test, effacé à la fin.
class TempDir {
 public:
  explicit TempDir(const std::string& name);
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  ~TempDir();
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

// Modèles ONNX des tests, entrées NCHW de hauteur et largeur libres :
//   agrandissement x2 façon SPAN : Conv 1×1 puis DepthToSpace (CRD), soit le plus proche voisin ;
//   RIFE au contrat d'entrée à 11 plans de kernels/spec/rife.h : a + t (b − a) ;
//   mouvement de S3 : flot nul (4 plans) et masque 0,5.
std::vector<std::byte> upscale_x2_model();
std::vector<std::byte> rife_blend_model();
std::vector<std::byte> motion_model();
std::filesystem::path write_model(const std::filesystem::path& dir, const std::string& name,
                                  const std::vector<std::byte>& bytes);
// Backend::build, comme prepare ; un artefact vide si le modèle est refusé.
Artifact build_artifact(Backend& backend, const std::filesystem::path& onnx, const std::string& id);
// build() puis load() sur le backend.
std::shared_ptr<Engine> load_engine(Backend& backend, const std::filesystem::path& onnx,
                                    const std::string& id);

// Slots hôtes du backend CPU (gpu/cpu/cpu_kernels.h) : image = adresse de la mémoire.
class HostRing {
 public:
  HostRing(std::size_t slots, int width, int height, std::uint32_t generation = 1);
  const std::vector<RingSlotDesc>& slots() const { return slots_; }
  std::span<const std::uint16_t> pixels(std::uint32_t slot) const { return memory_[slot]; }

 private:
  std::vector<std::vector<std::uint16_t>> memory_;
  std::vector<RingSlotDesc> slots_;
};

// Le thread main des tests : garde les images reçues ; refuse au-delà de `bound` en attente.
class CollectingSink final : public RingFrameSink {
 public:
  explicit CollectingSink(std::size_t bound = 64) : bound_(bound) {}
  bool try_push(RingFrame&& frame) override;
  std::vector<RingFrame> frames;                   // toutes, dans l'ordre de réception
  std::function<void(const RingFrame&)> on_frame;  // appelé à chaque réception
  std::size_t unread = 0;                          // reçues et pas encore rendues par free_all()
  // Rend chaque slot reçu non rendu, comme main après son rendu.
  std::vector<SlotFree> free_all();

 private:
  std::size_t bound_;
  std::size_t freed_ = 0;
};

// Plans des tests : sortie 2× la source 64×36 par le modèle « up2 ».
core::Plan upscale_plan();
core::Plan with_level(core::Plan plan, core::Ratio factor, core::Policy policy, core::Schedule schedule,
                      core::Pacing pacing = core::Pacing::Source, const std::string& model = "rife");

}  // namespace ajp::gpu::test
