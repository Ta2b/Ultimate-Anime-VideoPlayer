// Une bibliothèque de backend chargée à l'exécution (dlopen, LoadLibraryExW), ses points d'entrée vérifiés
// (check_backend_abi), et les instances qu'elle crée.
//
// Provisoire : le chargeur du module gpu (pc/gpu, loader.h) prend sa place à la fusion.
#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

#include "core/types.h"
#include "gpu/backend.h"
#include "vendor_hdr/dynamic_library.h"

namespace ajp::app::play {

class BackendLibrary {
 public:
  // Le premier fichier présent de candidates. Refus : aucun (LibraryUnavailable), symbole
  // absent, ABI ou empreinte différentes (AbiMismatch).
  static std::expected<std::shared_ptr<BackendLibrary>, Refusal> open(
      std::span<const std::filesystem::path> candidates);
  ~BackendLibrary();
  BackendLibrary(const BackendLibrary&) = delete;
  BackendLibrary& operator=(const BackendLibrary&) = delete;

  const std::filesystem::path& path() const { return path_; }

  struct Destroy {
    gpu::BackendDestroyFn destroy = nullptr;
    std::shared_ptr<BackendLibrary> library;  // la bibliothèque survit à ses instances
    void operator()(gpu::Backend* backend) const;
  };
  using Instance = std::unique_ptr<gpu::Backend, Destroy>;

  // Le premier backend de candidates qui accepte l'appareil de rendu de init : TensorRT sur une
  // carte NVIDIA, Vulkan ailleurs (AMD, Intel). Chaque candidat est ouvert puis essayé par une
  // instance aussitôt détruite ; un candidat absent, d'une autre ABI ou qui refuse l'appareil
  // laisse sa place au suivant. Refus : LibraryUnavailable, avec la raison de chaque candidat.
  static std::expected<std::shared_ptr<BackendLibrary>, Refusal> choose(
      std::span<const std::filesystem::path> candidates, const gpu::BackendInit& init);

  // Une instance de plus, pour un seul thread. Refus : celui du backend.
  static std::expected<Instance, Refusal> create(const std::shared_ptr<BackendLibrary>& library,
                                                 const gpu::BackendInit& init);

 private:
  BackendLibrary() = default;
  std::optional<vendor_hdr::DynamicLibrary> handle_;
  std::filesystem::path path_;
  gpu::BackendCreateFn create_ = nullptr;
  gpu::BackendDestroyFn destroy_ = nullptr;
};

}  // namespace ajp::app::play
