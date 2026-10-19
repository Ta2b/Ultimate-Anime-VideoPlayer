// Une bibliothèque de backend chargée par dlopen, ses points d'entrée vérifiés
// (check_backend_abi), et les instances qu'elle crée.
//
// Provisoire : le chargeur du module gpu (pc/gpu, loader.h) prend sa place à la fusion.
#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <span>

#include "core/types.h"
#include "gpu/backend.h"

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

  // Une instance de plus, pour un seul thread. Refus : celui du backend.
  static std::expected<Instance, Refusal> create(const std::shared_ptr<BackendLibrary>& library,
                                                 const gpu::BackendInit& init);

 private:
  BackendLibrary() = default;
  void* handle_ = nullptr;
  std::filesystem::path path_;
  gpu::BackendCreateFn create_ = nullptr;
  gpu::BackendDestroyFn destroy_ = nullptr;
};

}  // namespace ajp::app::play
