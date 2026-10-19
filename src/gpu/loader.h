// Chargeur des backends : une bibliothèque ajp_backend_<nom> (.so, .dll) par moteur, dans le
// dossier des moteurs installés par l'utilisateur (app le passe). Le chargeur lit l'ABI avant
// de créer le backend, le détruit par RAII avant de fermer la bibliothèque, et liste les
// backends d'un dossier pour le panneau des capacités.
#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/rife_input.h"
#include "vendor_hdr/dynamic_library.h"

namespace ajp::gpu {

// « ajp_backend_vulkan.so » sous Linux et macOS, « ajp_backend_vulkan.dll » sous Windows.
std::string backend_file_name(std::string_view name);

// Un backend créé dans sa bibliothèque. Le backend est détruit par ajp_backend_destroy de
// sa propre bibliothèque, avant que celle-ci se ferme.
class LoadedBackend {
 public:
  // Ouvre dir/ajp_backend_<name>, vérifie ajp_backend_abi() et ajp_backend_layout()
  // (check_backend_abi), puis crée le backend avec init.
  // Refus : nom hors de [a-z0-9_] (FieldOutOfRange), bibliothèque ou symbole absents
  // (LibraryUnavailable), autre ABI (AbiMismatch), ou le refus que rend ajp_backend_create
  // (DeviceMismatch quand l'appareil de rendu ne lui convient pas…).
  static std::expected<LoadedBackend, Refusal> load(const std::filesystem::path& dir, std::string_view name,
                                                    const BackendInit& init);
  static std::expected<LoadedBackend, Refusal> load_file(const std::filesystem::path& path,
                                                         const BackendInit& init);

  Backend& backend() const { return *backend_; }
  // Noyaux d'entrée de RIFE (rife_input.h), nullptr si la bibliothèque ne les exporte pas.
  RifeInputOps* rife_input() const { return rife_input_; }
  const std::filesystem::path& path() const { return library_.path(); }

 private:
  using Destroy = BackendDestroyFn;

  LoadedBackend(vendor_hdr::DynamicLibrary library, Backend* backend, Destroy destroy, RifeInputOps* rife)
      : library_(std::move(library)), backend_(backend, destroy), rife_input_(rife) {}

  vendor_hdr::DynamicLibrary library_;  // premier membre : fermée après la destruction du backend
  std::unique_ptr<Backend, Destroy> backend_;
  RifeInputOps* rife_input_ = nullptr;
};

// Un backend présent dans un dossier, ouvert le temps de lire son ABI.
struct BackendListing {
  std::string name;  // « cpu », « vulkan », « tensorrt »…
  std::filesystem::path path;
  std::expected<int, Refusal> abi;      // ajp_backend_abi(), ou pourquoi il n'a pas été lu
  std::expected<void, Refusal> usable;  // check_backend_abi() : version et empreinte binaire
};

// Les fichiers ajp_backend_<nom> de dir, triés par nom ; aucun backend n'est créé.
// Refus : dossier absent ou illisible (FileUnreadable).
std::expected<std::vector<BackendListing>, Refusal> list_backends(const std::filesystem::path& dir);

}  // namespace ajp::gpu
