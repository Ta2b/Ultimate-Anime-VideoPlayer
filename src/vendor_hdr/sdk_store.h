// Magasin des SDK TrueHDR : l'utilisateur choisit le dossier d'un SDK, le magasin le vérifie,
// le copie dans le dossier de données et le choisit d'un coup ; avant tout chargement, il
// revérifie la copie (fichiers, architecture, version, ABI) et dit pourquoi elle ne convient pas.
//
// Porté de manager-v2 Services/TrueHdrSdkStore.cs : copies immuables, choix atomique par un
// manifeste écrit puis renommé, un refus ne touche jamais le SDK déjà choisi. Différences : on
// importe un dossier déjà extrait, pas une archive, et la version du SDK est vérifiée.
//
// Disposition, sous <données>/optional-sdks/<rtx-video|maxine-vfx>/ :
//   current.toml              manifeste du SDK choisi
//   versions/<version>-<n>/   copie d'un import, jamais modifiée ensuite
//   .import-<n>/              copie en cours, supprimée en cas d'échec ou d'annulation
//
// Un import Maxine copie environ 135 Mo (seules les bibliothèques que TrueHDR charge, voir
// sdk_layout.cpp) : l'appelant le lance hors du thread main et ne lance qu'un import à la fois
// par SDK.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <stop_token>
#include <string>

#include "core/types.h"
#include "vendor_hdr/sdk_kind.h"
#include "vendor_hdr/sdk_version.h"

namespace ajp::vendor_hdr {

enum class SdkState : std::uint8_t {
  NotInstalled,  // rien d'importé
  Ready,         // importé, vérifié, version prise en charge
  Damaged,       // copie modifiée ou incomplète, manifeste illisible : réimporter
  WrongVersion,  // version du SDK ou format du manifeste que ce lecteur ne prend pas en charge
};

// Un SDK vérifié par le magasin : la seule chose que les chargeurs acceptent.
class InstalledSdk {
 public:
  SdkKind kind() const { return kind_; }
  const std::filesystem::path& root() const { return root_; }
  const SdkVersion& version() const { return version_; }

 private:
  friend class SdkStore;
  InstalledSdk(SdkKind kind, std::filesystem::path root, SdkVersion version)
      : kind_(kind), root_(std::move(root)), version_(version) {}

  SdkKind kind_;
  std::filesystem::path root_;
  SdkVersion version_;
};

struct SdkStatus {
  SdkKind kind = SdkKind::RtxVideo;
  SdkState state = SdkState::NotInstalled;
  std::string reason;                     // phrase française ; vide quand Ready
  std::optional<InstalledSdk> installed;  // Ready seulement
};

// Vérifie un dossier sans rien copier : fichiers exigés, format des bibliothèques, version prise
// en charge, et pour Maxine l'ABI de ses en-têtes. Rend la version lue.
std::expected<SdkVersion, Refusal> inspect_sdk(SdkKind kind, const std::filesystem::path& folder);

class SdkStore {
 public:
  SdkStore(const std::filesystem::path& data_directory, SdkKind kind);

  SdkKind kind() const { return kind_; }

  // Relit le manifeste et revérifie la copie : à appeler juste avant de charger le SDK.
  SdkStatus status() const;

  // Vérifie folder, le copie, revérifie la copie puis la choisit. En cas de refus ou
  // d'annulation, le SDK choisi auparavant reste en place.
  std::expected<SdkStatus, Refusal> import(const std::filesystem::path& folder, std::stop_token stop = {});

 private:
  std::filesystem::path root_;
  SdkKind kind_;
};

}  // namespace ajp::vendor_hdr
