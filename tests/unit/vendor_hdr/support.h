// Outils des tests vendor_hdr : dossiers temporaires, faux binaires ELF et PE, faux SDK.
// Tout est généré pendant le test : aucun binaire n'entre dans le dépôt.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vendor_hdr/sdk_version.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::vendor_hdr::test {

// Dossier temporaire unique, supprimé à la destruction.
class TempDir {
 public:
  TempDir();
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  ~TempDir();
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void write_file(const std::filesystem::path& file, std::string_view bytes);

// En-tête ELF64 minimal d'une bibliothèque partagée x86-64 (machine 62), ou ARM64 (183).
std::string elf_header(std::uint16_t machine = 62);

// DLL PE32+ minimale : machine AMD64 (0x8664) ou autre, section .rsrc avec ou sans
// VS_FIXEDFILEINFO portant version.
std::string pe_dll(std::optional<SdkVersion> version, std::uint16_t machine = 0x8664);

// En-têtes Maxine qui définissent chaque constante de maxine::kAbi à sa valeur attendue.
std::string abi_header_text();

enum class LibraryContent : std::uint8_t {
  ElfStub,  // en-tête ELF valide, impossible à charger : pour le magasin
  Empty,    // fichiers vides
  Fakes,    // vraies bibliothèques factices construites par les tests : pour les chargeurs
};

struct MaxineSdkOptions {
  LibraryContent libraries = LibraryContent::ElfStub;
  std::optional<std::string> version_header = "1.3.0.0";  // version.h à la racine ; rien sinon
  bool feature = true;                                    // fonction TrueHDR installée
  bool headers = true;
};

// Arborescence d'un SDK Maxine installé (/usr/local/VideoFX) sous root.
void make_maxine_sdk(const std::filesystem::path& root, const MaxineSdkOptions& options = {});

// Dossier d'un RTX Video SDK : la DLL d'entrée (de cette version) et nvngx_truehdr.dll.
void make_rtx_sdk(const std::filesystem::path& root, std::optional<SdkVersion> version);

// Flux gpu factice : note les appels, et peut échouer sur commande.
class RecordingStream final : public GpuStream {
 public:
  Done hand_to_legacy() override;
  Done take_from_legacy() override;
  std::vector<std::string> events;
  bool fail_hand = false;
};

// Tampon hôte tenant lieu de mémoire CUDA pour les faux SDK.
struct HostImage {
  HostImage(int width, int height, TrueHdrPixel pixel);
  TrueHdrImage view();
  std::vector<std::uint32_t> pixels;  // un mot de 4 octets par pixel
  int width;
  int height;
  TrueHdrPixel pixel;
};

}  // namespace ajp::vendor_hdr::test
