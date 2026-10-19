// Le vrai SDK Maxine VFX installé sur la machine (AJP_MAXINE_SDK_DIR, /usr/local/VideoFX par
// défaut), lu comme le magasin le lit : sans GPU, mais SKIP sans SDK (la CI n'en a pas).
// Fige ce qui a été vérifié le 19 octobre 2026 sur VFXSDK_linux 1.3.0.0 et nvvfxtruehdr 1.3.0.0
// (rapport ~/ajp-reports/truehdr.md) : constantes de maxine_abi.h, marqueurs de version, jeu
// d'exécution de sdk_layout et copie réduite du magasin.
#include "real_sdk.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>

#include <catch2/catch_test_macros.hpp>

#include "support.h"
#include "vendor_hdr/maxine_abi_check.h"
#include "vendor_hdr/sdk_layout.h"
#include "vendor_hdr/sdk_store.h"
#include "vendor_hdr/sdk_version.h"

using namespace ajp::vendor_hdr;
using namespace ajp::vendor_hdr::test;
namespace fs = std::filesystem;

namespace {

std::uintmax_t tree_bytes(const fs::path& root) {
  std::uintmax_t total = 0;
  for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file() && !entry.is_symlink()) total += entry.file_size();
  }
  return total;
}

}  // namespace

TEST_CASE("SDK Maxine installé : constantes, version et fichiers", "[nvidia][sdk]") {
  const fs::path sdk = require_maxine_sdk();
  // Toutes les constantes de maxine_abi.h, dont le sélecteur du pic (« Luminance », pas
  // « MaxLuminance » comme supposé en vague 1).
  CHECK(check_maxine_abi(sdk));
  CHECK(read_sdk_version(SdkKind::MaxineVfx, sdk)->text() == "1.3.0.0");
  const auto inspected = inspect_sdk(SdkKind::MaxineVfx, sdk);
  REQUIRE(inspected);
  CHECK(inspected->text() == "1.3.0.0");
  // Le modèle NGX de la fonction, trouvé par préfixe.
  const RequiredFile ngx{.path = "features/nvvfxtruehdr/lib/libnvidia-ngx-truehdr.so", .prefix = true};
  const std::optional<fs::path> located = locate(sdk, ngx);
  REQUIRE(located);
  CHECK(located->filename().string().starts_with("libnvidia-ngx-truehdr.so.1."));
}

TEST_CASE("SDK Maxine installé : importé dans le magasin, copie réduite", "[nvidia][sdk]") {
  const fs::path sdk = require_maxine_sdk();
  TempDir data;
  SdkStore store(data.path(), SdkKind::MaxineVfx);
  auto imported = store.import(sdk);
  REQUIRE(imported);
  CHECK(imported->state == SdkState::Ready);
  REQUIRE(imported->installed);
  const fs::path copy = imported->installed->root();
  CHECK(imported->installed->version().text() == "1.3.0.0");

  // Chaque bibliothèque du jeu d'exécution est là, lien et cible.
  for (const std::string_view runtime : maxine_runtime()) {
    std::error_code error;
    CHECK(fs::is_regular_file(copy / runtime, error));
  }
  // Ce que TrueHDR ne charge pas reste dans le SDK de l'utilisateur.
  CHECK_FALSE(fs::exists(copy / "external/tensorrt"));
  CHECK_FALSE(fs::exists(fs::symlink_status(copy / "external/cuda/lib/libcublasLt.so.12")));
  // Environ 135 Mo au lieu des 4,3 Go du SDK (le Manager actuel copie tout external/).
  CHECK(tree_bytes(copy) < std::uintmax_t{300} * 1024 * 1024);
  // Revérifiée telle quelle au démarrage suivant.
  CHECK(SdkStore(data.path(), SdkKind::MaxineVfx).status().state == SdkState::Ready);
}
