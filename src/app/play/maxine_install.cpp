#include "app/play/maxine_install.h"

#include <chrono>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <system_error>
#include <vector>

#include "app/play/engine_tools.h"
#include "vendor_hdr/sdk_store.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Fichiers secrets (clé, jeton) effacés à la sortie de portée, comme le dossier d'extraction.
class Scratch {
 public:
  explicit Scratch(std::filesystem::path dir) : dir_(std::move(dir)) {}
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  Scratch(Scratch&&) = delete;
  Scratch& operator=(Scratch&&) = delete;
  ~Scratch() {
    std::error_code error;
    std::filesystem::remove_all(dir_, error);
  }
  const std::filesystem::path& dir() const { return dir_; }

  // Écrit text dans name, lisible du seul utilisateur.
  std::expected<std::filesystem::path, Refusal> secret(std::string_view name, std::string_view text) const {
    const std::filesystem::path path = dir_ / name;
    {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) return refuse(RefusalCode::FileUnwritable, path.string());
    }
    std::error_code error;
    std::filesystem::permissions(
        path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write, error);
    if (error)
      return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", path.string(), error.message()));
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    if (!out) return refuse(RefusalCode::FileUnwritable, path.string());
    return path;
  }

 private:
  std::filesystem::path dir_;
};

// La valeur de "token" dans la réponse JSON du service d'authentification NGC.
std::optional<std::string> token_of(const std::filesystem::path& response) {
  std::ifstream in(response, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  const std::size_t key = text.find("\"token\"");
  if (key == std::string::npos) return std::nullopt;
  const std::size_t open = text.find('"', text.find(':', key) + 1);
  const std::size_t close = open == std::string::npos ? open : text.find('"', open + 1);
  if (close == std::string::npos || close == open + 1) return std::nullopt;
  return text.substr(open + 1, close - open - 1);
}

// Le dossier du SDK extrait : celui qui contient features/install_feature.sh, à deux niveaux au
// plus (l'archive range tout sous VideoFX/).
std::optional<std::filesystem::path> sdk_root(const std::filesystem::path& extracted) {
  std::error_code error;
  const auto holds = [&](const std::filesystem::path& dir) {
    return std::filesystem::is_regular_file(dir / "features" / "install_feature.sh", error);
  };
  if (holds(extracted)) return extracted;
  for (const auto& child : std::filesystem::directory_iterator(extracted, error)) {
    if (!child.is_directory(error)) continue;
    if (holds(child.path())) return child.path();
    for (const auto& grandchild : std::filesystem::directory_iterator(child.path(), error)) {
      if (grandchild.is_directory(error) && holds(grandchild.path())) return grandchild.path();
    }
  }
  return std::nullopt;
}

}  // namespace

std::expected<void, Refusal> install_maxine(const EngineCatalog& catalog, const core::NgcSdkSource& source,
                                            const std::string& api_key, const std::filesystem::path& sdk_dir,
                                            const InstallProgress& progress) {
  const auto curl = system_tool("curl");
  const auto tar = system_tool("tar");
  const auto bash = system_tool("bash");
  if (!curl || !tar || !bash)
    return refuse(RefusalCode::LibraryUnavailable, "curl, tar ou bash introuvables");
  if (api_key.empty() || api_key.find_first_of("\n\r ") != std::string::npos)
    return refuse(RefusalCode::FieldOutOfRange, "clé d'API NGC vide ou mal collée");
  const auto report = [&](float fraction, const std::string& stage) {
    if (progress.report) progress.report(fraction, stage);
  };
  std::error_code error;
  const Scratch scratch(catalog.layout.downloads / "maxine-vfx");
  std::filesystem::remove_all(scratch.dir(), error);
  std::filesystem::create_directories(scratch.dir() / "sdk", error);
  if (error)
    return refuse(RefusalCode::FileUnwritable,
                  std::format("{} : {}", scratch.dir().string(), error.message()));

  // 1. Le jeton NGC, par la clé de l'utilisateur ($oauthtoken est le nom d'utilisateur fixe de NGC).
  report(0.0F, "connexion à NVIDIA NGC");
  auto netrc = scratch.secret(
      "netrc", std::format("machine authn.nvidia.com login $oauthtoken password {}\n", api_key));
  if (!netrc) return std::unexpected(netrc.error());
  const std::filesystem::path response = scratch.dir() / "token.json";
  if (auto ok =
          run_tool({.program = *curl,
                    .args = {"--fail", "--silent", "--show-error", "--netrc-file", netrc->string(),
                             "--output", response.string(),
                             std::format("https://authn.nvidia.com/token?service=ngc&scope=group/ngc:{}&"
                                         "group/ngc:{}/{}",
                                         source.org, source.org, source.team)},
                    .timeout = std::chrono::minutes(2)},
                   {}, progress.stop, "jeton NGC (clé refusée ?)");
      !ok) {
    return std::unexpected(ok.error());
  }
  const std::optional<std::string> token = token_of(response);
  if (!token) return refuse(RefusalCode::DecodeFailed, "NGC n'a pas rendu de jeton : clé d'API refusée");
  auto header = scratch.secret("header", std::format("Authorization: Bearer {}\n", *token));
  if (!header) return std::unexpected(header.error());

  // 2. Le cœur du SDK : l'API des ressources de NGC renvoie vers un lien signé, suivi sans l'en-tête.
  const std::filesystem::path archive = scratch.dir() / source.file;
  if (auto ok = run_tool(
          {.program = *curl,
           .args = {"--fail", "--location", "--retry", "3", "--progress-bar", "--header",
                    "@" + header->string(), "--output", archive.string(),
                    std::format(
                        "https://api.ngc.nvidia.com/v2/org/{}/team/{}/resources/{}/versions/{}/files/{}",
                        source.org, source.team, source.resource, source.version, source.file)},
           .timeout = std::chrono::hours(2)},
          [&](std::string_view line) {
            if (const auto f = curl_percent(line)) report(0.05F + *f * 0.5F, "téléchargement du SDK Maxine");
          },
          progress.stop, "téléchargement du SDK Maxine");
      !ok) {
    return std::unexpected(ok.error());
  }

  // 3. Extraction (xz ou gzip, que tar reconnaît seul), puis la fonction TrueHDR par le script du SDK.
  report(0.55F, "extraction du SDK Maxine");
  if (auto ok = run_tool({.program = *tar,
                          .args = {"-xf", archive.string(), "-C", (scratch.dir() / "sdk").string()},
                          .timeout = std::chrono::minutes(30)},
                         {}, progress.stop, "extraction du SDK Maxine");
      !ok) {
    return std::unexpected(ok.error());
  }
  std::filesystem::remove(archive, error);
  const std::optional<std::filesystem::path> root = sdk_root(scratch.dir() / "sdk");
  if (!root) return refuse(RefusalCode::SdkRejected, "archive Maxine sans features/install_feature.sh");
  report(0.6F, "installation de TrueHDR (NGC)");
  if (auto ok = run_tool({.program = *bash,
                          .args = {"./install_feature.sh", "--feature", source.feature, "--ngc-org",
                                   source.org, "--ngc-team", source.team},
                          .timeout = std::chrono::hours(1),
                          .working_dir = *root / "features",
                          .extra_env = {"NGC_CLI_API_KEY=" + api_key}},
                         {}, progress.stop, "install_feature.sh");
      !ok) {
    return std::unexpected(ok.error());
  }

  // 4. L'import vérifié : fichiers exigés, architecture, version, ABI.
  report(0.9F, "vérification et import du SDK Maxine");
  vendor_hdr::SdkStore store(sdk_dir, vendor_hdr::SdkKind::MaxineVfx);
  auto imported = store.import(*root, progress.stop);
  if (!imported) return std::unexpected(imported.error());
  if (!imported->installed) return refuse(RefusalCode::SdkRejected, imported->reason);
  report(1.0F, "RTX HDR installé");
  return {};
}

}  // namespace ajp::app::play
