// Repris de animejanai-inference-rtx@118c272 src/cache_io.h:9-45,
// src/cache_stage_owner.h:8-45 et src/model_identity.cpp:33-41.
// Le nom et le contenu du marqueur restent ceux du fork : un dossier orphelin
// laissé par le fork se récupère ici, et inversement.
#include "prepare/cache_io.h"

#include <cerrno>
#include <format>
#include <fstream>
#include <random>
#include <tuple>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace ajp::prepare {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kOwnerFile = ".aji-stage-owner";
constexpr std::string_view kOwnerVersion = "aji-stage-owner-v1";

std::string random_suffix() {
  std::random_device random;
  return std::format("{}-{}", random(), random());
}

std::expected<void, Refusal> mark_stage(const fs::path& folder, std::string_view owner) {
  const fs::path marker = folder / kOwnerFile;
  std::ofstream file(marker, std::ios::binary);
  file << kOwnerVersion << '\n' << owner << '\n';
  file.close();
  if (!file) return std::unexpected(io_refusal("marquage du dossier de travail", marker, {}));
  return {};
}

bool starts_with(const fs::path& entry, std::string_view prefix) {
  return entry.filename().string().starts_with(prefix);
}

// Parcourt parent sans exception ; visit rend false pour arrêter le parcours.
template <class Visit>
std::error_code for_each_entry(const fs::path& parent, Visit visit) {
  std::error_code error;
  fs::directory_iterator it(parent, error);
  for (const fs::directory_iterator end; !error && it != end; it.increment(error)) {
    if (!visit(it->path())) break;
  }
  return error;
}

}  // namespace

std::string io_message(std::string_view operation, const fs::path& path, std::error_code error) {
  if (!error) error = std::make_error_code(std::errc::io_error);
  std::string_view hint = " Vérifiez le stockage du cache, puis rouvrez la vidéo.";
  if (error == std::errc::permission_denied || error == std::errc::read_only_file_system) {
    hint =
        " Rendez au cache ses droits d'écriture ou choisissez un stockage inscriptible, puis rouvrez la "
        "vidéo.";
  }
  bool full = error == std::errc::no_space_on_device;
#ifdef EDQUOT
  full = full || error == std::error_code(EDQUOT, std::generic_category());
#endif
  if (full) hint = " Libérez de la place ou du quota pour le cache, puis rouvrez la vidéo.";
  return std::format("{} : {} : {}.{}", operation, path.string(), error.message(), hint);
}

Refusal io_refusal(std::string_view operation, const fs::path& path, std::error_code error) {
  return {RefusalCode::CacheIo, io_message(operation, path, error)};
}

std::expected<void, Refusal> write_atomic(const fs::path& destination, std::string_view bytes) {
  const fs::path temporary =
      destination.parent_path() / (destination.filename().string() + ".tmp-" + random_suffix());
  {
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    file.close();
    if (!file) {
      std::error_code ignored;
      fs::remove(temporary, ignored);
      return std::unexpected(io_refusal("écriture du temporaire", temporary, {}));
    }
  }
  std::error_code error;
  fs::rename(temporary, destination, error);
  if (error) {
    std::error_code ignored;
    fs::remove(temporary, ignored);
    return std::unexpected(io_refusal("publication", destination, error));
  }
  return {};
}

std::expected<fs::path, Refusal> reserve_stage(const fs::path& parent, std::string_view prefix,
                                               std::string_view owner) {
  std::error_code error;
  for (int attempt = 0; attempt < 16; ++attempt) {
    fs::path folder = parent / (std::string(prefix) + random_suffix());
    if (!fs::create_directory(folder, error)) {
      if (error) return std::unexpected(io_refusal("réservation du dossier de travail", folder, error));
      continue;  // nom déjà pris : un autre tirage
    }
    if (auto marked = mark_stage(folder, owner); !marked) {
      fs::remove_all(folder, error);
      return std::unexpected(marked.error());
    }
    return folder;
  }
  return std::unexpected(io_refusal("réservation du dossier de travail", parent, {}));
}

bool owns_stage(const fs::path& folder, std::string_view owner) {
  std::error_code error;
  if (fs::is_symlink(folder, error) || fs::is_symlink(folder / kOwnerFile, error)) return false;
  std::ifstream file(folder / kOwnerFile, std::ios::binary);
  std::string version;
  std::string stored;
  std::string extra;
  return std::getline(file, version) && version == kOwnerVersion && std::getline(file, stored) &&
         stored == owner && !std::getline(file, extra) && file.eof() && !file.bad();
}

bool has_owned_stages(const fs::path& parent, std::string_view prefix, std::string_view owner) {
  bool found = false;
  // Un dossier illisible ne contient rien qu'on puisse reprendre : son erreur ne change rien.
  std::ignore = for_each_entry(parent, [&](const fs::path& entry) {
    found = starts_with(entry, prefix) && owns_stage(entry, owner);
    return !found;
  });
  return found;
}

std::expected<void, Refusal> recover_owned_stages(const fs::path& parent, std::string_view prefix,
                                                  std::string_view owner) {
  // Liste d'abord, supprime ensuite : on ne modifie pas un dossier en le parcourant.
  std::vector<fs::path> orphans;
  const std::error_code error = for_each_entry(parent, [&](const fs::path& entry) {
    if (starts_with(entry, prefix) && owns_stage(entry, owner)) orphans.push_back(entry);
    return true;
  });
  if (error) return std::unexpected(io_refusal("lecture du cache", parent, error));
  for (const fs::path& orphan : orphans) {
    if (auto cleaned = cleanup_stage(orphan); !cleaned) return cleaned;
  }
  return {};
}

std::expected<void, Refusal> cleanup_stage(const fs::path& folder) {
  if (folder.empty()) return {};
#ifndef _WIN32
  if (::access(folder.parent_path().c_str(), W_OK) != 0 && (errno == EACCES || errno == EROFS)) {
    return std::unexpected(
        io_refusal("suppression du dossier de travail", folder, {errno, std::generic_category()}));
  }
#endif
  std::error_code error;
  fs::remove_all(folder, error);
  if (error) return std::unexpected(io_refusal("suppression du dossier de travail", folder, error));
  return {};
}

}  // namespace ajp::prepare
