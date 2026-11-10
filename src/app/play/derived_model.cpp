#include "app/play/derived_model.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <format>
#include <map>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "core/file_sync.h"
#include "core/utf8_path.h"
#include "prepare/model_source.h"
#include "prepare/sha256.h"

namespace ajp::app::play {
namespace {

// « 1920x1088 » en tête de text, suivi d'un tiret : rend la taille, ou rien.
std::string_view size_token(std::string_view text) {
  const auto digits = [&](std::size_t from) {
    std::size_t at = from;
    while (at < text.size() && std::isdigit(static_cast<unsigned char>(text[at])) != 0) ++at;
    return at;
  };
  const std::size_t cross = digits(0);
  if (cross == 0 || cross >= text.size() || text[cross] != 'x') return {};
  const std::size_t end = digits(cross + 1);
  if (end == cross + 1 || end >= text.size() || text[end] != '-') return {};
  return text.substr(0, end);
}

struct SizedGroup {
  // L'origine de l'horloge des fichiers peut être dans le futur (2174 avec libstdc++) : partir du minimum.
  std::filesystem::file_time_type served = std::filesystem::file_time_type::min();
  std::vector<std::filesystem::path> files;
};

}  // namespace

int purge_sized_derivatives(const std::filesystem::path& dir, std::string_view family,
                            const std::filesystem::path& used, int keep) {
  const std::string stem = core::to_utf8(used.stem());
  if (stem.size() < 16) return 0;
  const std::string_view sha16 = std::string_view(stem).substr(stem.size() - 16);
  std::error_code error;
  std::filesystem::last_write_time(used, std::filesystem::file_time_type::clock::now(), error);
  int removed = 0;
  const auto drop = [&](const std::filesystem::path& file) {
    std::error_code gone;
    if (std::filesystem::remove(file, gone)) ++removed;
  };
  std::map<std::string, SizedGroup, std::less<>> groups;
  std::filesystem::directory_iterator it(dir, error);
  for (; !error && it != std::filesystem::directory_iterator(); it.increment(error)) {
    const std::string name = core::to_utf8(it->path().filename());
    if (!name.starts_with(family)) continue;
    const std::string_view size = size_token(std::string_view(name).substr(family.size()));
    if (size.empty()) continue;
    if (name.find(sha16) == std::string::npos) {
      drop(it->path());
      continue;
    }
    SizedGroup& group = groups[std::string(size)];
    std::error_code unread;
    const auto written = std::filesystem::last_write_time(it->path(), unread);
    if (!unread) group.served = std::max(group.served, written);
    group.files.push_back(it->path());
  }
  std::vector<const SizedGroup*> order;
  order.reserve(groups.size());
  for (const auto& [size, group] : groups) order.push_back(&group);
  std::ranges::sort(order, [](const SizedGroup* a, const SizedGroup* b) { return a->served > b->served; });
  for (std::size_t i = static_cast<std::size_t>(std::max(keep, 1)); i < order.size(); ++i)
    for (const std::filesystem::path& file : order[i]->files) drop(file);
  return removed;
}

std::expected<std::filesystem::path, Refusal> derived_model(const std::filesystem::path& source,
                                                            std::string_view suffix,
                                                            const std::filesystem::path& dir,
                                                            const ModelTransform& transform) {
  auto sha = prepare::sha256_file_known(source);
  if (!sha) return std::unexpected(std::move(sha.error()));
  // La version des transformations entre dans le nom (A36-04) : un correctif de with_motion_outputs,
  // with_drba_motion_outputs ou with_fixed_dims refait les dérivés au lieu de reprendre les anciens.
  const std::string prefix = std::format("{}-{}-", core::to_utf8(source.stem()), suffix);
  std::filesystem::path target =
      dir /
      core::path_from_utf8(std::format("{}t{}-{}.onnx", prefix, kDerivedModelVersion, sha->substr(0, 16)));
  std::error_code error;
  // Un fichier vide ou illisible n'est pas un modèle : un reste d'un arrêt brutal se refait.
  if (const auto size = std::filesystem::file_size(target, error); !error && size > 0) return target;

  auto model = prepare::read_onnx_bytes(source);
  if (!model) return std::unexpected(std::move(model.error()));
  auto derived = transform(*model);
  if (!derived) return std::unexpected(std::move(derived.error()));

  if (auto written = core::write_atomic(target, *derived); !written) return std::unexpected(written.error());
  // Les dérivés d'avant du même modèle et du même suffixe (autre source, autre version) ne servent plus.
  std::filesystem::directory_iterator it(dir, error);
  for (; !error && it != std::filesystem::directory_iterator(); it.increment(error)) {
    const std::string name = core::to_utf8(it->path().filename());
    if (it->path() != target && name.starts_with(prefix) && name.ends_with(".onnx")) {
      std::error_code gone;
      std::filesystem::remove(it->path(), gone);
    }
  }
  return target;
}

}  // namespace ajp::app::play
