#include "app/play/derived_model.h"

#include <format>
#include <string>
#include <system_error>
#include <utility>

#include "core/file_sync.h"
#include "core/utf8_path.h"
#include "prepare/model_source.h"
#include "prepare/sha256.h"

namespace ajp::app::play {
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
