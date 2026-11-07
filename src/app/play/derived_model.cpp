#include "app/play/derived_model.h"

#include <format>
#include <system_error>
#include <utility>

#include "core/file_sync.h"
#include "prepare/model_source.h"
#include "prepare/sha256.h"

namespace ajp::app::play {
std::expected<std::filesystem::path, Refusal> derived_model(const std::filesystem::path& source,
                                                            std::string_view suffix,
                                                            const std::filesystem::path& dir,
                                                            const ModelTransform& transform) {
  auto sha = prepare::sha256_file_known(source);
  if (!sha) return std::unexpected(std::move(sha.error()));
  std::filesystem::path target =
      dir / std::format("{}-{}-{}.onnx", source.stem().string(), suffix, sha->substr(0, 16));
  std::error_code error;
  // Un fichier vide ou illisible n'est pas un modèle : un reste d'un arrêt brutal se refait.
  if (const auto size = std::filesystem::file_size(target, error); !error && size > 0) return target;

  auto model = prepare::read_onnx_bytes(source);
  if (!model) return std::unexpected(std::move(model.error()));
  auto derived = transform(*model);
  if (!derived) return std::unexpected(std::move(derived.error()));

  if (auto written = core::write_atomic(target, *derived); !written) return std::unexpected(written.error());
  return target;
}

}  // namespace ajp::app::play
