#include "app/play/derived_model.h"

#include <format>
#include <fstream>
#include <system_error>
#include <utility>

#include "prepare/model_source.h"
#include "prepare/sha256.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace

std::expected<std::filesystem::path, Refusal> derived_model(const std::filesystem::path& source,
                                                            std::string_view suffix,
                                                            const std::filesystem::path& dir,
                                                            const ModelTransform& transform) {
  auto sha = prepare::sha256_file_known(source);
  if (!sha) return std::unexpected(std::move(sha.error()));
  std::filesystem::path target =
      dir / std::format("{}-{}-{}.onnx", source.stem().string(), suffix, sha->substr(0, 16));
  std::error_code error;
  if (std::filesystem::exists(target, error)) return target;

  auto model = prepare::read_onnx_bytes(source);
  if (!model) return std::unexpected(std::move(model.error()));
  auto derived = transform(*model);
  if (!derived) return std::unexpected(std::move(derived.error()));

  std::filesystem::create_directories(dir, error);
  std::filesystem::path temp = target;
  temp += ".tmp";
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(derived->data()), static_cast<std::streamsize>(derived->size()));
    if (!out.flush()) return refuse(RefusalCode::FileUnwritable, temp.string());
  }
  std::filesystem::rename(temp, target, error);
  if (error)
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", target.string(), error.message()));
  return target;
}

}  // namespace ajp::app::play
