#include "app/play/motion_model.h"

#include <cstddef>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>
#include <vector>

#include "prepare/onnx_outputs.h"
#include "prepare/sha256.h"

namespace ajp::app::play {
namespace {

// Tenseurs du flot et du masque dans les exports RIFE 4.2x (vs-rife, ceux d'AnimeJaNai) : ceux que
// la variante du lecteur actuel (.aji-motion-v3) sort aussi.
std::vector<prepare::OnnxOutput> motion_outputs() {
  return {{.name = "/Add_7_output_0", .elem_type = 10}, {.name = "/Sigmoid_output_0", .elem_type = 10}};
}

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace

std::expected<std::filesystem::path, Refusal> motion_model(const std::filesystem::path& rife_onnx,
                                                           const std::filesystem::path& dir) {
  auto sha = prepare::sha256_file(rife_onnx);
  if (!sha) return std::unexpected(std::move(sha.error()));
  std::filesystem::path target =
      dir / std::format("{}-mouvement-{}.onnx", rife_onnx.stem().string(), sha->substr(0, 16));
  std::error_code error;
  if (std::filesystem::exists(target, error)) return target;

  std::ifstream in(rife_onnx, std::ios::binary);
  if (!in) return refuse(RefusalCode::ModelUnreadable, rife_onnx.string());
  const std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::vector<std::byte> model(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) model[i] = static_cast<std::byte>(raw[i]);
  auto derived = prepare::with_outputs(model, motion_outputs());
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
