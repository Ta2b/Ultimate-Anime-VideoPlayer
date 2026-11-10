#include "core/text_file.h"

#include <format>
#include <fstream>
#include <sstream>
#include <system_error>

#include "core/utf8_path.h"

namespace ajp::core {

std::expected<std::string, Refusal> read_text_file(const std::filesystem::path& path) {
  // Un dossier s'ouvre sous Linux et se lit vide : seul un fichier ordinaire est lu.
  if (std::error_code error; !std::filesystem::is_regular_file(path, error)) {
    return refuse(RefusalCode::FileUnreadable, to_utf8(path));
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) return refuse(RefusalCode::FileUnreadable, to_utf8(path));
  std::ostringstream text;
  text << in.rdbuf();
  if (in.bad()) return refuse(RefusalCode::FileUnreadable, to_utf8(path));
  return text.str();
}

std::expected<std::optional<std::string>, Refusal> read_text_file_if_present(
    const std::filesystem::path& path) {
  std::error_code error;
  const std::filesystem::file_status status = std::filesystem::status(path, error);
  if (status.type() == std::filesystem::file_type::not_found) return std::optional<std::string>{};
  if (error)
    return refuse(RefusalCode::FileUnreadable, std::format("{} : {}", to_utf8(path), error.message()));
  auto text = read_text_file(path);
  if (!text) return std::unexpected(text.error());
  return std::optional<std::string>{std::move(*text)};
}

}  // namespace ajp::core
