#include "vendor_hdr/sdk_copy.h"

#include <format>
#include <string>
#include <system_error>

#include "vendor_hdr/path_text.h"

namespace ajp::vendor_hdr {
namespace {

namespace fs = std::filesystem;

Done io_failure(const fs::path& relative, const std::error_code& error) {
  return refuse(RefusalCode::IoFailed,
                std::format("copie de {} impossible : {}", utf8(relative), error.message()));
}

// Recrée le lien source (relative sous from) en lien relatif, s'il vise l'intérieur du SDK.
// Repris dans l'esprit de manager-v2 Services/TrueHdrSdkStore.cs (ResolveLink).
Done copy_link(const fs::path& real_from, const fs::path& source, const fs::path& target,
               const fs::path& relative) {
  std::error_code error;
  const fs::path link = fs::read_symlink(source, error);
  if (error) return io_failure(relative, error);
  const fs::path aimed = fs::weakly_canonical(source.parent_path() / link, error);
  if (error) return io_failure(relative, error);
  const fs::path inside = aimed.lexically_relative(real_from);
  if (inside.empty() || *inside.begin() == "..") {
    return refuse(RefusalCode::SdkRejected,
                  std::format("{} est un lien qui sort du SDK ({})", utf8(relative), utf8(link)));
  }
  fs::create_directories(target.parent_path(), error);
  if (!error) fs::create_symlink(inside.lexically_relative(relative.parent_path()), target, error);
  if (error) return io_failure(relative, error);
  return {};
}

Done copy_one(const fs::path& real_from, const fs::path& to, const fs::path& relative) {
  const fs::path source = real_from / relative;
  const fs::path target = to / relative;
  std::error_code error;
  const fs::file_status status = fs::symlink_status(source, error);
  if (error) return io_failure(relative, error);
  if (fs::is_symlink(status)) return copy_link(real_from, source, target, relative);
  if (fs::is_directory(status)) {
    fs::create_directories(target, error);
  } else if (fs::is_regular_file(status)) {
    fs::create_directories(target.parent_path(), error);
    if (!error) fs::copy_file(source, target, fs::copy_options::overwrite_existing, error);
  } else {
    return refuse(RefusalCode::SdkRejected,
                  std::format("{} est un fichier spécial, refusé dans un SDK", utf8(relative)));
  }
  if (error) return io_failure(relative, error);
  return {};
}

// Une entrée imbriquée (external/cuda/lib/libcudart.so.12) ne sort pas du SDK par un dossier
// parent qui serait un lien : copy_link ne voit que le dernier élément du chemin.
Done check_parents(const fs::path& real_from, const fs::path& entry) {
  if (!entry.has_parent_path()) return {};
  std::error_code error;
  const fs::path parent = fs::canonical(real_from / entry.parent_path(), error);
  if (error) return {};  // dossier absent : l'entrée facultative l'est aussi
  const fs::path inside = parent.lexically_relative(real_from);
  if (inside.empty() || *inside.begin() == "..") {
    return refuse(RefusalCode::SdkRejected,
                  std::format("{} passe par un lien qui sort du SDK ({})", utf8(entry), utf8(parent)));
  }
  return {};
}

Done copy_entry(const fs::path& real_from, const fs::path& to, const fs::path& entry,
                const std::stop_token& stop) {
  if (Done inside = check_parents(real_from, entry); !inside) return inside;
  std::error_code error;
  const fs::file_status status = fs::symlink_status(real_from / entry, error);
  if (error || !fs::exists(status)) return {};  // entrée facultative absente
  if (Done copied = copy_one(real_from, to, entry); !copied) return copied;
  if (fs::is_symlink(status)) {
    // Une entrée nommée par son lien (external/cuda/lib/libcudart.so.12) apporte aussi le
    // fichier qu'il vise, déjà vérifié dans le SDK par copy_link.
    const fs::path aimed = fs::canonical(real_from / entry, error);
    if (error) return io_failure(entry, error);
    return copy_entry(real_from, to, aimed.lexically_relative(real_from), stop);
  }
  if (!fs::is_directory(status)) return {};
  // Les liens vers des dossiers ne sont pas suivis : copy_one les recrée tels quels.
  for (fs::recursive_directory_iterator it(real_from / entry, error), end; !error && it != end;
       it.increment(error)) {
    if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, "import annulé");
    const fs::path relative = entry / it->path().lexically_relative(real_from / entry);
    if (Done copied = copy_one(real_from, to, relative); !copied) return copied;
  }
  if (error) return io_failure(entry, error);
  return {};
}

}  // namespace

Done copy_sdk_entries(const fs::path& from, const fs::path& to, std::span<const std::string_view> entries,
                      const std::stop_token& stop) {
  std::error_code error;
  const fs::path real_from = fs::canonical(from, error);
  if (error) return io_failure(from, error);
  for (const std::string_view entry : entries) {
    if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, "import annulé");
    if (Done copied = copy_entry(real_from, to, fs::path(entry), stop); !copied) return copied;
  }
  return {};
}

}  // namespace ajp::vendor_hdr
