// Les fichiers tiers de NVIDIA d'une installation de moteur (engine_install.h) : repris d'un dossier voisin
// quand le même SHA-256 y est déjà, sinon tirés par plages d'une roue en ligne, décompressés et vérifiés.
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "app/play/engine_install.h"
#include "app/play/engine_tools.h"
#include "app/play/i18n.h"
#include "core/file_sync.h"
#include "core/inflate.h"
#include "core/log.h"
#include "core/utf8_path.h"
#include "prepare/utf8.h"

namespace ajp::app::play {

using prepare::ansi_safe_path;  // curl relit sa ligne de commande en ANSI sous Windows

bool reuse_vendor_file(const core::VendorFile& file, const std::filesystem::path& dest,
                       const std::filesystem::path& engines_root, const std::stop_token& stop) {
  std::error_code error;
  // Parcours sans exception (A06-10) : un dossier voisin illisible arrête la recherche, il ne fait pas
  // échouer l'installation.
  std::filesystem::directory_iterator it(engines_root, error);
  for (; !error && it != std::filesystem::directory_iterator(); it.increment(error)) {
    const std::filesystem::path candidate = it->path() / file.name;
    if (candidate.parent_path() == dest.parent_path()) continue;
    std::error_code checked;
    if (!std::filesystem::is_regular_file(candidate, checked) ||
        std::filesystem::file_size(candidate, checked) != file.size || checked)
      continue;
    if (!core::verify_file(candidate, file.size, file.sha256, stop)) continue;
    std::error_code placed;
    std::filesystem::create_directories(dest.parent_path(), placed);
    std::filesystem::remove(dest, placed);
    placed.clear();
    std::filesystem::create_hard_link(candidate, dest, placed);
    if (placed) {
      placed.clear();
      std::filesystem::copy_file(candidate, dest, std::filesystem::copy_options::overwrite_existing, placed);
    }
    if (!placed) return true;
  }
  return false;
}

// La taille d'un fichier, 0 s'il n'existe pas ; `error` dit s'il ne se lit pas.
std::uint64_t size_or_zero(const std::filesystem::path& path, std::error_code& error) {
  return std::filesystem::exists(path, error) ? std::filesystem::file_size(path, error) : 0;
}

enum class Appended { Added, TooLong, Unwritable };

// Ajoute au .part le morceau reçu, même coupé. TooLong si le morceau dépasse la plage demandée (`limit`
// octets) : un serveur qui l'ignore rend la roue entière (200), le morceau est alors jeté (A06-07).
// Unwritable si l'écriture échoue (disque plein) : l'ancien code l'ignorait et laissait un .part trop court,
// repris comme s'il était bon (B06-04).
Appended append_chunk(const std::filesystem::path& chunk, const std::filesystem::path& packed,
                      std::uint64_t limit) {
  std::error_code error;
  const std::uintmax_t size =
      std::filesystem::exists(chunk, error) ? std::filesystem::file_size(chunk, error) : 0;
  if (error) return Appended::Added;  // rien de lisible : la reprise suivante le dira
  if (size > limit) return Appended::TooLong;
  if (size == 0) return Appended::Added;
  std::ifstream piece(chunk, std::ios::binary);
  std::ofstream whole(packed, std::ios::binary | std::ios::app);
  if (!piece || !whole) return Appended::Unwritable;
  whole << piece.rdbuf();
  whole.flush();
  if (!whole || piece.bad()) return Appended::Unwritable;
  return Appended::Added;
}

// Tire les octets compressés de `file` dans `packed`, par plages, en reprenant un .part laissé (P3.7).
std::expected<void, Refusal> pull_packed(const core::VendorFile& file, const std::filesystem::path& curl,
                                         const std::filesystem::path& packed,
                                         const std::filesystem::path& chunk,
                                         const InstallProgress& progress) {
  std::error_code error;
  // P3.7 : un .part laissé par un essai coupé se reprend par plage ; trop long, il repart de zéro.
  std::uint64_t have = size_or_zero(packed, error);
  if (error || have > file.packed_size) {
    std::filesystem::remove(packed, error);
    have = 0;
  }
  const std::uint64_t last = file.data_offset + file.packed_size - 1;
  // Trois reprises au plus après une coupure, chacune depuis ce qui est déjà là (curl réessaie aussi
  // lui-même).
  std::optional<Refusal> failure;
  for (int attempt = 0; attempt < 4 && have < file.packed_size; ++attempt) {
    std::filesystem::remove(chunk, error);
    std::vector<std::string> args = curl_download_args(progress.allow_local_files);
    // --max-filesize : un serveur ou un proxy qui ignore la plage et rend la roue entière (200) est arrêté
    // tout de suite (A06-07).
    args.insert(args.end(), {"--progress-bar", "--max-filesize", std::to_string(file.packed_size - have),
                             "--range", std::format("{}-{}", file.data_offset + have, last), "--output",
                             ansi_safe_path(chunk), file.url});
    const double done = static_cast<double>(have) / static_cast<double>(file.packed_size);
    auto fetched = run_tool(
        {.program = curl, .args = std::move(args), .timeout = std::chrono::hours(2)},
        [&](std::string_view line) {
          if (const auto f = curl_percent(line); f && progress.report)
            progress.report(static_cast<float>(done + ((1.0 - done) * *f)) * 0.85F, file.name);
        },
        progress.stop, i18n::tr_format("downloading {}", file.name));
    // Ce qui est arrivé, même coupé, s'ajoute au .part ; un morceau plus long que la plage demandée n'est pas
    // la plage (réponse 200 entière) : jeté (A06-07).
    switch (append_chunk(chunk, packed, file.packed_size - have)) {
      case Appended::Added:
        break;
      case Appended::TooLong:
        fetched = refuse(RefusalCode::HashMismatch,
                         i18n::tr_format("{}: the server ignored the byte range", file.name));
        break;
      case Appended::Unwritable:
        std::filesystem::remove(chunk, error);
        return refuse(RefusalCode::FileUnwritable, core::to_utf8(packed));
    }
    std::filesystem::remove(chunk, error);
    have = size_or_zero(packed, error);
    if (fetched) break;
    failure = fetched.error();
    if (failure->code == RefusalCode::Cancelled) return std::unexpected(*failure);
  }
  if (have < file.packed_size && failure) return std::unexpected(*failure);
  return {};
}

namespace {

// Décompresse (ou recopie) `packed` dans `partial`, puis vérifie la taille et le SHA-256 de l'index ;
// `partial` est retiré sur tout refus.
std::expected<void, Refusal> unpack_vendor_file(const core::VendorFile& file,
                                                const std::filesystem::path& packed,
                                                const std::filesystem::path& partial,
                                                const std::stop_token& stop) {
  std::ifstream in(packed, std::ios::binary);
  std::ofstream out(partial, std::ios::binary | std::ios::trunc);
  const auto close = [&] {
    in.close();
    out.close();
  };
  if (!in || !out) {
    close();
    return refuse(RefusalCode::FileUnwritable, partial.string());
  }
  core::Sha256 hash;
  std::uint64_t produced = 0;
  bool write_failed = false;
  bool overflow = false;
  const auto sink = [&](std::span<const std::uint8_t> bytes) {
    if (stop.stop_requested()) return false;
    // Jamais plus que la taille de l'index : un flux deflate forgé peut produire mille fois sa taille
    // (A06-05).
    if (produced + bytes.size() > file.size) {
      overflow = true;
      return false;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
      write_failed = true;
      return false;
    }
    hash.update(std::as_bytes(bytes));
    produced += bytes.size();
    return true;
  };
  std::expected<void, Refusal> result;
  if (file.deflated) {
    auto inflated = core::inflate_raw(
        [&](std::span<std::uint8_t> buffer) {
          in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
          return static_cast<std::size_t>(in.gcount());
        },
        sink);
    if (!inflated) result = std::unexpected(inflated.error());
  } else {
    std::array<std::uint8_t, 1U << 16> buffer{};
    while (in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size())) ||
           in.gcount() > 0) {
      if (!sink(std::span<const std::uint8_t>(buffer).first(static_cast<std::size_t>(in.gcount())))) break;
    }
  }
  const bool flushed = static_cast<bool>(out.flush());
  close();
  const auto discard = [&] {
    std::error_code ignored;
    std::filesystem::remove(partial, ignored);
  };
  if (stop.stop_requested()) {
    discard();
    return refuse(RefusalCode::Cancelled, file.name);
  }
  if (write_failed || !flushed) {
    discard();
    return refuse(RefusalCode::FileUnwritable, partial.string());
  }
  if (overflow) {
    discard();
    return refuse(RefusalCode::HashMismatch,
                  std::format("{} : plus de {} octets décompressés, refusé", file.name, file.size));
  }
  if (!result) {
    discard();
    return std::unexpected(result.error());
  }
  const std::string sha = core::to_hex(hash.finish());
  if (produced != file.size || sha != file.sha256) {
    discard();
    return refuse(RefusalCode::HashMismatch,
                  std::format("{} : {} octets, SHA-256 {} ; {} octets et {} attendus", file.name, produced,
                              sha, file.size, file.sha256));
  }
  return {};
}

}  // namespace

std::expected<void, Refusal> fetch_vendor_file(const core::VendorFile& file,
                                               const std::filesystem::path& dest,
                                               const std::filesystem::path& scratch,
                                               const InstallProgress& progress) {
  const auto curl = system_tool("curl");
  if (!curl) return refuse(RefusalCode::LibraryUnavailable, i18n::tr("curl not found on this system"));
  std::error_code error;
  std::filesystem::create_directories(scratch, error);
  // Nommés par l'empreinte du fichier (A06-04) : un .part d'une autre version au même nom ne se reprend pas.
  const std::filesystem::path packed = scratch / (file.sha256 + ".part");
  const std::filesystem::path chunk = scratch / (file.sha256 + ".morceau");
  const auto cleanup = [&] {
    std::error_code ignored;
    std::filesystem::remove(packed, ignored);
    std::filesystem::remove(chunk, ignored);
  };
  if (file.packed_size == 0) return refuse(RefusalCode::FieldOutOfRange, file.name + i18n::tr(": zero size"));
  if (auto pulled = pull_packed(file, *curl, packed, chunk, progress); !pulled) {
    // Une annulation garde le .part : la reprise repart de là, pas de zéro (B06-05). Une autre erreur le
    // garde aussi, pull_packed ne laisse que des octets valides ; seul le morceau en cours part.
    std::error_code ignored;
    std::filesystem::remove(chunk, ignored);
    return pulled;
  }
  const std::uintmax_t got = std::filesystem::file_size(packed, error);
  if (error || got != file.packed_size) {
    cleanup();
    return refuse(RefusalCode::HashMismatch, i18n::tr_format("{}: {} bytes received, {} expected", file.name,
                                                             error ? 0 : got, file.packed_size));
  }
  if (progress.report) progress.report(0.9F, file.name);  // tiré : la décompression commence
  std::filesystem::path partial = dest;
  partial += ".partiel";
  const auto unpacked = unpack_vendor_file(file, packed, partial, progress.stop);
  // Annulé pendant la décompression, le .part (déjà entier, vérifié) reste : la reprise le décompresse sans
  // le retélécharger (B06-05).
  if (progress.stop.stop_requested()) {
    std::error_code ignored;
    std::filesystem::remove(chunk, ignored);
  } else {
    cleanup();
  }
  if (!unpacked) return unpacked;
  error = core::rename_retrying(partial, dest, kInstallRenamePatience);
  if (error) {
    std::error_code ignored;
    std::filesystem::remove(partial, ignored);
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", dest.string(), error.message()));
  }
  if (progress.report) progress.report(1.0F, file.name);
  return {};
}

}  // namespace ajp::app::play
