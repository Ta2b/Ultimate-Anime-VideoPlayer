// L'écriture d'un paquet SPAN FP8 et de son reçu (span_derive.h, commit_derivation) : par fichiers
// provisoires renommés ensuite, l'ancien gardé sous « .old » le temps de l'opération.
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <string_view>
#include <system_error>

#include "app/play/span_derive.h"
#include "core/file_sync.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

std::filesystem::path partial_of(const std::filesystem::path& path) {
  std::filesystem::path out = path;
  out += ".part";
  return out;
}

std::expected<void, Refusal> write_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (file)
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  file.close();
  if (!file)
    return refuse(RefusalCode::IoFailed, std::format("{} : écriture impossible", prepare::utf8_path(path)));
  if (const std::error_code synced = core::sync_file(path)) {
    return refuse(RefusalCode::IoFailed, std::format("{} : écriture sur le disque impossible ({})",
                                                     prepare::utf8_path(path), synced.message()));
  }
  return {};
}

// Un fichier remplacé par un autre déjà écrit sous un nom provisoire : l'ancien passe à « .old » le temps de
// l'opération, et revient si elle échoue.
class Swap {
 public:
  explicit Swap(std::filesystem::path target)
      : target_(std::move(target)), part_(partial_of(target_)), old_(target_) {
    old_ += ".old";
    // Le « .old » d'une opération interrompue (coupure entre deux renommages) : seul le processus qui
    // l'avait créé le retirait, il restait sur le disque (A02-14). Si la coupure est tombée entre les deux
    // renommages, la cible manque et « .old » est la seule copie : il revient à sa place (B07-12).
    std::error_code error;
    if (std::filesystem::exists(old_, error) && !std::filesystem::exists(target_, error))
      std::filesystem::rename(old_, target_, error);
    std::filesystem::remove(old_, error);
  }
  Swap(const Swap&) = delete;
  Swap& operator=(const Swap&) = delete;
  const std::filesystem::path& part() const { return part_; }

  std::expected<void, Refusal> install() {
    std::error_code error;
    if (std::filesystem::exists(target_, error)) {
      std::filesystem::rename(target_, old_, error);
      if (error) return failure(error);
      had_old_ = true;
    }
    std::filesystem::rename(part_, target_, error);
    if (error) return failure(error);
    installed_ = true;
    return {};
  }
  // Remet l'état d'avant : le nouveau fichier part, l'ancien revient, le provisoire disparaît.
  void rollback() {
    std::error_code error;
    if (installed_) std::filesystem::remove(target_, error);
    if (had_old_) std::filesystem::rename(old_, target_, error);
    std::filesystem::remove(part_, error);
    installed_ = false;
    had_old_ = false;
  }
  // L'opération a réussi : l'ancien fichier n'a plus de raison d'être.
  void finish() {
    std::error_code error;
    if (had_old_) std::filesystem::remove(old_, error);
    had_old_ = false;
  }

 private:
  std::unexpected<Refusal> failure(const std::error_code& error) const {
    return refuse(RefusalCode::IoFailed,
                  std::format("{} : {}", prepare::utf8_path(target_), error.message()));
  }

  std::filesystem::path target_;
  std::filesystem::path part_;
  std::filesystem::path old_;
  bool had_old_ = false;
  bool installed_ = false;
};

}  // namespace

std::expected<void, Refusal> commit_derivation(const std::filesystem::path& package,
                                               const std::filesystem::path& receipt,
                                               std::span<const std::byte> package_bytes,
                                               std::string_view receipt_text, bool passed) {
  std::error_code error;
  std::filesystem::create_directories(receipt.parent_path(), error);
  Swap package_swap(package);
  Swap receipt_swap(receipt);
  const auto fail = [&](std::expected<void, Refusal> why) {
    package_swap.rollback();
    receipt_swap.rollback();
    return why;
  };
  // Tout s'écrit d'abord sous un nom provisoire, sur le disque avant son renommage (P5.11) : un disque
  // plein ou une coupure laisse l'ancien état entier, jamais un paquet ou un reçu vide sous son nom
  // définitif (A02-14).
  if (passed) {
    if (auto ok = write_file(package_swap.part(), package_bytes); !ok) return fail(ok);
  }
  const auto* text = reinterpret_cast<const std::byte*>(receipt_text.data());
  if (auto ok = write_file(receipt_swap.part(), std::span(text, receipt_text.size())); !ok) return fail(ok);
  if (passed) {
    if (auto ok = package_swap.install(); !ok) return fail(ok);
  }
  if (auto ok = receipt_swap.install(); !ok) return fail(ok);
  core::sync_directory(receipt.parent_path());
  if (package.parent_path() != receipt.parent_path()) core::sync_directory(package.parent_path());
  package_swap.finish();
  receipt_swap.finish();
  // Refusé : le reçu dit « passed = false », et le paquet d'une dérivation d'avant n'y correspond plus.
  if (!passed) std::filesystem::remove(package, error);
  return {};
}

}  // namespace ajp::app::play
