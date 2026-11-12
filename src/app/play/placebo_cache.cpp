// Le cache des shaders et pipelines de libplacebo (ScreenOptions::shader_cache) : au plus 64 Mo, relu d'un
// fichier à l'ouverture de l'écran, gardé en cours de séance et à la fermeture. Un fichier abîmé ou d'une
// autre version de libplacebo est simplement ignoré (pl_cache_load le refuse) et sera réécrit.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <libplacebo/cache.h>
#include <libplacebo/vulkan.h>

#include "app/play/placebo_impl.h"
#include "core/file_sync.h"

namespace ajp::app::play {
namespace {

// Les « <cache>.<n>.part » d'une fermeture interrompue (save_shader_cache) : effacés passé une heure, le
// temps qu'un autre lecteur finisse d'écrire le sien (A10-11).
void purge_cache_parts(const std::filesystem::path& file) {
  std::error_code error;
  const std::u8string prefix = file.filename().u8string() + u8".";
  const auto now = std::filesystem::file_time_type::clock::now();
  for (std::filesystem::directory_iterator it(file.parent_path(), error), end; !error && it != end;
       it.increment(error)) {
    const std::u8string name = it->path().filename().u8string();
    if (!name.starts_with(prefix) || !name.ends_with(u8".part")) continue;
    std::error_code ignored;
    const auto written = std::filesystem::last_write_time(it->path(), ignored);
    if (!ignored && now - written > std::chrono::hours(1)) std::filesystem::remove(it->path(), ignored);
  }
}

// L'écriture d'un instantané du cache, par l'écriture atomique du projet (core::write_atomic : à côté, sur le
// disque, puis renommé) : une fermeture brutale ne laisse jamais un fichier à moitié écrit. pl_cache est sûr
// d'un thread à l'autre (libplacebo/cache.h).
void write_cache(pl_cache cache, const std::filesystem::path& file, spdlog::logger& channel) {
  const std::size_t size = pl_cache_save(cache, nullptr, 0);
  if (size == 0) return;
  std::vector<std::byte> bytes(size);
  bytes.resize(pl_cache_save(cache, reinterpret_cast<std::uint8_t*>(bytes.data()), bytes.size()));
  if (auto written = core::write_atomic(file, bytes); !written)
    channel.warn("cache des shaders non écrit : {}", written.error().detail);
}

// Le cache est aussi gardé en cours de séance (C10-05) : il ne l'était qu'à la fermeture normale, et un arrêt
// brutal (plantage, exit_stuck, fin de session Windows, kill) perdait tout ce qui avait été compilé ; la
// séance suivante recompilait encore le grade, la pile de LUT et Anime4K pendant ses premières images.
// Toutes les kKeepEveryUs de présentation, s'il a changé, sur un thread à part : rien sur le chemin de
// l'image qu'un relevé de sa signature.
constexpr std::int64_t kKeepEveryUs = 20'000'000;

}  // namespace

void PlaceboScreen::Impl::open_shader_cache(const std::filesystem::path& file) {
  Impl& impl = *this;
  if (file.empty()) return;
  pl_cache_params params = pl_cache_default_params;
  params.log = impl.log;
  params.max_total_size = 64u << 20u;
  impl.shader_cache = pl_cache_create(&params);
  if (impl.shader_cache == nullptr) return;
  impl.shader_cache_file = file;
  purge_cache_parts(file);
  std::error_code error;
  const std::uintmax_t on_disk = std::filesystem::file_size(file, error);
  if (!error && on_disk > params.max_total_size) {
    // Jamais écrit par nous à cette taille : abîmé ou d'ailleurs, il n'est pas lu en mémoire (A10-11).
    impl.channel->warn("cache des shaders : {} Mo sur le disque, plus que le plafond, ignoré",
                       on_disk >> 20u);
  } else if (std::ifstream in(file, std::ios::binary); !error && in) {
    // D'un seul bloc, à la taille du fichier : jusqu'à 64 Mo sur le chemin de l'ouverture de la fenêtre.
    std::vector<char> bytes(static_cast<std::size_t>(on_disk));
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    bytes.resize(static_cast<std::size_t>(in.gcount()));
    const int loaded =
        pl_cache_load(impl.shader_cache, reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    impl.channel->info("cache des shaders : {} objets relus ({} Ko)", std::max(loaded, 0),
                       bytes.size() >> 10U);
  }
  impl.shader_cache_saved = pl_cache_signature(impl.shader_cache);
  pl_gpu_set_cache(impl.vk->gpu, impl.shader_cache);
}

void PlaceboScreen::Impl::keep_shader_cache(std::int64_t now_us) {
  if (shader_cache == nullptr || shader_cache_writing.load(std::memory_order_acquire)) return;
  if (shader_cache_checked_us == 0) shader_cache_checked_us = now_us;
  if (now_us - shader_cache_checked_us < kKeepEveryUs) return;
  shader_cache_checked_us = now_us;
  const std::uint64_t signature = pl_cache_signature(shader_cache);
  if (signature == shader_cache_saved) return;
  shader_cache_saved = signature;
  if (shader_cache_writer.joinable()) shader_cache_writer.join();  // fini : shader_cache_writing est faux
  shader_cache_writing.store(true, std::memory_order_release);
  shader_cache_writer = std::thread([this] {
    write_cache(shader_cache, shader_cache_file, *channel);
    shader_cache_writing.store(false, std::memory_order_release);
  });
}

// Gardé à la fermeture. Un cache que la lecture n'a pas changé depuis sa dernière écriture n'est pas réécrit
// (B36-05 : jusqu'à 64 Mo à chaque fermeture).
void PlaceboScreen::Impl::save_shader_cache() {
  if (shader_cache_writer.joinable()) shader_cache_writer.join();
  if (shader_cache == nullptr) return;
  if (vk != nullptr) pl_gpu_set_cache(vk->gpu, nullptr);
  if (pl_cache_signature(shader_cache) != shader_cache_saved)
    write_cache(shader_cache, shader_cache_file, *channel);
  pl_cache_destroy(&shader_cache);
}

}  // namespace ajp::app::play
