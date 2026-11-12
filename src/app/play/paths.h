// Les emplacements du lecteur, en un seul endroit (P5.32) : chaque dossier et chaque fichier se calcule ici,
// pas à l'endroit qui s'en sert. Trois racines : le dossier des données (paths.heavy : paquets des moteurs,
// modèles téléchargés, LUT), celui des caches qui se refont seuls (paths.cache : moteurs construits, modèles
// dérivés, plans, cache des shaders), et le dossier des réglages (celui de play.toml, que --data-dir peut
// déplacer). Tests : tests/unit/app/paths_test.cpp.
#pragma once

#include <filesystem>

#include "app/play/player.h"

namespace ajp::app::play::paths {

// Le dossier des données : paths.heavy, sinon le parent des moteurs construits.
inline std::filesystem::path data_root(const PlayerPaths& p) {
  return p.heavy.empty() ? p.engines.parent_path() : p.heavy;
}
// Le dossier des caches qui se refont seuls : paths.cache, sinon celui des données (A36-08).
inline std::filesystem::path cache_root(const PlayerPaths& p) {
  return p.cache.empty() ? data_root(p) : p.cache;
}
// Les modèles dérivés (FP8, INT8, DistilDRBA fusionnés, Core ML) : « modeles ».
inline std::filesystem::path derived_models(const PlayerPaths& p) {
  return cache_root(p) / "modeles";
}
// Les LUT du grade, copiées au premier lancement.
inline std::filesystem::path luts(const PlayerPaths& p) {
  return data_root(p) / "luts";
}
// Le dossier d'installation des moteurs téléchargés, quand le réglage est vide.
inline std::filesystem::path backends_install(const PlayerPaths& p) {
  return data_root(p) / "backends";
}
// Le cache des shaders de libplacebo et celui de la pré-analyse des plans.
inline std::filesystem::path shader_cache(const PlayerPaths& p) {
  return cache_root(p) / "shaders.cache";
}
inline std::filesystem::path shot_cache(const PlayerPaths& p) {
  return cache_root(p) / "plans";
}
// La séquence du banc (media/bench_clip.h), écrite au premier banc : dans le dossier local, elle ne voyage
// pas avec le profil itinérant (B36-08).
inline std::filesystem::path bench_clip(const PlayerPaths& p) {
  return data_root(p) / "bench-clip-v4.mkv";
}
// Le dossier des réglages : paths.settings, ou le dossier des données sans --data-dir.
inline std::filesystem::path settings_dir(const PlayerPaths& p) {
  return p.settings.empty() ? data_root(p) : p.settings;
}

// Les fichiers du dossier des réglages.
inline std::filesystem::path play_settings(const std::filesystem::path& dir) {
  return dir / "play.toml";
}
inline std::filesystem::path engine_settings(const std::filesystem::path& dir) {
  return dir / "settings.toml";
}
inline std::filesystem::path user_keys(const std::filesystem::path& dir) {
  return dir / "user.toml";
}
inline std::filesystem::path bench(const std::filesystem::path& dir) {
  return dir / "bench.toml";
}
inline std::filesystem::path resume(const std::filesystem::path& dir) {
  return dir / "resume.tsv";
}
inline std::filesystem::path auto_memory(const std::filesystem::path& dir) {
  return dir / "auto_memory.toml";
}
// update.toml : l'état des mises à jour, dans le dossier du lecteur (paths.data).
inline std::filesystem::path update_state(const std::filesystem::path& app_data) {
  return app_data / "update.toml";
}

}  // namespace ajp::app::play::paths
