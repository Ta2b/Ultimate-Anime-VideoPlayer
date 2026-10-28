// Réglages de l'application, hors profil : settings.toml, dans le dossier de
// données de l'utilisateur. Absent au premier lancement : les défauts valent.
//
//   version = 1
//   first_run_done = true
//   first_run_step = 8
//   open_with_registered = true
//   update_mode = "notify"      # notify, auto, never
//   update_channel = "stable"   # stable, preview
//   listening = "speakers"      # headphones, speakers, cinema ; absent : pas encore dit
//   [engines]
//   chosen = ["vulkan", "tensorrt"]
//   installed = ["vulkan", "tensorrt"]
//   preferred = { upscale = "tensorrt", interpolate = "tensorrt" }
//   [languages]
//   audio = ["ja", "en"]
//   subtitles = ["fr", "en"]
//   [playback]
//   resume_position = true
//   audio_delay_ms = 0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/engines.h"
#include "core/profile.h"
#include "core/types.h"

namespace ajp::core {

// Ce pour quoi un moteur est préféré. TrueHdr n'accepte que les SDK RTX Video et Maxine.
enum class Usage : std::uint8_t { Upscale, Interpolate, TrueHdr };
inline constexpr std::size_t kUsageCount = 3;

inline constexpr int kSettingsVersion = 1;

namespace limits {
inline constexpr Range<std::int64_t> kAudioDelayMs{-10'000, 10'000};
inline constexpr std::size_t kMaxLanguages = 8;
}  // namespace limits

struct Settings {
  bool first_run_done = false;
  int first_run_step = 0;                // le dernier écran de l'assistant vu (0 : aucun), pour reprendre en route
  bool open_with_registered = false;     // l'inscription « Ouvrir avec » a été faite une fois : pas de réinscription
  std::string update_mode = "notify";    // « notify » (Me prévenir), « auto », « never » (docs/ux-parametres.md, 4.5)
  std::string update_channel = "stable"; // « stable » ou « preview »
  std::string listening;                 // « Comment écoutez-vous ? » : « headphones », « speakers », « cinema » ; vide : non dit
  std::vector<EngineId> chosen_engines{EngineId::Vulkan};     // cochés dans l'écran « Moteurs »
  std::vector<EngineId> installed_engines{EngineId::Vulkan};  // présents et vérifiés
  std::array<std::optional<EngineId>, kUsageCount> preferred{EngineId::Vulkan, EngineId::Vulkan,
                                                             std::nullopt};
  std::vector<std::string> audio_languages;  // étiquettes BCP 47, par priorité : « ja », « pt-BR »
  std::vector<std::string> subtitle_languages;
  bool resume_position = true;  // reprendre une vidéo là où elle s'est arrêtée
  std::int64_t audio_delay_ms = 0;
  friend bool operator==(const Settings&, const Settings&) = default;
};

std::string_view usage_name(Usage usage);  // « upscale », « interpolate », « truehdr »

// Plages et cohérence : sans doublon, Vulkan toujours choisi et installé, moteur
// préféré installé et propre à son usage.
std::expected<void, Refusal> validate_settings(const Settings& settings);

std::expected<Settings, Refusal> parse_settings(std::string_view toml_text);
// Fichier absent : réglages par défaut. Fichier présent mais faux : refus, jamais de défaut silencieux.
std::expected<Settings, Refusal> load_settings(const std::filesystem::path& path);

// Texte de settings, fusionné dans existing_toml dont les clés inconnues restent.
std::expected<std::string, Refusal> format_settings(const Settings& settings,
                                                    std::string_view existing_toml = "");
// Écriture atomique, après validation ; garde les clés inconnues du fichier en place.
std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const Settings& settings);

}  // namespace ajp::core
