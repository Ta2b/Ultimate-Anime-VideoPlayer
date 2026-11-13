// Le fichier play.toml des réglages : lecture (parse_settings, load_settings), écriture
// (format_settings, save_settings). Les valeurs lues passent par validate() (settings.cpp).
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <initializer_list>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "app/play/auto_config.h"
#include "app/play/invalid_copies.h"
#include "app/play/settings.h"
#include "app/play/settings_refuse.h"
#include "app/play/settings_toml_lenient.h"
#include "core/log.h"
#include "core/text_file.h"
#include "core/utf8_path.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {
namespace {

// Lit table[key] s'il existe, du type de out (entier, booléen, réel, texte) ; FieldType sinon.
template <class T>
std::expected<void, Refusal> read(const toml::table& table, std::string_view section, std::string_view key,
                                  T& out) {
  const toml::node* node = table.get(key);
  if (node == nullptr) return {};
  if constexpr (std::is_same_v<T, bool>) {
    if (auto v = node->value_exact<bool>()) {
      out = *v;
      return {};
    }
  } else if constexpr (std::is_same_v<T, int>) {
    if (auto v = node->value_exact<std::int64_t>(); v && *v >= INT32_MIN && *v <= INT32_MAX) {
      out = static_cast<int>(*v);
      return {};
    }
  } else if constexpr (std::is_same_v<T, float>) {
    if (auto v = node->value<double>()) {  // « 1 » se lit comme 1.0
      out = static_cast<float>(*v);
      return {};
    }
  } else if constexpr (std::is_same_v<T, double>) {
    if (auto v = node->value<double>()) {
      out = *v;
      return {};
    }
  } else {
    if (auto v = node->value_exact<std::string>()) {
      out = *v;
      return {};
    }
  }
  return refuse_key(RefusalCode::FieldType, std::format("{}.{}", section, key));
}

// Un énuméré lu par son nom (name(valeur)) : la valeur du texte, ou FieldOutOfRange pour un autre. Les noms
// retirés (retired) donnent retired_as sans plainte.
template <class E>
std::expected<void, Refusal> read_named(const toml::table& table, std::string_view section,
                                        std::string_view key, E& out, int count,
                                        std::span<const std::string_view> retired = {}, E retired_as = E{}) {
  std::string text(name(out));
  if (auto ok = read(table, section, key, text); !ok) return ok;
  if (std::ranges::contains(retired, text)) {
    out = retired_as;
    return {};
  }
  for (int i = 0; i < count; ++i) {
    if (name(static_cast<E>(i)) == text) {
      out = static_cast<E>(i);
      return {};
    }
  }
  return refuse_key(RefusalCode::FieldOutOfRange, std::format("{}.{}", section, key),
                    std::format(" : « {} »", text));
}

// L'ancienne liste de langues : un tableau de balises (« ["fre", "eng"] ») ; FieldType sinon.
std::expected<void, Refusal> read_legacy_languages(const toml::table& table, std::string_view section,
                                                   std::vector<std::string>& out) {
  const toml::node* node = table.get("languages");
  if (node == nullptr) return {};
  const toml::array* list = node->as_array();
  if (list == nullptr) return refuse_key(RefusalCode::FieldType, std::format("{}.languages", section));
  std::vector<std::string> languages;
  for (const toml::node& item : *list) {
    const std::optional<std::string> tag = item.value_exact<std::string>();
    if (!tag) return refuse_key(RefusalCode::FieldType, std::format("{}.languages", section));
    languages.push_back(*tag);
  }
  out = std::move(languages);
  return {};
}

// interpolation.auto_mode (« libre » : Rapide, « recommande » : Stable ; toute autre valeur est refusée) et
// auto_off, « Couper l'interpolation si la carte ne suit pas », cochée dans un fichier d'avant elle.
void read_auto_mode(const toml::table& table, PlaySettings& s,
                    std::vector<std::expected<void, Refusal>>& reads) {
  reads.push_back(read(table, "interpolation", "auto_off", s.auto_off));
  std::string auto_mode = s.auto_free ? "libre" : "recommande";
  reads.push_back(read(table, "interpolation", "auto_mode", auto_mode));
  if (auto_mode == "libre" || auto_mode == "recommande") {
    s.auto_free = auto_mode == "libre";
  } else {
    reads.emplace_back(refuse_key(RefusalCode::FieldOutOfRange, "interpolation.auto_mode",
                                  std::format(" : « {} » (recommande ou libre)", auto_mode)));
  }
}

// interpolation.rife : un entier (×3), ou un réel en moitié (×1.5).
std::expected<void, Refusal> read_factor(const toml::table& table, PlaySettings& s) {
  const toml::node* node = table.get("rife");
  if (node == nullptr) return {};
  if (auto whole = node->value_exact<std::int64_t>(); whole && *whole >= INT32_MIN && *whole <= INT32_MAX) {
    s.rife_factor = static_cast<int>(*whole);
    s.rife_factor_den = 1;
    return {};
  }
  if (auto real = node->value_exact<double>()) {
    const double halves = *real * 2.0;
    if (halves != std::floor(halves) || halves < 0.0 || halves > 1000.0) {
      return refuse_key(RefusalCode::FieldOutOfRange, "interpolation.rife",
                        std::format(" : {} n'est pas une moitié", *real));
    }
    const int count = static_cast<int>(halves);
    s.rife_factor = count % 2 == 0 ? count / 2 : count;
    s.rife_factor_den = count % 2 == 0 ? 1 : 2;
    return {};
  }
  return refuse_key(RefusalCode::FieldType, "interpolation.rife");
}

const toml::table* section(const toml::table& root, std::string_view name) {
  const toml::node* node = root.get(name);
  return node != nullptr ? node->as_table() : nullptr;
}

}  // namespace

namespace {

std::expected<PlaySettings, Refusal> settings_from_table(const toml::table& root);

}  // namespace

std::expected<PlaySettings, Refusal> parse_settings(std::string_view toml_text) {
  toml::parse_result parsed = toml::parse(toml_text);
  if (!parsed) {
    return refuse(RefusalCode::ProfileUnreadable,
                  std::format("play.toml ligne {} : {}", parsed.error().source().begin.line,
                              parsed.error().description()));
  }
  return settings_from_table(parsed.table());
}

namespace {

// [engines] ; fast_precision, absente d'un fichier d'avant elle, reste cochée.
void read_engines(const toml::table& engines, PlaySettings& s,
                  std::vector<std::expected<void, Refusal>>& reads) {
  reads.push_back(read(engines, "engines", "install_dir", s.engines_dir));
  reads.push_back(read(engines, "engines", "gpu", s.gpu));
  reads.push_back(read(engines, "engines", "fast_precision", s.fast_precision));
}

// [updates], lu par son nom (P5.29) ; un nom inconnu refuse la clé.
void read_updates(const toml::table& updates, PlaySettings& s,
                  std::vector<std::expected<void, Refusal>>& reads) {
  const auto choice = [&](std::string_view key, auto& value, const auto& names) {
    std::string text(name(value));
    reads.push_back(read(updates, "updates", key, text));
    using Choice = std::remove_reference_t<decltype(value)>;
    if (const auto known = choice_named<Choice>(names, text))
      value = *known;
    else
      reads.emplace_back(refuse_key(RefusalCode::FieldOutOfRange, std::format("updates.{}", key),
                                    std::format(" : « {} »", text)));
  };
  // « auto », d'avant la simple notification, se lit « notify » ; channel (canal Préversions retiré) est
  // ignoré.
  if (updates["mode"].value_or(std::string()) == "auto")
    s.update_mode = UpdateMode::Notify;
  else
    choice("mode", s.update_mode, kUpdateModeNames);
  choice("frequency", s.update_frequency, kUpdateFrequencyNames);
}

// [grade] : le grade, la pile de LUT, le limiteur de flashs et le vrai noir.
void read_grade(const toml::table& grade, PlaySettings& s, std::vector<std::expected<void, Refusal>>& reads) {
  std::string preset;  // l'ancienne clé « preset » : lue, puis ignorée
  render::GradeSettings& v = s.grade_values;
  reads.push_back(read(grade, "grade", "enabled", s.grade));
  reads.push_back(read(grade, "grade", "preset", preset));
  reads.push_back(read(grade, "grade", "shadows", v.shadows));
  reads.push_back(read(grade, "grade", "highlights_rtx", v.highlights_rtx));
  reads.push_back(read(grade, "grade", "area_rtx", v.area_rtx));
  reads.push_back(read(grade, "grade", "look_contrast", s.look_contrast));
  reads.push_back(read(grade, "grade", "look_highlight", s.look_highlight));
  reads.push_back(read(grade, "grade", "flash_guard", s.flash_guard));
  reads.push_back(read(grade, "grade", "look_one_table", s.look_table));
  reads.push_back(read(grade, "grade", "grade_before_warp", s.grade_before_warp));
  reads.push_back(read(grade, "grade", "face_skin", s.face_skin));
  reads.push_back(read(grade, "grade", "surface_white", s.surface_white));
  reads.push_back(read(grade, "grade", "anime_contrast", v.anime_contrast));
  reads.push_back(read(grade, "grade", "anime_saturation", v.anime_saturation));
  reads.push_back(read(grade, "grade", "anime_highlights", v.anime_highlights));
  reads.push_back(read_named(grade, "grade", "oled_black", s.oled_black, kOledBlackCount));
  reads.push_back(read(grade, "grade", "oled_black_steps", s.oled_black_steps));
  // Plus de préréglages : les valeurs du fichier font foi.
  // Écrits au centième (format_settings) : relus au centième, la réécriture rend les mêmes réglages.
  const auto hundredth = [](double& x) { x = std::round(x * 100.0) / 100.0; };
  for (double* x : {&v.shadows, &v.highlights_rtx, &v.area_rtx, &s.oled_black_steps}) hundredth(*x);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
std::expected<PlaySettings, Refusal> settings_from_table(const toml::table& root) {
  PlaySettings s;
  std::vector<std::string> legacy_audio;
  std::vector<std::string> legacy_subtitles;
  std::vector<std::expected<void, Refusal>> reads;
  const toml::table* automatic = section(root, "auto");
  if (automatic == nullptr && !root.empty()) s.preset = Preset::Custom;  // play.toml d'avant [auto]
  if (automatic != nullptr) {
    std::string preset(key(s.preset));
    std::string priority(key(s.priority));
    reads.push_back(read(*automatic, "auto", "preset", preset));
    reads.push_back(read(*automatic, "auto", "priority", priority));
    if (const auto found = preset_from_key(preset)) {
      s.preset = *found;
    } else {
      reads.emplace_back(
          refuse_key(RefusalCode::FieldOutOfRange, "auto.preset", std::format(" : « {} »", preset)));
    }
    if (const auto found = priority_from_key(priority)) {
      s.priority = *found;
    } else {
      reads.emplace_back(
          refuse_key(RefusalCode::FieldOutOfRange, "auto.priority", std::format(" : « {} »", priority)));
    }
  }
  if (const toml::table* image = section(root, "image")) {
    reads.push_back(read(*image, "image", "upscale", s.upscale));
    reads.push_back(read(*image, "image", "restore_lines", s.restore_lines));
    reads.push_back(read(*image, "image", "model", s.model));
    reads.push_back(read(*image, "image", "work_height", s.work_height));
    reads.push_back(read_named(*image, "image", "downscale", s.downscale, kDownscaleCount));
    reads.push_back(read_named(*image, "image", "span_precision", s.span_precision, kPrecisionTierCount));
  }
  if (const toml::table* interpolation = section(root, "interpolation")) {
    reads.push_back(read_factor(*interpolation, s));
    reads.push_back(read(*interpolation, "interpolation", "smooth_motion", s.smooth_motion));
    reads.push_back(read(*interpolation, "interpolation", "model", s.rife_model));
    reads.push_back(read(*interpolation, "interpolation", "keep_poses", s.keep_poses));
    reads.push_back(read(*interpolation, "interpolation", "smooth_drawings", s.smooth_drawings));
    reads.push_back(read(*interpolation, "interpolation", "pair_scale", s.pair_scale));
    reads.push_back(read(*interpolation, "interpolation", "rife_fused", s.rife_fused));
    reads.push_back(read(*interpolation, "interpolation", "cubic_warp", s.cubic_warp));
    reads.push_back(read(*interpolation, "interpolation", "drba_fused", s.drba_fused));
    // gmfss_fused, gmfss_quality et gmfss_precision : GMFSS retiré (0.10.0), clés ignorées.
    reads.push_back(read_named(*interpolation, "interpolation", "motion_precision", s.motion_precision,
                               kPrecisionTierCount));
    reads.push_back(read(*interpolation, "interpolation", "auto", s.rife_auto));
    reads.push_back(read(*interpolation, "interpolation", "auto_cap", s.auto_cap));
    read_auto_mode(*interpolation, s, reads);
    // span_order ; les fichiers d'avant portent warp_after_span, qui ne compte que s'il est vrai (faux
    // était le défaut, écrit partout : il laisse Auto).
    bool legacy_after = false;
    reads.push_back(read(*interpolation, "interpolation", "warp_after_span", legacy_after));
    if (legacy_after) s.span_order = SpanOrder::After;
    reads.push_back(read_named(*interpolation, "interpolation", "span_order", s.span_order, kSpanOrderCount));
    // Le flux optique matériel des RTX et GMFSS (P5.2, 0.10.0) ont été retirés : RIFE les remplace, sans
    // plainte.
    constexpr std::array<std::string_view, 3> kRetiredMotion{"optical_flow", "gmfss", "gmfss_union"};
    reads.push_back(read_named(*interpolation, "interpolation", "engine", s.motion_engine, kMotionEngineCount,
                               kRetiredMotion, MotionEngine::Rife));
  }
  // Le pic visé par le grade HDR. Le mode HDR n'est plus un réglage (il suit le système) : hdr.engine et la
  // section rtx_hdr d'avant (RTX HDR retiré, sauf son pic) sont ignorés, comme les clés du HDR adaptatif.
  if (const toml::table* hdr = section(root, "hdr")) {
    reads.push_back(read(*hdr, "hdr", "peak", s.true_hdr_peak));
  } else if (const toml::table* rtx = section(root, "rtx_hdr")) {
    reads.push_back(read(*rtx, "rtx_hdr", "peak", s.true_hdr_peak));
  }
  if (const toml::table* audio = section(root, "audio")) {
    reads.push_back(read(*audio, "audio", "delay_ms", s.audio_delay_ms));
    reads.push_back(read(*audio, "audio", "volume", s.volume));
    reads.push_back(read(*audio, "audio", "device", s.audio_device));
    if (const toml::table* delays = (*audio)["delays"].as_table()) {
      // Une entrée d'un autre type ou hors bornes est écartée seule.
      for (const auto& [device, value] : *delays) {
        const auto ms = value.value_exact<std::int64_t>();
        if (ms && *ms >= limits::kAudioDelayMs.lo && *ms <= limits::kAudioDelayMs.hi)
          s.audio_device_delays_ms[std::string(device.str())] = static_cast<int>(*ms);
      }
    }
    // listening, faithful, profile, anime, hrir… : le traitement du son retiré, ces clés sont ignorées.
    reads.push_back(read_legacy_languages(*audio, "audio", legacy_audio));
  }
  if (const toml::table* sync = section(root, "sync")) {
    reads.push_back(read(*sync, "sync", "display", s.display_sync));
    reads.push_back(read(*sync, "sync", "video_latency_ms", s.video_latency_ms));
  }
  if (const toml::table* picture = section(root, "picture")) {
    reads.push_back(read(*picture, "picture", "deband", s.deband));
    reads.push_back(read(*picture, "picture", "sharpness", s.sharpness));
    std::string scaler(name(s.final_scaler));
    reads.push_back(read(*picture, "picture", "scaler", scaler));
    bool scaler_known = false;
    for (int i = 0; i < kFinalScalerCount; ++i) {
      if (name(static_cast<FinalScaler>(i)) == scaler) {
        s.final_scaler = static_cast<FinalScaler>(i);
        scaler_known = true;
      }
    }
    if (!scaler_known)
      reads.emplace_back(
          refuse_key(RefusalCode::FieldOutOfRange, "picture.scaler", std::format(" : « {} »", scaler)));
  }
  if (const toml::table* interface = section(root, "interface")) {
    reads.push_back(read(*interface, "interface", "language", s.language));
    if (s.language != "en" && s.language != "fr")
      reads.emplace_back(refuse_key(RefusalCode::FieldOutOfRange, "interface.language",
                                    std::format(" : « {} »", s.language)));
  }
  if (const toml::table* engines = section(root, "engines")) read_engines(*engines, s, reads);
  if (const toml::table* updates = section(root, "updates")) read_updates(*updates, s, reads);
  // [span_prepare] background et [default_player] dismissed, d'avant 0.10, ne servaient à rien : ignorés.
  if (const toml::table* playlist = section(root, "playlist"))
    reads.push_back(read(*playlist, "playlist", "autoplay_next", s.autoplay_next));
  if (const toml::table* subtitles = section(root, "subtitles")) {
    reads.push_back(read(*subtitles, "subtitles", "scale", s.subtitle_scale));
    reads.push_back(read_legacy_languages(*subtitles, "subtitles", legacy_subtitles));
  }
  if (const toml::table* languages = section(root, "languages")) {
    reads.push_back(read(*languages, "languages", "audio_1", s.language_presets[0].audio));
    reads.push_back(read(*languages, "languages", "subtitles_1", s.language_presets[0].subtitles));
    reads.push_back(read(*languages, "languages", "audio_2", s.language_presets[1].audio));
    reads.push_back(read(*languages, "languages", "subtitles_2", s.language_presets[1].subtitles));
  } else if (!legacy_audio.empty() || !legacy_subtitles.empty()) {
    // Le format d'avant : deux listes, converties en deux préréglages (rang 1, puis rang 2).
    const auto rank = [](const std::vector<std::string>& list, std::size_t i) {
      return i < list.size() ? list[i] : std::string{};
    };
    s.language_presets = {
        {rank(legacy_audio, 0), rank(legacy_subtitles, 0)},
        {legacy_audio.size() > 1 ? legacy_audio[1] : rank(legacy_audio, 0), rank(legacy_subtitles, 1)}};
  }
  if (const toml::table* oled = section(root, "oled")) {
    reads.push_back(read(*oled, "oled", "care", s.oled_care));
    reads.push_back(read(*oled, "oled", "steady_cadence", s.steady_cadence));
  }
  if (const toml::table* grade = section(root, "grade")) read_grade(*grade, s, reads);
  if (const toml::table* stats = section(root, "stats")) {
    reads.push_back(read(*stats, "stats", "all", s.stats.all));
    reads.push_back(read(*stats, "stats", "chain", s.stats.chain));
    reads.push_back(read(*stats, "stats", "frames", s.stats.frames));
    reads.push_back(read(*stats, "stats", "rife", s.stats.rife));
    reads.push_back(read(*stats, "stats", "tracks", s.stats.tracks));
    reads.push_back(read(*stats, "stats", "output", s.stats.output));
    reads.push_back(read(*stats, "stats", "position", s.stats.position));
    reads.push_back(read(*stats, "stats", "memory", s.stats.memory));
  }
  for (const auto& done : reads) {
    if (!done) return std::unexpected(done.error());
  }
  return validate(s);
}

}  // namespace

std::expected<PlaySettings, Refusal> parse_settings_lenient(std::string_view toml_text,
                                                            std::vector<std::string>* dropped) {
  toml::parse_result parsed = toml::parse(toml_text);
  if (!parsed) {
    return refuse(RefusalCode::ProfileUnreadable,
                  std::format("play.toml ligne {} : {}", parsed.error().source().begin.line,
                              parsed.error().description()));
  }
  toml::table root = std::move(parsed).table();
  return read_settings_lenient(root, dropped);
}

std::expected<PlaySettings, Refusal> read_settings_lenient(toml::table& root,
                                                           std::vector<std::string>* dropped) {
  // Une valeur fautive est écartée seule (le défaut la remplace) : le reste du fichier reste en vigueur.
  constexpr int kMaxDropped = 64;
  for (int attempt = 0; attempt <= kMaxDropped; ++attempt) {
    auto settings = settings_from_table(root);
    if (settings) return settings;
    const Refusal& refusal = settings.error();
    const std::size_t dot = refusal.key.find('.');
    toml::table* table = nullptr;
    std::string key_name;
    if (dot != std::string::npos && dot > 0 && dot + 1 < refusal.key.size()) {
      toml::node* node = root.get(refusal.key.substr(0, dot));
      table = node != nullptr ? node->as_table() : nullptr;
      key_name = refusal.key.substr(dot + 1);
    }
    if (table != nullptr && table->erase(key_name)) {
      if (dropped != nullptr) dropped->push_back(refusal.detail);
      continue;
    }
    // Le refus ne nomme pas une clé du fichier (une règle entre deux valeurs, une liste) : la première clé
    // dont le retrait le lève est écartée, puis, à défaut, la première section. Une nouvelle version du
    // lecteur ne remet jamais tout le fichier aux défauts pour une valeur qu'elle n'accepte plus.
    if (!drop_offending(root, refusal, dropped, settings_from_table)) return settings;
  }
  return settings_from_table(root);
}

namespace {

// Ce que ce processus a lu ou écrit en dernier pour chaque fichier, mis en forme (P2.14) : la base de la
// fusion à la réécriture, quand un autre lancement a réécrit le fichier entre-temps.
struct SeenSettings {
  std::mutex mutex;
  std::map<std::filesystem::path, toml::table> tables;
  std::set<std::filesystem::path> unreadable;  // lus sans succès : jamais réécrits (B22-02)
};

SeenSettings& seen() {
  static SeenSettings instance;
  return instance;
}

void remember(const std::filesystem::path& path, toml::table table) {
  SeenSettings& s = seen();
  const std::scoped_lock lock(s.mutex);
  s.tables.insert_or_assign(path, std::move(table));
}

std::optional<toml::table> remembered(const std::filesystem::path& path) {
  SeenSettings& s = seen();
  const std::scoped_lock lock(s.mutex);
  const auto found = s.tables.find(path);
  if (found == s.tables.end()) return std::nullopt;
  return found->second;
}

void set_unreadable(const std::filesystem::path& path, bool unreadable) {
  SeenSettings& s = seen();
  const std::scoped_lock lock(s.mutex);
  if (unreadable)
    s.unreadable.insert(path);
  else
    s.unreadable.erase(path);
}

bool is_unreadable(const std::filesystem::path& path) {
  SeenSettings& s = seen();
  const std::scoped_lock lock(s.mutex);
  return s.unreadable.contains(path);
}

// Le texte du fichier, absent (nullopt) ou refusé. Un verrou passager (antivirus, indexeur sous Windows : le
// fichier est ouvert sans partage de lecture) se réessaie quelques fois ; un fichier qui reste illisible est
// refusé, jamais lu comme « absent » (B22-02).
std::expected<std::optional<std::string>, Refusal> read_file(const std::filesystem::path& path) {
  // Un play.toml normal tient dans quelques Ko : au-delà d'un Mo, ce n'est pas le nôtre, et la lecture lâche
  // (une copie de la table par clé) coûterait des minutes. Refusé, nommé, jamais écrasé (C22-07).
  constexpr std::uintmax_t kMaxBytes = std::uintmax_t{1} << 20;
  std::error_code size_error;
  if (const auto size = std::filesystem::file_size(path, size_error); !size_error && size > kMaxBytes)
    return refuse(RefusalCode::FileUnreadable,
                  std::format("{} : {} octets, au-delà de {} (pas un fichier de réglages)",
                              core::to_utf8(path), size, kMaxBytes));
  constexpr int kAttempts = 4;
  constexpr auto kPause = std::chrono::milliseconds(120);
  auto text = core::read_text_file_if_present(path);
  for (int attempt = 1; !text && attempt < kAttempts; ++attempt) {
    std::this_thread::sleep_for(kPause);
    text = core::read_text_file_if_present(path);
  }
  return text;
}

// Un play.toml au TOML invalide va être remplacé : une copie le garde d'abord (« play.toml.invalide-AAAAMMJJ-
// HHMMSS »), pour qu'une édition à la main faite en séance ne se perde pas (B22-04). Refusé sans copie.
std::expected<void, Refusal> keep_invalid_copy(const std::filesystem::path& path) {
  // Au plus kKept copies, les plus récentes ; une copie identique déjà gardée suffit (C22-06) : un disque qui
  // refuse l'écriture ne remplit pas le dossier d'une copie toutes les dix secondes.
  constexpr std::size_t kKept = 5;
  const std::string prefix = core::to_utf8(path.filename()) + ".invalide-";
  std::vector<std::filesystem::path> copies;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(path.parent_path(), error)) {
    if (core::to_utf8(entry.path().filename()).starts_with(prefix)) copies.push_back(entry.path());
  }
  std::ranges::sort(copies);  // l'horodatage trie les noms
  const auto current = core::read_text_file_if_present(path);
  for (const std::filesystem::path& copy : copies) {
    const auto kept = core::read_text_file_if_present(copy);
    if (current && kept && *current == *kept) return {};
  }
  const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
  std::filesystem::path aside = path;
  aside += std::format(".invalide-{:%Y%m%d-%H%M%S}", now);
  std::filesystem::copy_file(path, aside, std::filesystem::copy_options::overwrite_existing, error);
  if (error)
    return refuse(RefusalCode::FileUnwritable, std::format("{} : copie de l'ancien fichier impossible ({})",
                                                           core::to_utf8(path), error.message()));
  log_channel("app")->warn("{} invalide : copie gardée dans {}", core::to_utf8(path.filename()),
                           core::to_utf8(aside.filename()));
  prune_invalid_copies(path, kKept);  // C36-10 : le plafond commun, ici celui de C22-06
  return {};
}

}  // namespace

std::expected<PlaySettings, Refusal> load_settings(const std::filesystem::path& path,
                                                   std::vector<std::string>* ignored) {
  auto read_text = read_file(path);
  if (!read_text) {
    // Réglages par défaut pour la séance, mais le fichier n'est pas à eux : save_settings ne l'écrase pas.
    set_unreadable(path, true);
    return std::unexpected(read_text.error());
  }
  // Lu, ou absent : le verrou passager est tombé, les écritures reprennent (C22-01, C36-01).
  set_unreadable(path, false);
  if (!*read_text) return PlaySettings{};  // absent
  const std::string& text = **read_text;
  std::vector<std::string> dropped;
  auto settings = parse_settings_lenient(text, &dropped);
  if (const toml::parse_result parsed = toml::parse(text); parsed) {
    // La version lue (P2.22) : une plus récente garde ses clés à la réécriture, et se dit au journal.
    if (const auto version = parsed.table()["version"].value<std::int64_t>();
        version && *version > kSettingsVersion)
      log_channel("app")->warn("{} : version {} plus récente que {}, clés inconnues gardées",
                               core::to_utf8(path), *version, kSettingsVersion);
  }
  for (const std::string& why : dropped)
    log_channel("app")->warn("{} : valeur écartée, défaut gardé ({})", core::to_utf8(path), why);
  if (ignored != nullptr) *ignored = std::move(dropped);
  if (!settings)
    return std::unexpected(
        Refusal{.code = settings.error().code,
                .detail = std::format("{} : {}", core::to_utf8(path), settings.error().detail),
                .key = settings.error().key});
  if (toml::parse_result seen_now = toml::parse(format_settings(*settings)); seen_now)
    remember(path, std::move(seen_now).table());
  return settings;
}

std::expected<core::Settings, Refusal> load_engine_settings(const std::filesystem::path& path,
                                                            std::vector<std::string>* skipped) {
  if (auto text = read_file(path); !text) return std::unexpected(text.error());
  return core::load_settings(path, skipped);
}

std::expected<PlaySettings, Refusal> peek_settings(const std::filesystem::path& path) {
  auto text = core::read_text_file_if_present(path);
  if (!text) return std::unexpected(text.error());
  if (!*text) return PlaySettings{};
  return parse_settings_lenient(**text);
}

namespace {

// L'ordre des écritures de play.toml (C30-03) : le fil « écriture » et le thread principal écrivent le même
// fichier. Une écriture n'a lieu que sous ce verrou, et jamais par-dessus un instantané plus récent.
struct SaveOrder {
  std::mutex mutex;
  std::atomic<std::uint64_t> next{1};
  std::map<std::filesystem::path, std::uint64_t> written;
};

SaveOrder& save_order() {
  static SaveOrder order;
  return order;
}

std::expected<void, Refusal> save_settings_locked(const std::filesystem::path& path,
                                                  const PlaySettings& settings);

}  // namespace

std::uint64_t settings_ticket() {
  return save_order().next.fetch_add(1, std::memory_order_relaxed);
}

std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const PlaySettings& settings) {
  return save_settings(path, settings, settings_ticket());
}

std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const PlaySettings& settings,
                                           std::uint64_t ticket) {
  SaveOrder& order = save_order();
  const std::scoped_lock lock(order.mutex);
  if (const auto at = order.written.find(path); at != order.written.end() && at->second > ticket) {
    log_channel("app")->debug("{} : écriture plus ancienne que la dernière, sautée",
                              core::to_utf8(path.filename()));
    return {};
  }
  auto written = save_settings_locked(path, settings);
  if (written) order.written[path] = ticket;
  return written;
}

namespace {

std::expected<void, Refusal> save_settings_locked(const std::filesystem::path& path,
                                                  const PlaySettings& settings) {
  if (auto valid = validate(settings); !valid) return std::unexpected(valid.error());
  // Un fichier qui ne s'est pas lu au lancement : la séance tourne sur les défauts, les écrire le
  // remplacerait (B22-02).
  // C36-01 : relu à chaque essai ; un refus passager (antivirus, indexeur) ne bloque plus toute la séance.
  if (is_unreadable(path)) {
    if (auto again = read_file(path); !again)
      return refuse(RefusalCode::FileUnwritable,
                    std::format("{} : illisible au lancement, non réécrit", core::to_utf8(path)));
    set_unreadable(path, false);
    log_channel("app")->info("{} se lit de nouveau : les changements de la séance y sont écrits",
                             core::to_utf8(path.filename()));
    // La séance a démarré sur les défauts : ils sont la base de la fusion, et seules les valeurs changées
    // depuis remplacent celles du fichier.
    if (!remembered(path))
      if (toml::parse_result base = toml::parse(format_settings(PlaySettings{})); base)
        remember(path, std::move(base).table());
  }
  std::string text = format_settings(settings);
  // Mis en forme et lu une seule fois : la fusion et la base de la suivante partent de la même table
  // (A22-16).
  toml::parse_result fresh = toml::parse(text);
  // Les clés d'une version plus récente restent (P2.22) : revenir à celle-ci ne les efface plus. Un fichier
  // qui existe sans se lire n'est pas écrasé : l'écriture est refusée, et reprise au tour suivant.
  auto existing = read_file(path);
  if (!existing) return std::unexpected(existing.error());
  set_unreadable(path, false);
  if (*existing) {
    toml::parse_result old = toml::parse(**existing);
    if (!old) {
      if (auto copy = keep_invalid_copy(path); !copy) return copy;
    }
    if (old && fresh) {
      toml::table merged = fresh.table();
      // Le fichier lu comme au chargement : une valeur qu'il ne connaît pas (version plus récente) est
      // écartée, sans faire échouer la base de keep_unknown_keys (A36-01).
      toml::table pruned = old.table();
      const auto on_disk = read_settings_lenient(pruned, nullptr);
      toml::parse_result disk = on_disk ? toml::parse(format_settings(*on_disk)) : toml::parse_result{};
      // Réécrit entre-temps par un autre lancement (P2.14) : ses clés restent, seules les nôtres changent.
      if (const auto before = remembered(path); before && disk) {
        merged = disk.table();
        apply_own_changes(merged, *before, fresh.table());
      }
      if (on_disk) {
        keep_unknown_keys(merged, pruned, settings_from_table);
        // La valeur écartée revient telle quelle tant que ce processus ne l'a pas changée (A22-06).
        if (disk) keep_rejected_values(merged, old.table(), pruned, disk.table());
      }
      // Jamais abaissée : la version récente, relancée, sait qu'un lecteur plus ancien est passé (A36-01).
      if (const auto version = old.table()["version"].value<std::int64_t>();
          version && *version > kSettingsVersion)
        merged.insert_or_assign("version", *version);
      std::ostringstream out;
      out << "# Réglages de « ajp play », écrits par le menu des paramètres.\n" << merged << '\n';
      text = out.str();
    }
  }
  auto written = core::write_text_atomically(path, text);
  // La base de la fusion suivante ne change qu'une fois l'écriture faite : refusée, la suivante reporte
  // encore ce changement (A22-02).
  if (!written) return written;
  if (fresh) remember(path, std::move(fresh).table());
  return written;
}

}  // namespace

}  // namespace ajp::app::play
