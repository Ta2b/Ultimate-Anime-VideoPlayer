// Le fichier play.toml des réglages : lecture (parse_settings, load_settings), écriture
// (format_settings, save_settings). Les valeurs lues passent par validate() (settings.cpp).
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "app/play/settings.h"
#include "core/user_overlay.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {
namespace {

constexpr std::int64_t kSettingsVersion = 1;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

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
  return refuse(RefusalCode::FieldType, std::format("{}.{}", section, key));
}

// Une liste de langues : un tableau de balises (« ["fre", "eng"] ») ; FieldType sinon.
std::expected<void, Refusal> read_languages(const toml::table& table, std::string_view section,
                                            std::vector<std::string>& out) {
  const toml::node* node = table.get("languages");
  if (node == nullptr) return {};
  const toml::array* list = node->as_array();
  if (list == nullptr) return refuse(RefusalCode::FieldType, std::format("{}.languages", section));
  std::vector<std::string> languages;
  for (const toml::node& item : *list) {
    const std::optional<std::string> tag = item.value_exact<std::string>();
    if (!tag) return refuse(RefusalCode::FieldType, std::format("{}.languages", section));
    languages.push_back(*tag);
  }
  out = std::move(languages);
  return {};
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
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("interpolation.rife : {} n'est pas une moitié", *real));
    }
    const int count = static_cast<int>(halves);
    s.rife_factor = count % 2 == 0 ? count / 2 : count;
    s.rife_factor_den = count % 2 == 0 ? 1 : 2;
    return {};
  }
  return refuse(RefusalCode::FieldType, "interpolation.rife");
}

const toml::table* section(const toml::table& root, std::string_view name) {
  const toml::node* node = root.get(name);
  return node != nullptr ? node->as_table() : nullptr;
}

}  // namespace

std::expected<PlaySettings, Refusal> parse_settings(std::string_view toml_text) {
  toml::parse_result parsed = toml::parse(toml_text);
  if (!parsed) {
    return refuse(RefusalCode::ProfileUnreadable,
                  std::format("play.toml ligne {} : {}", parsed.error().source().begin.line,
                              parsed.error().description()));
  }
  const toml::table& root = parsed.table();
  PlaySettings s;
  std::vector<std::expected<void, Refusal>> reads;
  if (const toml::table* image = section(root, "image")) {
    std::string downscale(name(s.downscale));
    reads.push_back(read(*image, "image", "upscale", s.upscale));
    reads.push_back(read(*image, "image", "model", s.model));
    reads.push_back(read(*image, "image", "work_height", s.work_height));
    reads.push_back(read(*image, "image", "downscale", downscale));
    bool known = false;
    for (int i = 0; i < kDownscaleCount; ++i) {
      const auto filter = static_cast<Downscale>(i);
      if (name(filter) == downscale) {
        s.downscale = filter;
        known = true;
      }
    }
    if (!known)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("image.downscale : « {} »", downscale)));
  }
  if (const toml::table* interpolation = section(root, "interpolation")) {
    reads.push_back(read_factor(*interpolation, s));
    reads.push_back(read(*interpolation, "interpolation", "smooth_motion", s.smooth_motion));
    reads.push_back(read(*interpolation, "interpolation", "model", s.rife_model));
    reads.push_back(read(*interpolation, "interpolation", "keep_poses", s.keep_poses));
    // span_order ; les fichiers d'avant portent warp_after_span, qui ne compte que s'il est vrai (faux
    // était le défaut, écrit partout : il laisse Auto).
    bool legacy_after = false;
    reads.push_back(read(*interpolation, "interpolation", "warp_after_span", legacy_after));
    if (legacy_after) s.span_order = SpanOrder::After;
    std::string order(name(s.span_order));
    reads.push_back(read(*interpolation, "interpolation", "span_order", order));
    bool order_known = false;
    for (int i = 0; i < kSpanOrderCount; ++i) {
      if (name(static_cast<SpanOrder>(i)) == order) {
        s.span_order = static_cast<SpanOrder>(i);
        order_known = true;
      }
    }
    if (!order_known)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("interpolation.span_order : « {} »", order)));
    std::string engine(name(s.motion_engine));
    reads.push_back(read(*interpolation, "interpolation", "engine", engine));
    // Le flux optique matériel des RTX a été retiré : RIFE le remplace, sans plainte.
    bool engine_known = engine == "optical_flow";
    if (engine_known) s.motion_engine = MotionEngine::Rife;
    for (int i = 0; i < kMotionEngineCount; ++i) {
      if (name(static_cast<MotionEngine>(i)) == engine) {
        s.motion_engine = static_cast<MotionEngine>(i);
        engine_known = true;
      }
    }
    if (!engine_known)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("interpolation.engine : « {} »", engine)));
  }
  // Le mode HDR de l'application : son moteur (RTX HDR sur NVIDIA, HDR adaptatif partout, ou SDR)
  // et les réglages qu'ils partagent. Les fichiers d'avant (section rtx_hdr) se lisent encore.
  if (const toml::table* hdr = section(root, "hdr")) {
    std::string engine(hdr_engine_key(s));
    reads.push_back(read(*hdr, "hdr", "engine", engine));
    if (engine == "rtx" || engine == "adaptive" || engine == "sdr") {
      s.true_hdr = engine == "rtx";
      s.hdr_native = engine == "adaptive";
    } else {
      reads.emplace_back(refuse(RefusalCode::FieldOutOfRange, std::format("hdr.engine : « {} »", engine)));
    }
    reads.push_back(read(*hdr, "hdr", "contrast", s.true_hdr_contrast));
    reads.push_back(read(*hdr, "hdr", "saturation", s.true_hdr_saturation));
    reads.push_back(read(*hdr, "hdr", "middle_gray", s.true_hdr_gray));
    reads.push_back(read(*hdr, "hdr", "peak", s.true_hdr_peak));
    reads.push_back(read(*hdr, "hdr", "debanding", s.true_hdr_debanding));
  } else if (const toml::table* rtx = section(root, "rtx_hdr")) {
    reads.push_back(read(*rtx, "rtx_hdr", "enabled", s.true_hdr));
    reads.push_back(read(*rtx, "rtx_hdr", "contrast", s.true_hdr_contrast));
    reads.push_back(read(*rtx, "rtx_hdr", "saturation", s.true_hdr_saturation));
    reads.push_back(read(*rtx, "rtx_hdr", "middle_gray", s.true_hdr_gray));
    reads.push_back(read(*rtx, "rtx_hdr", "peak", s.true_hdr_peak));
    reads.push_back(read(*rtx, "rtx_hdr", "debanding", s.true_hdr_debanding));
    reads.push_back(read(*rtx, "rtx_hdr", "native", s.hdr_native));
  }
  if (const toml::table* audio = section(root, "audio")) {
    reads.push_back(read(*audio, "audio", "delay_ms", s.audio_delay_ms));
    reads.push_back(read(*audio, "audio", "volume", s.volume));
    reads.push_back(read_languages(*audio, "audio", s.audio_languages));
  }
  if (const toml::table* sync = section(root, "sync")) {
    reads.push_back(read(*sync, "sync", "display", s.display_sync));
    reads.push_back(read(*sync, "sync", "video_latency_ms", s.video_latency_ms));
  }
  if (const toml::table* picture = section(root, "picture")) {
    reads.push_back(read(*picture, "picture", "deband", s.deband));
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
      reads.emplace_back(refuse(RefusalCode::FieldOutOfRange, std::format("picture.scaler : « {} »", scaler)));
  }
  if (const toml::table* subtitles = section(root, "subtitles")) {
    reads.push_back(read(*subtitles, "subtitles", "scale", s.subtitle_scale));
    reads.push_back(read_languages(*subtitles, "subtitles", s.subtitle_languages));
  }
  if (const toml::table* oled = section(root, "oled")) {
    reads.push_back(read(*oled, "oled", "care", s.oled_care));
    reads.push_back(read(*oled, "oled", "steady_cadence", s.steady_cadence));
  }
  if (const toml::table* grade = section(root, "grade")) {
    std::string preset(render::preset_key(s.grade_preset));
    render::GradeSettings& v = s.grade_values;
    reads.push_back(read(*grade, "grade", "enabled", s.grade));
    reads.push_back(read(*grade, "grade", "preset", preset));
    reads.push_back(read(*grade, "grade", "shadows", v.shadows));
    reads.push_back(read(*grade, "grade", "highlights", v.highlights));
    reads.push_back(read(*grade, "grade", "highlights_rtx", v.highlights_rtx));
    reads.push_back(read(*grade, "grade", "vibrance", v.vibrance));
    reads.push_back(read(*grade, "grade", "vibrance_rtx", v.vibrance_rtx));
    reads.push_back(read(*grade, "grade", "area_rtx", v.area_rtx));
    reads.push_back(read(*grade, "grade", "guard", v.guard));
    reads.push_back(read(*grade, "grade", "expand", v.expand));
    reads.push_back(read(*grade, "grade", "gamut_p3", v.gamut_p3));
    std::string oled(name(s.oled_black));
    reads.push_back(read(*grade, "grade", "oled_black", oled));
    reads.push_back(read(*grade, "grade", "oled_black_steps", s.oled_black_steps));
    bool oled_known = false;
    for (int i = 0; i < kOledBlackCount; ++i) {
      if (name(static_cast<OledBlack>(i)) == oled) {
        s.oled_black = static_cast<OledBlack>(i);
        oled_known = true;
      }
    }
    if (!oled_known)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("grade.oled_black : « {} »", oled)));
    bool known = false;
    for (int i = 0; i < render::kGradePresetCount; ++i) {
      const auto candidate = static_cast<render::GradePreset>(i);
      if (render::preset_key(candidate) == preset) {
        s.grade_preset = candidate;
        known = true;
      }
    }
    if (!known)
      reads.emplace_back(refuse(RefusalCode::FieldOutOfRange, std::format("grade.preset : « {} »", preset)));
    s.grade_values = with_preset(s.grade_preset, v);
  }
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

std::string format_settings(const PlaySettings& s) {
  toml::table root;
  root.insert("version", kSettingsVersion);
  root.insert("image", toml::table{{"upscale", s.upscale},
                                   {"model", s.model},
                                   {"work_height", s.work_height},
                                   {"downscale", std::string(name(s.downscale))}});
  toml::table interpolation{{"engine", std::string(name(s.motion_engine))},
                            {"model", s.rife_model},
                            {"keep_poses", s.keep_poses},
                            {"span_order", std::string(name(s.span_order))},
                            {"smooth_motion", s.smooth_motion}};
  if (s.rife_factor_den == 2) {
    interpolation.insert("rife", static_cast<double>(s.rife_factor) / 2.0);
  } else {
    interpolation.insert("rife", s.rife_factor);
  }
  root.insert("interpolation", std::move(interpolation));
  root.insert("hdr", toml::table{{"engine", std::string(hdr_engine_key(s))},
                                 {"contrast", s.true_hdr_contrast},
                                 {"saturation", s.true_hdr_saturation},
                                 {"middle_gray", s.true_hdr_gray},
                                 {"peak", s.true_hdr_peak},
                                 {"debanding", s.true_hdr_debanding}});
  // Le volume au centième : 0.85 et non 0.8500000238418579.
  const auto languages = [](const std::vector<std::string>& tags) {
    toml::array list;
    for (const std::string& tag : tags) list.push_back(tag);
    return list;
  };
  root.insert("audio", toml::table{{"delay_ms", s.audio_delay_ms},
                                   {"volume", std::round(static_cast<double>(s.volume) * 100.0) / 100.0},
                                   {"languages", languages(s.audio_languages)}});
  root.insert("sync", toml::table{{"display", s.display_sync}, {"video_latency_ms", s.video_latency_ms}});
  root.insert("picture", toml::table{{"deband", s.deband}, {"scaler", std::string(name(s.final_scaler))}});
  root.insert("subtitles",
              toml::table{{"scale", std::round(static_cast<double>(s.subtitle_scale) * 100.0) / 100.0},
                          {"languages", languages(s.subtitle_languages)}});
  // Les réglages du grade au centième, comme le menu les montre.
  const auto hundredth = [](double v) { return std::round(v * 100.0) / 100.0; };
  const render::GradeSettings& g = s.grade_values;
  root.insert("grade", toml::table{{"enabled", s.grade},
                                   {"preset", std::string(render::preset_key(s.grade_preset))},
                                   {"shadows", hundredth(g.shadows)},
                                   {"highlights", hundredth(g.highlights)},
                                   {"highlights_rtx", hundredth(g.highlights_rtx)},
                                   {"vibrance", hundredth(g.vibrance)},
                                   {"vibrance_rtx", hundredth(g.vibrance_rtx)},
                                   {"area_rtx", hundredth(g.area_rtx)},
                                   {"guard", hundredth(g.guard)},
                                   {"expand", hundredth(g.expand)},
                                   {"gamut_p3", g.gamut_p3},
                                   {"oled_black", std::string(name(s.oled_black))},
                                   {"oled_black_steps", hundredth(s.oled_black_steps)}});
  root.insert("oled", toml::table{{"care", s.oled_care}, {"steady_cadence", s.steady_cadence}});
  root.insert("stats", toml::table{{"all", s.stats.all},
                                   {"chain", s.stats.chain},
                                   {"frames", s.stats.frames},
                                   {"rife", s.stats.rife},
                                   {"tracks", s.stats.tracks},
                                   {"output", s.stats.output},
                                   {"position", s.stats.position},
                                   {"memory", s.stats.memory}});
  std::ostringstream out;
  out << "# Réglages de « ajp play », écrits par le menu des paramètres.\n" << root << '\n';
  return out.str();
}

std::expected<PlaySettings, Refusal> load_settings(const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::exists(path, error)) return PlaySettings{};
  std::ifstream file(path, std::ios::binary);
  if (!file) return refuse(RefusalCode::FileUnreadable, path.string());
  std::ostringstream text;
  text << file.rdbuf();
  auto settings = parse_settings(text.str());
  if (!settings)
    return refuse(settings.error().code, std::format("{} : {}", path.string(), settings.error().detail));
  return settings;
}

std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const PlaySettings& settings) {
  if (auto valid = validate(settings); !valid) return std::unexpected(valid.error());
  return core::write_text_atomically(path, format_settings(settings));
}

}  // namespace ajp::app::play
