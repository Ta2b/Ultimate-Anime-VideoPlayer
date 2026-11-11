#include "render/grade.h"

#include <algorithm>
#include <array>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "render/output_tune.h"
#include "render/shaders.h"

namespace ajp::render {
namespace {

// Un //!PARAM et le champ de GradeParams qui le règle.
struct ParamBinding {
  std::string_view name;
  double GradeParams::* field = nullptr;
};

double value_of(const GradeParams& params, const ParamBinding& binding) {
  return params.*binding.field;
}

constexpr std::array kGradeParams = {
    ParamBinding{"peak", &GradeParams::peak},
    ParamBinding{"black_lo", &GradeParams::black_lo},
    ParamBinding{"black_hi", &GradeParams::black_hi},
    ParamBinding{"paper", &GradeParams::paper},
    ParamBinding{"shadows", &GradeParams::shadows},
    ParamBinding{"shadow_knee", &GradeParams::shadow_knee},
    ParamBinding{"spec_gain", &GradeParams::spec_gain},
    ParamBinding{"adapt", &GradeParams::adapt},
    ParamBinding{"sdr_white", &GradeParams::sdr_white},
    ParamBinding{"sdr_out", &GradeParams::sdr_out},
    ParamBinding{"sdr_tm", &GradeParams::sdr_tm},
    ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
    ParamBinding{"chroma", &GradeParams::chroma},
    ParamBinding{"pivot", &GradeParams::pivot},
    ParamBinding{"fs_apl", &GradeParams::fs_apl},
    ParamBinding{"fs_series", &GradeParams::fs_series},
    ParamBinding{"an_lift", &GradeParams::an_lift},
    ParamBinding{"an_punch", &GradeParams::an_punch},
    ParamBinding{"an_chroma", &GradeParams::an_chroma},
    ParamBinding{"an_spec", &GradeParams::an_spec},
};

constexpr std::array kLookParams = {
    ParamBinding{"look_white", &GradeParams::look_white}, ParamBinding{"look_pq", &GradeParams::look_pq},
    ParamBinding{"look_clean", &GradeParams::look_clean}, ParamBinding{"sdr_out", &GradeParams::sdr_out},
    ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},   ParamBinding{"peak", &GradeParams::peak},
};

// Sortie SDR de la méthode anime (GradeParams::sdr_tm) : la première passe met l'image SDR de la cible en PQ
// BT.2020 (son blanc à sdr_white), toutes les passes suivantes travaillent comme vers un écran HDR ; la
// dernière (anime-hdr-black.hook) la ramène au SDR.
constexpr std::string_view kSdrToPq = R"(
//!HOOK OUTPUT
//!BIND HOOKED
//!WHEN sdr_tm
//!DESC AnimeJaNai HDR anime - image SDR en PQ (sortie SDR)

vec4 hook() {
    vec4 px = HOOKED_tex(HOOKED_pos);
    const mat3 to2020 = mat3(0.6274040, 0.0690970, 0.0163916,
                             0.3292820, 0.9195400, 0.0880132,
                             0.0433136, 0.0113612, 0.8955950);
    vec3 n = to2020 * (pow(clamp(px.rgb, 0.0, 1.0), vec3(sdr_gamma)) * sdr_white);
    vec3 q = pow(clamp(n / 10000.0, 0.0, 1.0), vec3(0.1593017578125));
    px.rgb = pow((0.8359375 + 18.8515625 * q) / (1.0 + 18.6875 * q), vec3(78.84375));
    return px;
}
)";

// Le nom du paramètre ou de la texture du `slot`-ième emplacement (0 : le premier, sans numéro).
std::string slot_name(std::string_view base, std::size_t slot) {
  return slot == 0 ? std::string(base) : std::format("{}{}", base, slot + 1);
}

// Le texte du hook du look pour les LUT posées. Les marques //@LOOK_...@ du fichier reçoivent les intensités
// (un //!PARAM par emplacement, réglable sans relire le hook), les textures et la chaîne : chaque emplacement
// occupé lit sa LUT et s'y mélange, dans l'ordre. Deux emplacements de même contenu partagent une texture
// (la pile par défaut : 11 emplacements, 4 textures). Les blocs //!TEXTURE vont derrière le texte : le
// corps d'une texture court jusqu'à la prochaine directive.
std::string look_text(const std::array<const CubeLut*, kLookSlots>& luts) {
  std::vector<const CubeLut*> distinct;
  std::array<std::size_t, kLookSlots> texture{};
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    if (luts[i] == nullptr) continue;
    const auto same = std::ranges::find_if(
        distinct, [&](const CubeLut* seen) { return seen == luts[i] || *seen == *luts[i]; });
    texture[i] = static_cast<std::size_t>(same - distinct.begin());
    if (same == distinct.end()) distinct.push_back(luts[i]);
  }
  std::string params;
  std::string chain = "vec3 ajl_chain(vec3 c) {\n";
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    const std::string mix = slot_name("look_mix", i);
    if (i > 0) {
      params += std::format(
          "//!PARAM {}\n//!DESC Intensité de la LUT {} (0 = aucune)\n//!TYPE float\n//!MINIMUM 0.0\n"
          "//!MAXIMUM 1.0\n0.0\n\n",
          mix, i + 1);
    }
    if (luts[i] == nullptr) continue;
    chain += std::format(
        "    if ({0} > 0.0) {{\n        vec3 pos = (clamp(c, 0.0, 1.0) * {1}.0 + 0.5) / {2}.0;\n"
        "        c = mix(c, clamp({3}_tex(pos).rgb, 0.0, 1.0), {0});\n    }}\n",
        mix, luts[i]->size - 1, luts[i]->size, slot_name("AJG_LUT", texture[i]));
  }
  chain += "    return c;\n}\n";
  std::string binds;
  for (std::size_t t = 0; t < distinct.size(); ++t)
    binds += std::format("//!BIND {}\n", slot_name("AJG_LUT", t));
  std::string text(shaders::anime_hdr_look);
  const auto replace = [&text](std::string_view mark, const std::string& with) {
    const std::size_t at = text.find(mark);
    if (at != std::string::npos) text.replace(at, mark.size(), with);
  };
  replace("//@LOOK_PARAMS@", params);
  replace("//@LOOK_BINDS@\n", binds);  // sans LUT, la ligne disparaît : une ligne vide couperait l'en-tête
  replace("//@LOOK_CHAIN@", chain);
  text += '\n';
  for (std::size_t t = 0; t < distinct.size(); ++t)
    text += lut_texture_block(*distinct[t], slot_name("AJG_LUT", t));
  return text;
}

constexpr std::array kBlackParams = {
    ParamBinding{"black_lo", &GradeParams::black_lo}, ParamBinding{"black_hi", &GradeParams::black_hi},
    ParamBinding{"sdr_out", &GradeParams::sdr_out},   ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
    ParamBinding{"peak", &GradeParams::peak},         ParamBinding{"sdr_tm", &GradeParams::sdr_tm},
    ParamBinding{"tm_white", &GradeParams::tm_white}, ParamBinding{"tm_top", &GradeParams::tm_top},
};

pl_var_data* find_param(const pl_hook& hook, std::string_view name) {
  for (int i = 0; i < hook.num_parameters; ++i) {
    const pl_hook_par& par = hook.parameters[i];
    if (par.type == PL_VAR_FLOAT && name == par.name) return par.data;
  }
  return nullptr;
}

struct ParsedHook {
  const pl_hook* hook = nullptr;
  std::vector<std::pair<pl_var_data*, ParamBinding>> params;
};

std::expected<ParsedHook, Refusal> parse(pl_gpu gpu, std::string_view text, std::string_view file,
                                         std::span<const ParamBinding> bindings,
                                         std::span<const ParamBinding> more = {}) {
  ParsedHook out;
  out.hook = pl_mpv_user_shader_parse(gpu, text.data(), text.size());
  if (out.hook == nullptr) {
    return std::unexpected(
        Refusal{RefusalCode::VendorCallFailed, std::format("pl_mpv_user_shader_parse ({})", file)});
  }
  std::vector<ParamBinding> all(bindings.begin(), bindings.end());
  all.insert(all.end(), more.begin(), more.end());
  for (const ParamBinding& binding : all) {
    pl_var_data* data = find_param(*out.hook, binding.name);
    if (data == nullptr) {
      pl_mpv_user_shader_destroy(&out.hook);
      return std::unexpected(
          Refusal{RefusalCode::FieldMissing, std::format("//!PARAM {} absent de {}", binding.name, file)});
    }
    out.params.emplace_back(data, binding);
  }
  return out;
}

// Le hook du look lu par libplacebo, qui crée les textures : sur le thread qui rend.
std::expected<ParsedHook, Refusal> parse_look(pl_gpu gpu, std::string_view text) {
  // Une marque laissée (fin de ligne inattendue) couperait l'en-tête : le hook ne compilerait qu'au rendu,
  // et libplacebo le couperait sans le dire au lecteur.
  for (const std::string_view mark : {"//@LOOK_PARAMS@", "//@LOOK_BINDS@", "//@LOOK_CHAIN@"})
    if (text.find(mark) != std::string_view::npos)
      return std::unexpected(
          Refusal{RefusalCode::FieldMissing, "anime-hdr-look.hook : marque LOOK non remplacée"});
  return parse(gpu, text, "anime-hdr-look.hook", kLookParams);
}

}  // namespace

struct Grade::Impl {
  float video_fps = 24.0F;
  std::optional<Grade::KwinMatch> kwin;  // set_kwin_match, réécrit dans un hook refait
  void put_kwin();
  // La sortie HDR (décision 16) : le sommet de la courbe, le plafond des éclats et l'alpha de la passe
  // anime, recalculés dès que les paramètres, la pile ou l'écran changent ; rien sans la dernière passe.
  std::optional<OutputTune> tune;
  void put_output();
  ParsedHook grade;
  ParsedHook look;
  ParsedHook black;
  const pl_hook* calibration = nullptr;
  std::array<const pl_hook*, 3> hooks{};
  GradeParams params;
  pl_gpu gpu = nullptr;
  // Les LUT du look (copies : celles de set_looks ne vivent que le temps de l'appel).
  std::array<std::shared_ptr<const CubeLut>, kLookSlots> sources;
  std::array<const CubeLut*, kLookSlots> looks() const {
    std::array<const CubeLut*, kLookSlots> out{};
    for (std::size_t i = 0; i < kLookSlots; ++i) out[i] = sources[i].get();
    return out;
  }
  void write_params();

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    pl_mpv_user_shader_destroy(&grade.hook);
    pl_mpv_user_shader_destroy(&look.hook);
    pl_mpv_user_shader_destroy(&black.hook);
    pl_mpv_user_shader_destroy(&calibration);
  }
};

std::expected<Grade, Refusal> Grade::create(pl_gpu gpu) {
  auto impl = std::make_unique<Impl>();
  impl->gpu = gpu;
  // Les paramètres, la mise en PQ d'une sortie SDR, puis les trois passes (carte locale, zone, passe anime).
  std::string text(shaders::anime_hdr_params);
  text += kSdrToPq;
  text += shaders::anime_hdr_anime;
  auto grade = parse(gpu, text, "anime-hdr-anime", kGradeParams);
  if (!grade) return std::unexpected(grade.error());
  impl->grade = std::move(*grade);
  auto look = parse_look(gpu, look_text({}));
  if (!look) return std::unexpected(look.error());
  impl->look = std::move(*look);
  auto black = parse(gpu, shaders::anime_hdr_black, "anime-hdr-black.hook", kBlackParams);
  if (!black) return std::unexpected(black.error());
  impl->black = std::move(*black);
  auto calibration = parse(gpu, shaders::anime_hdr_calibration, "anime-hdr-calibration.hook", {});
  if (!calibration) return std::unexpected(calibration.error());
  impl->calibration = calibration->hook;
  impl->hooks = {impl->grade.hook, impl->look.hook, impl->black.hook};
  Grade out(std::move(impl));
  out.set_params(GradeParams{});
  return out;
}

std::expected<void, Refusal> Grade::set_looks(const std::array<const CubeLut*, kLookSlots>& luts) {
  auto look = parse_look(impl_->gpu, look_text(luts));
  if (!look) return std::unexpected(look.error());
  pl_mpv_user_shader_destroy(&impl_->look.hook);
  impl_->look = std::move(*look);
  impl_->hooks[1] = impl_->look.hook;
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    if (luts[i] != nullptr)
      impl_->sources[i] = std::make_shared<const CubeLut>(*luts[i]);
    else
      impl_->sources[i].reset();
  }
  impl_->write_params();  // les paramètres repartent dans le nouveau hook
  return {};
}

std::array<const CubeLut*, kLookSlots> Grade::looks() const {
  return impl_->looks();
}

Grade::Grade(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Grade::Grade(Grade&&) noexcept = default;
Grade& Grade::operator=(Grade&&) noexcept = default;
Grade::~Grade() = default;

void Grade::set_params(const GradeParams& params) {
  impl_->params = params;
  impl_->write_params();
}

void Grade::Impl::write_params() {
  for (const ParsedHook* hook : {&grade, &look, &black}) {
    for (const auto& [data, binding] : hook->params) data->f = static_cast<float>(value_of(params, binding));
  }
  // L'intensité de chaque emplacement ; un emplacement sans LUT reste à 0, sauté.
  const pl_hook& look_hook = *look.hook;
  const std::array<const CubeLut*, kLookSlots> luts = looks();
  const bool active = LookChain{.luts = luts, .mixes = params.look_mixes}.active();
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    if (pl_var_data* mix = find_param(look_hook, slot_name("look_mix", i)))
      mix->f = sources[i] ? static_cast<float>(std::clamp(params.look_mixes[i], 0.0, 1.0)) : 0.0F;
  }
  // Le pied adouci (décision 17) accompagne la pile d'HDR : ses seuils suivent le blanc relevé de la source.
  if (pl_var_data* toe = find_param(look_hook, "look_toe")) {
    const double white = params.adapt > 0.5 ? params.sdr_white : params.paper;
    toe->f = static_cast<float>(active && params.look_pq > 0.5 ? white * kAnimeLift : 0.0);
  }
  if (pl_var_data* on = find_param(look_hook, "look_on")) on->f = active ? 1.0F : 0.0F;
  put_output();
}

void Grade::Impl::put_output() {
  const std::array<const CubeLut*, kLookSlots> luts = looks();
  tune = kwin ? std::optional(output_tune(params, luts, kwin->reference, kwin->peak)) : std::nullopt;
  const auto put = [this](std::string_view name, double value) {
    for (const pl_hook* hook : {black.hook, grade.hook})
      if (pl_var_data* data = find_param(*hook, name)) data->f = static_cast<float>(value);
  };
  put("an_late", tune ? 1.0 : 0.0);
  if (tune) {
    put("kwin_curve", std::clamp(tune->curve_peak, 50.0, 100000.0));
    put("em_top", std::clamp(tune->top, 50.0, 10000.0));
  }
}

void Grade::set_video_fps(double fps) {
  impl_->video_fps = static_cast<float>(std::clamp(fps, 1.0, 1000.0));
  if (pl_var_data* data = find_param(*impl_->grade.hook, "video_fps")) data->f = impl_->video_fps;
}

void Grade::set_kwin_match(std::optional<KwinMatch> match) {
  impl_->kwin = match;
  impl_->put_kwin();
}

void Grade::Impl::put_kwin() {
  const auto put = [this](std::string_view name, double value) {
    for (const pl_hook* hook : {black.hook, grade.hook})
      if (pl_var_data* data = find_param(*hook, name)) data->f = static_cast<float>(value);
  };
  put("kwin_on", kwin ? 1.0 : 0.0);
  if (kwin) {
    put("kwin_ref", std::clamp(kwin->reference, 50.0, 1000.0));
    put("kwin_peak", std::clamp(kwin->peak, 100.0, 10000.0));
  }
  put_output();
}

std::optional<float> Grade::param(std::string_view name) const {
  if (const pl_var_data* data = find_param(*impl_->grade.hook, name)) return data->f;
  return std::nullopt;
}

std::optional<OutputTune> Grade::output() const {
  return impl_->tune;
}

const GradeParams& Grade::params() const {
  return impl_->params;
}

std::span<const pl_hook* const> Grade::hooks() const {
  return std::span<const pl_hook* const>(impl_->hooks.data(), impl_->hooks.size());
}

const pl_hook* Grade::calibration() const {
  return impl_->calibration;
}

}  // namespace ajp::render
