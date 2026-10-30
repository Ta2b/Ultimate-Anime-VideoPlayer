#include "render/grade.h"

#include <array>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "render/shaders.h"

namespace ajp::render {
namespace {

// Un //!PARAM et le champ de GradeParams qui le règle.
struct ParamBinding {
  std::string_view name;
  double GradeParams::* field;
};

constexpr std::array kGradeParams = {
    ParamBinding{"peak", &GradeParams::peak},
    ParamBinding{"paper", &GradeParams::paper},
    ParamBinding{"in_max", &GradeParams::in_max},
    ParamBinding{"shadows", &GradeParams::shadows},
    ParamBinding{"shadow_knee", &GradeParams::shadow_knee},
    ParamBinding{"area_gain", &GradeParams::area_gain},
    ParamBinding{"spec_gain", &GradeParams::spec_gain},
    ParamBinding{"vibrance", &GradeParams::vibrance},
    ParamBinding{"abl", &GradeParams::abl},
    ParamBinding{"guard", &GradeParams::guard},
    ParamBinding{"expand", &GradeParams::expand},
    ParamBinding{"abl_lo", &GradeParams::abl_lo},
    ParamBinding{"abl_hi", &GradeParams::abl_hi},
    ParamBinding{"gamut_p3", &GradeParams::gamut_p3},
    ParamBinding{"emulate", &GradeParams::emulate},
    ParamBinding{"adapt", &GradeParams::adapt},
    ParamBinding{"sdr_white", &GradeParams::sdr_white},
    ParamBinding{"em_base", &GradeParams::em_base},
    ParamBinding{"em_pow", &GradeParams::em_pow},
    ParamBinding{"em_lo", &GradeParams::em_lo},
    ParamBinding{"em_hi", &GradeParams::em_hi},
    ParamBinding{"em_sat", &GradeParams::em_sat},
    ParamBinding{"sdr_out", &GradeParams::sdr_out},
    ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
    ParamBinding{"punch", &GradeParams::punch},
    ParamBinding{"chroma", &GradeParams::chroma},
    ParamBinding{"pivot", &GradeParams::pivot},
};

constexpr std::array kLookParams = {
    ParamBinding{"look_mix", &GradeParams::look_mix},
    ParamBinding{"look_mix2", &GradeParams::look_mix2},
    ParamBinding{"look_mix3", &GradeParams::look_mix3},
    ParamBinding{"look_mix4", &GradeParams::look_mix4},
    ParamBinding{"look_mix5", &GradeParams::look_mix5},
    ParamBinding{"look_mix6", &GradeParams::look_mix6},
    ParamBinding{"look_mix7", &GradeParams::look_mix7},
    ParamBinding{"look_mix8", &GradeParams::look_mix8},
    ParamBinding{"look_white", &GradeParams::look_white},
    ParamBinding{"look_pq", &GradeParams::look_pq},
    ParamBinding{"sdr_out", &GradeParams::sdr_out},
    ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
    ParamBinding{"peak", &GradeParams::peak},
};

// Le nom du paramètre ou de la texture du `slot`-ième emplacement (0 : le premier, sans numéro).
std::string slot_name(std::string_view base, std::size_t slot) {
  return slot == 0 ? std::string(base) : std::format("{}{}", base, slot + 1);
}

// Le texte du hook du look : les emplacements (paramètres, textures, condition, chaîne) sont écrits pour
// kLookSlots LUT à partir des marques //@LOOK_...@ du fichier.
std::string look_source() {
  std::string params;
  std::string binds;
  std::string when = "//!WHEN look_mix";
  std::string chain = "vec3 ajl_chain(vec3 c) {\n";
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    const std::string mix = slot_name("look_mix", i);
    const std::string size = slot_name("look_size", i);
    const std::string tex = slot_name("AJG_LUT", i);
    if (i > 0)
      params += std::format(
          "//!PARAM {}\n//!DESC Intensité de la LUT {} (0 = aucune)\n//!TYPE float\n//!MINIMUM 0.0\n"
          "//!MAXIMUM 1.0\n0.0\n\n",
          mix, i + 1);
    params += std::format(
        "//!PARAM {}\n//!DESC Points par axe de la LUT {}\n//!TYPE float\n//!MINIMUM 2.0\n"
        "//!MAXIMUM 65.0\n2.0\n\n",
        size, i + 1);
    binds += std::format("//!BIND {}\n", tex);
    if (i > 0) when += std::format(" {} +", mix);
    chain += std::format(
        "    if ({0} > 0.0) {{\n        vec3 pos = (clamp(c, 0.0, 1.0) * ({1} - 1.0) + 0.5) / {1};\n"
        "        c = mix(c, clamp({2}_tex(pos).rgb, 0.0, 1.0), {0});\n    }}\n",
        mix, size, tex);
  }
  when += " 0 >";
  chain += "    return c;\n}\n";
  std::string text(shaders::anime_hdr_look);
  const auto replace = [&text](std::string_view mark, const std::string& with) {
    const std::size_t at = text.find(mark);
    if (at != std::string::npos) text.replace(at, mark.size(), with);
  };
  replace("//@LOOK_PARAMS@", params);
  if (!binds.empty()) binds.pop_back();
  replace("//@LOOK_BINDS@", binds);
  replace("//@LOOK_WHEN@", when);
  replace("//@LOOK_CHAIN@", chain);
  return text;
}

constexpr std::array kBlackParams = {
    ParamBinding{"black_lo", &GradeParams::black_lo}, ParamBinding{"black_hi", &GradeParams::black_hi},
    ParamBinding{"sdr_out", &GradeParams::sdr_out},   ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
    ParamBinding{"peak", &GradeParams::peak},
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
  std::vector<std::pair<pl_var_data*, double GradeParams::*>> params;
};

std::expected<ParsedHook, Refusal> parse(pl_gpu gpu, std::string_view text, std::string_view file,
                                         std::span<const ParamBinding> bindings) {
  ParsedHook out;
  out.hook = pl_mpv_user_shader_parse(gpu, text.data(), text.size());
  if (out.hook == nullptr) {
    return std::unexpected(
        Refusal{RefusalCode::VendorCallFailed, std::format("pl_mpv_user_shader_parse ({})", file)});
  }
  for (const ParamBinding& binding : bindings) {
    pl_var_data* data = find_param(*out.hook, binding.name);
    if (data == nullptr) {
      pl_mpv_user_shader_destroy(&out.hook);
      return std::unexpected(
          Refusal{RefusalCode::FieldMissing, std::format("//!PARAM {} absent de {}", binding.name, file)});
    }
    out.params.emplace_back(data, binding.field);
  }
  return out;
}

// Le hook du look : le bloc de la texture de la LUT (ou d'une identité 2x2x2), puis le texte embarqué.
std::expected<ParsedHook, Refusal> parse_look(pl_gpu gpu,
                                              const std::array<const CubeLut*, kLookSlots>& luts) {
  static const CubeLut kNeutral = [] {
    CubeLut neutral;
    neutral.size = 2;
    neutral.rgb = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
    return neutral;
  }();
  // Le corps d'une texture court jusqu'à la prochaine directive //! : le bloc va donc après le texte du hook.
  std::string text = look_source();
  text += '\n';
  for (std::size_t i = 0; i < kLookSlots; ++i)
    text += lut_texture_block(luts[i] != nullptr ? *luts[i] : kNeutral, slot_name("AJG_LUT", i));
  auto parsed = parse(gpu, text, "anime-hdr-look.hook", kLookParams);
  if (!parsed) return parsed;
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    const std::string size = slot_name("look_size", i);
    if (find_param(*parsed->hook, size) == nullptr) {
      pl_mpv_user_shader_destroy(&parsed->hook);
      return std::unexpected(
          Refusal{RefusalCode::FieldMissing, std::format("//!PARAM {} absent de anime-hdr-look.hook", size)});
    }
  }
  return parsed;
}

}  // namespace

struct Grade::Impl {
  ParsedHook grade;
  ParsedHook look;
  ParsedHook black;
  const pl_hook* calibration = nullptr;
  std::array<const pl_hook*, 3> hooks{};
  GradeParams params;
  pl_gpu gpu = nullptr;
  std::array<int, kLookSlots> look_sizes{};

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
  auto grade = parse(gpu, shaders::anime_hdr_grade, "anime-hdr-grade.hook", kGradeParams);
  if (!grade) return std::unexpected(grade.error());
  impl->grade = std::move(*grade);
  auto look = parse_look(gpu, {});
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
  auto look = parse_look(impl_->gpu, luts);
  if (!look) return std::unexpected(look.error());
  pl_mpv_user_shader_destroy(&impl_->look.hook);
  impl_->look = std::move(*look);
  for (std::size_t i = 0; i < kLookSlots; ++i) impl_->look_sizes[i] = luts[i] != nullptr ? luts[i]->size : 2;
  impl_->hooks[1] = impl_->look.hook;
  set_params(impl_->params);
  return {};
}

Grade::Grade(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Grade::Grade(Grade&&) noexcept = default;
Grade& Grade::operator=(Grade&&) noexcept = default;
Grade::~Grade() = default;

void Grade::set_params(const GradeParams& params) {
  impl_->params = params;
  for (const ParsedHook* hook : {&impl_->grade, &impl_->look, &impl_->black}) {
    for (const auto& [data, field] : hook->params) data->f = static_cast<float>(params.*field);
  }
  for (std::size_t i = 0; i < kLookSlots; ++i)
    if (pl_var_data* size = find_param(*impl_->look.hook, slot_name("look_size", i)))
      size->f = static_cast<float>(impl_->look_sizes[i] > 0 ? impl_->look_sizes[i] : 2);
}

const GradeParams& Grade::params() const {
  return impl_->params;
}

std::span<const pl_hook* const> Grade::hooks() const {
  return impl_->hooks;
}

const pl_hook* Grade::calibration() const {
  return impl_->calibration;
}

}  // namespace ajp::render
