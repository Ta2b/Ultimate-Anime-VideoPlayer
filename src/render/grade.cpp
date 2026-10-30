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
    ParamBinding{"look_mix", &GradeParams::look_mix},   ParamBinding{"look_mix2", &GradeParams::look_mix2},
    ParamBinding{"look_mix3", &GradeParams::look_mix3}, ParamBinding{"look_white", &GradeParams::look_white},
    ParamBinding{"look_pq", &GradeParams::look_pq},     ParamBinding{"sdr_out", &GradeParams::sdr_out},
    ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma}, ParamBinding{"peak", &GradeParams::peak},
};

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
  std::string text(shaders::anime_hdr_look);
  text += '\n';
  constexpr std::array<std::string_view, kLookSlots> kNames{"AJG_LUT", "AJG_LUT2", "AJG_LUT3"};
  for (std::size_t i = 0; i < kLookSlots; ++i)
    text += lut_texture_block(luts[i] != nullptr ? *luts[i] : kNeutral, kNames[i]);
  auto parsed = parse(gpu, text, "anime-hdr-look.hook", kLookParams);
  if (!parsed) return parsed;
  for (const std::string_view size : {"look_size", "look_size2", "look_size3"}) {
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
  std::array<int, kLookSlots> look_sizes{2, 2, 2};

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
  constexpr std::array<std::string_view, kLookSlots> kSizes{"look_size", "look_size2", "look_size3"};
  for (std::size_t i = 0; i < kLookSlots; ++i)
    if (pl_var_data* size = find_param(*impl_->look.hook, kSizes[i]))
      size->f = static_cast<float>(impl_->look_sizes[i]);
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
