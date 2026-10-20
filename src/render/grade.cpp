#include "render/grade.h"

#include <array>
#include <format>
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
    ParamBinding{"sdr_white", &GradeParams::sdr_white},
    ParamBinding{"em_base", &GradeParams::em_base},
    ParamBinding{"em_pow", &GradeParams::em_pow},
    ParamBinding{"em_lo", &GradeParams::em_lo},
    ParamBinding{"em_hi", &GradeParams::em_hi},
    ParamBinding{"em_sat", &GradeParams::em_sat},
    ParamBinding{"sdr_out", &GradeParams::sdr_out},
    ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
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

}  // namespace

struct Grade::Impl {
  ParsedHook grade;
  ParsedHook black;
  const pl_hook* calibration = nullptr;
  std::array<const pl_hook*, 2> hooks{};
  GradeParams params;

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    pl_mpv_user_shader_destroy(&grade.hook);
    pl_mpv_user_shader_destroy(&black.hook);
    pl_mpv_user_shader_destroy(&calibration);
  }
};

std::expected<Grade, Refusal> Grade::create(pl_gpu gpu) {
  auto impl = std::make_unique<Impl>();
  auto grade = parse(gpu, shaders::anime_hdr_grade, "anime-hdr-grade.hook", kGradeParams);
  if (!grade) return std::unexpected(grade.error());
  impl->grade = std::move(*grade);
  auto black = parse(gpu, shaders::anime_hdr_black, "anime-hdr-black.hook", kBlackParams);
  if (!black) return std::unexpected(black.error());
  impl->black = std::move(*black);
  auto calibration = parse(gpu, shaders::anime_hdr_calibration, "anime-hdr-calibration.hook", {});
  if (!calibration) return std::unexpected(calibration.error());
  impl->calibration = calibration->hook;
  impl->hooks = {impl->grade.hook, impl->black.hook};
  Grade out(std::move(impl));
  out.set_params(GradeParams{});
  return out;
}

Grade::Grade(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Grade::Grade(Grade&&) noexcept = default;
Grade& Grade::operator=(Grade&&) noexcept = default;
Grade::~Grade() = default;

void Grade::set_params(const GradeParams& params) {
  impl_->params = params;
  for (const ParsedHook* hook : {&impl_->grade, &impl_->black}) {
    for (const auto& [data, field] : hook->params) data->f = static_cast<float>(params.*field);
  }
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
