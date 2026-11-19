#include "render/effect_pass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <libplacebo/colorspace.h>
#include <libplacebo/dispatch.h>

namespace kirari::render {
namespace {

// L'image passe telle quelle, le masque (bilinéaire, aux bords de la grille tenu à sa dernière tuile) dans
// l'alpha. fx_scale ramène la position dans l'image à la grille, que ses dernières tuiles débordent.
constexpr const char* kHeader =
    "void fx_main() {\n"
    "    ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
    "    if (p.x >= int(fx_out.x) || p.y >= int(fx_out.y)) return;\n"
    "    vec4 c = HOOKED_mul * vec4(textureLod(HOOKED, (vec2(p) + 0.5) * HOOKED_pt, 0.0));\n"
    "    vec2 at = (vec2(p) + 0.5) / fx_out * fx_scale;\n"
    "    c.a = clamp(textureLod(AJG_FX, at, 0.0).x, 0.0, 1.0);\n"
    "    imageStore(out_image, p, c);\n"
    "}\n";

pl_shader_desc sampled(const char* name, pl_tex tex) {
  pl_shader_desc d{};
  d.desc.name = name;
  d.desc.type = PL_DESC_SAMPLED_TEX;
  d.binding.object = tex;
  d.binding.address_mode = PL_TEX_ADDRESS_CLAMP;
  d.binding.sample_mode = PL_TEX_SAMPLE_LINEAR;
  return d;
}

pl_shader_desc storage(const char* name, pl_tex tex) {
  pl_shader_desc d{};
  d.desc.name = name;
  d.desc.type = PL_DESC_STORAGE_IMG;
  d.desc.access = PL_DESC_ACCESS_WRITEONLY;
  d.binding.object = tex;
  return d;
}

pl_shader_var vec2_var(const char* name, const float* value) {
  return pl_shader_var{.var = pl_var_vec2(name), .data = value, .dynamic = true};
}

}  // namespace

struct EffectPass::Impl {
  pl_gpu gpu = nullptr;
  pl_fmt fmt = nullptr;
  pl_tex mask = nullptr;
  std::vector<std::uint8_t> bytes;  // la grille à envoyer
  int cols = 0;
  int rows = 0;
  std::array<float, 2> scale{1.0F, 1.0F};
  bool dirty = false;
  pl_hook hook{};

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() { pl_tex_destroy(gpu, &mask); }

  bool upload();
  pl_hook_res run(const pl_hook_params& params);
};

bool EffectPass::Impl::upload() {
  if (mask == nullptr || mask->params.w != cols || mask->params.h != rows) {
    pl_tex_destroy(gpu, &mask);
    pl_tex_params tp{};
    tp.w = cols;
    tp.h = rows;
    tp.format = fmt;
    tp.sampleable = true;
    tp.host_writable = true;
    tp.debug_tag = PL_DEBUG_TAG;
    mask = pl_tex_create(gpu, &tp);
    if (mask == nullptr) return false;
  }
  pl_tex_transfer_params tp{};
  tp.tex = mask;
  tp.row_pitch = static_cast<std::size_t>(cols);
  tp.ptr = bytes.data();
  if (!pl_tex_upload(gpu, &tp)) return false;
  dirty = false;
  return true;
}

pl_hook_res EffectPass::Impl::run(const pl_hook_params& params) {
  pl_hook_res res{};
  if (cols <= 0 || rows <= 0) return res;  // PL_HOOK_SIG_NONE : rien à poser
  if (dirty && !upload()) {
    res.failed = true;
    return res;
  }
  const int w = static_cast<int>(std::lround(pl_rect_w(params.rect)));
  const int h = static_cast<int>(std::lround(pl_rect_h(params.rect)));
  pl_tex out = params.get_tex(params.priv, w, h);
  if (out == nullptr || !out->params.storable) {
    res.failed = true;
    return res;
  }
  pl_color_repr repr = params.repr;
  const float mul = pl_color_repr_normalize(&repr);
  const std::array<float, 2> pt{1.0F / static_cast<float>(params.tex->params.w),
                                1.0F / static_cast<float>(params.tex->params.h)};
  const std::array<float, 2> size{static_cast<float>(w), static_cast<float>(h)};
  const std::array descs{sampled("HOOKED", params.tex), sampled("AJG_FX", mask), storage("out_image", out)};
  const std::array vars{
      pl_shader_var{.var = pl_var_float("HOOKED_mul"), .data = &mul, .dynamic = false},
      pl_shader_var{.var = pl_var_vec2("HOOKED_pt"), .data = pt.data(), .dynamic = false},
      vec2_var("fx_out", size.data()),
      vec2_var("fx_scale", scale.data()),
  };
  pl_custom_shader custom{};
  custom.header = kHeader;
  custom.body = "fx_main();";
  custom.description = "AnimeJaNai HDR - masque des effets";
  custom.input = PL_SHADER_SIG_NONE;
  custom.output = PL_SHADER_SIG_NONE;
  custom.descriptors = descs.data();
  custom.num_descriptors = static_cast<int>(descs.size());
  custom.variables = vars.data();
  custom.num_variables = static_cast<int>(vars.size());
  custom.compute = true;
  custom.compute_group_size[0] = 16;
  custom.compute_group_size[1] = 16;
  pl_shader sh = pl_dispatch_begin(params.dispatch);
  if (!pl_shader_custom(sh, &custom)) {
    pl_dispatch_abort(params.dispatch, &sh);
    res.failed = true;
    return res;
  }
  pl_dispatch_compute_params run{};
  run.shader = &sh;
  run.dispatch_size[0] = (w + 15) / 16;
  run.dispatch_size[1] = (h + 15) / 16;
  run.dispatch_size[2] = 1;
  run.width = w;
  run.height = h;
  if (!pl_dispatch_compute(params.dispatch, &run)) {
    res.failed = true;
    return res;
  }
  const float sx = static_cast<float>(w) / static_cast<float>(params.tex->params.w);
  const float sy = static_cast<float>(h) / static_cast<float>(params.tex->params.h);
  res.output = PL_HOOK_SIG_TEX;
  res.tex = out;
  res.repr = params.repr;
  res.color = params.color;
  res.components = 4;
  res.rect = {sx * params.rect.x0, sy * params.rect.y0, sx * params.rect.x1, sy * params.rect.y1};
  return res;
}

std::expected<std::unique_ptr<EffectPass>, Refusal> EffectPass::create(pl_gpu gpu) {
  auto impl = std::make_unique<Impl>();
  impl->gpu = gpu;
  impl->fmt = pl_find_named_fmt(gpu, "r8");
  constexpr auto kCaps = PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR;
  if (impl->fmt == nullptr || (impl->fmt->caps & kCaps) != kCaps)
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "masque des effets : pas de r8 filtrable"});
  impl->hook.stages = PL_HOOK_OUTPUT;
  impl->hook.input = PL_HOOK_SIG_TEX;
  impl->hook.priv = impl.get();
  impl->hook.hook = [](void* priv, const pl_hook_params* params) {
    return static_cast<Impl*>(priv)->run(*params);
  };
  impl->hook.signature = 0x414a47465845464bULL;  // « AJGFXEFK »
  return std::unique_ptr<EffectPass>(new EffectPass(std::move(impl)));
}

void EffectPass::set(int cols, int rows, std::span<const float> values, double cover_x, double cover_y) {
  Impl& s = *impl_;
  const auto count =
      static_cast<std::size_t>(std::max(cols, 0)) * static_cast<std::size_t>(std::max(rows, 0));
  if (count == 0 || values.size() != count) {
    s.cols = 0;
    s.rows = 0;
    return;
  }
  s.bytes.resize(count);
  for (std::size_t i = 0; i < count; ++i)
    s.bytes[i] = static_cast<std::uint8_t>(std::lround(std::clamp(values[i], 0.0F, 1.0F) * 255.0F));
  s.cols = cols;
  s.rows = rows;
  s.scale = {static_cast<float>(1.0 / std::max(cover_x, 1.0)),
             static_cast<float>(1.0 / std::max(cover_y, 1.0))};
  s.dirty = true;
}

const pl_hook* EffectPass::hook() const {
  return &impl_->hook;
}

EffectPass::EffectPass(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
EffectPass::~EffectPass() = default;

}  // namespace kirari::render
