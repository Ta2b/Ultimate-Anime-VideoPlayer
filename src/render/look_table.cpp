#include "render/look_table.h"

#include <cmath>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include <libplacebo/colorspace.h>
#include <libplacebo/dispatch.h>

namespace ajp::render {
namespace {

// Le nom d'un //!PARAM ou d'une texture du `slot`-ième emplacement (0 : le premier, sans numéro), comme
// grade.cpp les écrit dans le hook.
std::string slot_name(std::string_view base, std::size_t slot) {
  return slot == 0 ? std::string(base) : std::format("{}{}", base, slot + 1);
}

// Le texte GLSL du hook du look coupé en trois : les fonctions communes (matrices, PQ), la pile et
// ajl_look(), le passage du look (voisinage, tuiles, hook()).
struct LookText {
  std::string common;
  std::string look;
  std::string pass;
};

std::expected<LookText, Refusal> split(std::string_view text) {
  constexpr std::string_view kDesc = "//!DESC AnimeJaNai HDR - look (LUT 3D)\n";
  const std::size_t desc = text.find(kDesc);
  if (desc == std::string_view::npos)
    return std::unexpected(
        Refusal{RefusalCode::FieldMissing, "anime-hdr-look.hook : en-tête du look absent"});
  std::string_view glsl = text.substr(desc + kDesc.size());
  glsl = glsl.substr(0, glsl.find("\n//!TEXTURE"));  // les blocs des LUT suivent le texte
  const std::size_t chain = glsl.find("vec3 ajl_chain(vec3 c)");
  const std::size_t end = glsl.find("//@LOOK_END@");
  if (chain == std::string_view::npos || end == std::string_view::npos || end < chain)
    return std::unexpected(
        Refusal{RefusalCode::FieldMissing, "anime-hdr-look.hook : pile ou marque LOOK_END absente"});
  return LookText{.common = std::string(glsl.substr(0, chain)),
                  .look = std::string(glsl.substr(chain, end - chain)),
                  .pass = std::string(glsl.substr(end))};
}

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

pl_shader_var float_var(const char* name, const float* value) {
  return pl_shader_var{.var = pl_var_float(name), .data = value, .dynamic = false};
}

}  // namespace

struct LookTable::Impl {
  pl_gpu gpu = nullptr;
  pl_fmt fmt = nullptr;
  pl_tex table = nullptr;
  std::vector<pl_tex> luts;
  std::vector<std::string> lut_names;
  std::vector<std::string> mix_names;  // look_mix… des emplacements, pour les variables de la construction
  std::string build_header;
  std::string pass_header;
  Inputs in;
  bool built = false;
  std::uint64_t builds = 0;
  pl_hook hook{};

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    for (pl_tex& lut : luts) pl_tex_destroy(gpu, &lut);
    pl_tex_destroy(gpu, &table);
  }

  bool build(pl_dispatch dispatch);
  pl_hook_res run(const pl_hook_params& params);
};

// La table : chaque nœud (r, g, b) / 128 passe par ajl_look(), le calcul exact du hook, pile, pied adouci et
// domaines compris.
bool LookTable::Impl::build(pl_dispatch dispatch) {
  pl_shader sh = pl_dispatch_begin(dispatch);
  std::vector<pl_shader_desc> descs;
  for (std::size_t t = 0; t < luts.size(); ++t) descs.push_back(sampled(lut_names[t].c_str(), luts[t]));
  descs.push_back(storage("AJL_TABLE", table));
  std::vector<pl_shader_var> vars;
  for (std::size_t i = 0; i < kLookSlots; ++i) vars.push_back(float_var(mix_names[i].c_str(), &in.mixes[i]));
  vars.push_back(float_var("look_white", &in.white));
  vars.push_back(float_var("look_pq", &in.pq));
  vars.push_back(float_var("look_toe", &in.toe));
  vars.push_back(float_var("look_gamma", &in.gamma));
  pl_custom_shader custom{};
  custom.header = build_header.c_str();
  custom.body = "ajl_build();";
  custom.description = "AnimeJaNai HDR - look : construction de la table";
  custom.input = PL_SHADER_SIG_NONE;
  custom.output = PL_SHADER_SIG_NONE;
  custom.descriptors = descs.data();
  custom.num_descriptors = static_cast<int>(descs.size());
  custom.variables = vars.data();
  custom.num_variables = static_cast<int>(vars.size());
  custom.compute = true;
  custom.compute_group_size[0] = 8;
  custom.compute_group_size[1] = 8;
  if (!pl_shader_custom(sh, &custom)) {
    pl_dispatch_abort(dispatch, &sh);
    return false;
  }
  constexpr int kGroups = (kLookTableSize + 7) / 8;
  pl_dispatch_compute_params run{};
  run.shader = &sh;
  run.dispatch_size[0] = kGroups;
  run.dispatch_size[1] = kGroups;
  run.dispatch_size[2] = kLookTableSize;
  run.width = kLookTableSize;
  run.height = kLookTableSize;
  const bool ok = pl_dispatch_compute(dispatch, &run);
  if (ok) ++builds;
  return ok;
}

// Le passage du look de anime-hdr-look.hook, ses mêmes lectures (HOOKED_tex, HOOKED_pt), sa même sortie
// (une texture de la taille du rectangle, comme custom_mpv.c), ajl_look() lisant la table.
pl_hook_res LookTable::Impl::run(const pl_hook_params& params) {
  pl_hook_res res{};
  if (!in.on) return res;  // PL_HOOK_SIG_NONE : passe sautée, comme //!WHEN look_on
  if (!built) {
    if (!build(params.dispatch)) {
      res.failed = true;
      return res;
    }
    built = true;
  }
  const int w = static_cast<int>(std::lround(pl_rect_w(params.rect)));
  const int h = static_cast<int>(std::lround(pl_rect_h(params.rect)));
  pl_tex out = params.get_tex(params.priv, w, h);
  if (out == nullptr || !out->params.storable) {
    res.failed = true;
    return res;
  }
  pl_color_repr repr = params.repr;
  const float scale = pl_color_repr_normalize(&repr);
  const std::array<float, 2> pt{1.0F / static_cast<float>(params.tex->params.w),
                                1.0F / static_cast<float>(params.tex->params.h)};
  const std::array descs{sampled("HOOKED", params.tex), sampled("AJL_TABLE", table),
                         storage("out_image", out)};
  const std::array vars{
      float_var("HOOKED_mul", &scale),
      pl_shader_var{.var = pl_var_vec2("HOOKED_pt"), .data = pt.data(), .dynamic = false},
      float_var("look_clean", &in.clean),
  };
  pl_custom_shader custom{};
  custom.header = pass_header.c_str();
  custom.body = "hook();";
  custom.description = "AnimeJaNai HDR - look (table 3D)";
  custom.input = PL_SHADER_SIG_NONE;
  custom.output = PL_SHADER_SIG_NONE;
  custom.descriptors = descs.data();
  custom.num_descriptors = static_cast<int>(descs.size());
  custom.variables = vars.data();
  custom.num_variables = static_cast<int>(vars.size());
  custom.compute = true;
  custom.compute_group_size[0] = 16;
  custom.compute_group_size[1] = 16;
  custom.compute_shmem = sizeof(float) * 28 * 28;
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

std::expected<std::unique_ptr<LookTable>, Refusal> LookTable::create(pl_gpu gpu) {
  auto impl = std::make_unique<Impl>();
  impl->gpu = gpu;
  impl->fmt = pl_find_named_fmt(gpu, "rgba32f");
  constexpr auto kCaps = PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR | PL_FMT_CAP_STORABLE;
  if (impl->fmt == nullptr || (impl->fmt->caps & kCaps) != kCaps)
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed,
                                   "look en une table : pas de rgba32f filtrable et inscriptible"});
  pl_tex_params tp{};
  tp.w = kLookTableSize;
  tp.h = kLookTableSize;
  tp.d = kLookTableSize;
  tp.format = impl->fmt;
  tp.sampleable = true;
  tp.storable = true;
  tp.debug_tag = PL_DEBUG_TAG;
  impl->table = pl_tex_create(gpu, &tp);
  if (impl->table == nullptr)
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "look en une table : pl_tex_create"});
  for (std::size_t i = 0; i < kLookSlots; ++i) impl->mix_names.push_back(slot_name("look_mix", i));
  impl->hook.stages = PL_HOOK_OUTPUT;
  impl->hook.input = PL_HOOK_SIG_TEX;
  impl->hook.priv = impl.get();
  impl->hook.hook = [](void* priv, const pl_hook_params* params) {
    return static_cast<Impl*>(priv)->run(*params);
  };
  impl->hook.signature = 0x414a4c5441424c45ULL;  // « AJLTABLE »
  return std::unique_ptr<LookTable>(new LookTable(std::move(impl)));
}

std::expected<void, Refusal> LookTable::set_looks(std::string_view look_hook_text,
                                                  std::span<const CubeLut* const> textures) {
  auto text = split(look_hook_text);
  if (!text) return std::unexpected(text.error());
  std::vector<pl_tex> luts;
  std::vector<std::string> names;
  const auto drop = [this](std::vector<pl_tex>& made) {
    for (pl_tex& tex : made) pl_tex_destroy(impl_->gpu, &tex);
  };
  std::string macros;
  for (std::size_t t = 0; t < textures.size(); ++t) {
    const CubeLut& lut = *textures[t];
    std::vector<float> rgba(lut.rgb.size() / 3 * 4);
    for (std::size_t i = 0; i < lut.rgb.size() / 3; ++i) {
      for (std::size_t c = 0; c < 3; ++c) rgba[4 * i + c] = lut.rgb[3 * i + c];
      rgba[4 * i + 3] = 1.0F;
    }
    pl_tex_params tp{};
    tp.w = lut.size;
    tp.h = lut.size;
    tp.d = lut.size;
    tp.format = impl_->fmt;
    tp.sampleable = true;
    tp.initial_data = rgba.data();
    tp.debug_tag = PL_DEBUG_TAG;
    pl_tex tex = pl_tex_create(impl_->gpu, &tp);
    if (tex == nullptr) {
      drop(luts);
      return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "look en une table : texture d'une LUT"});
    }
    luts.push_back(tex);
    names.push_back(slot_name("AJG_LUT", t));
    macros += std::format("#define {0}_tex(pos) textureLod({0}, pos, 0.0)\n", names.back());
  }
  constexpr int kLast = kLookTableSize - 1;
  impl_->build_header = macros + text->common + text->look +
                        std::format(
                            "void ajl_build() {{\n"
                            "    ivec3 p = ivec3(gl_GlobalInvocationID);\n"
                            "    if (any(greaterThan(p, ivec3({0})))) return;\n"
                            "    imageStore(AJL_TABLE, p, vec4(ajl_look(vec3(p) / {0}.0), 1.0));\n"
                            "}}\n",
                            kLast);
  impl_->pass_header = std::format(
      "#define HOOKED_tex(pos) (HOOKED_mul * vec4(textureLod(HOOKED, pos, 0.0)))\n{}"
      "vec3 ajl_look(vec3 e) {{\n"
      "    return textureLod(AJL_TABLE, (clamp(e, 0.0, 1.0) * {}.0 + 0.5) / {}.0, 0.0).rgb;\n"
      "}}\n{}",
      text->common, kLast, kLookTableSize, text->pass);
  drop(impl_->luts);
  impl_->luts = std::move(luts);
  impl_->lut_names = std::move(names);
  impl_->built = false;
  return {};
}

void LookTable::set_inputs(const Inputs& inputs) {
  if (inputs == impl_->in) return;
  // L'allumage et le voisinage ne changent pas la table.
  Inputs same = inputs;
  same.on = impl_->in.on;
  same.clean = impl_->in.clean;
  if (!(same == impl_->in)) impl_->built = false;
  impl_->in = inputs;
}

const pl_hook* LookTable::hook() const {
  return &impl_->hook;
}

std::uint64_t LookTable::builds() const {
  return impl_->builds;
}

LookTable::LookTable(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LookTable::~LookTable() = default;

}  // namespace ajp::render
