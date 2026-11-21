#include "render/effect_pass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <libplacebo/gpu.h>
#include <libplacebo/shaders.h>

namespace kirari::render {
namespace {

// Le masque (bilinéaire, aux bords de la grille tenu à sa dernière tuile) dans l'alpha de la couleur en
// attente. fx_pos va de 0 à 1 sur l'image ; fx_scale la ramène à la grille, que ses dernières tuiles
// débordent.
constexpr const char* kBody = "color.a = clamp(textureLod(AJG_FX, fx_pos * fx_scale, 0.0).x, 0.0, 1.0);";
constexpr const char* kNeutralBody = "color.a = 127.0 / 255.0;";
constexpr const char* kGuideBody =
    "color.a = (127.0 + 128.0 * textureLod(AJG_GLOW, fx_pos * glow_scale, 0.0).x) / 255.0;";
constexpr const char* kBothBody = R"glsl(
  color.a = clamp(textureLod(AJG_FX, fx_pos * fx_scale, 0.0).x, 0.0, 1.0);
  if (color.a * 255.0 > 126.5) {
    float ajg_glow = textureLod(AJG_GLOW, fx_pos * glow_scale, 0.0).x;
    if (ajg_glow > 0.0) color.a = max(color.a, (127.0 + 128.0 * ajg_glow) / 255.0);
  }
)glsl";

pl_shader_desc sampled(const char* name, pl_tex tex, pl_tex_sample_mode mode) {
  pl_shader_desc d{};
  d.desc.name = name;
  d.desc.type = PL_DESC_SAMPLED_TEX;
  d.binding.object = tex;
  d.binding.address_mode = PL_TEX_ADDRESS_CLAMP;
  d.binding.sample_mode = mode;
  return d;
}

// Les coins de l'image, ligne par ligne depuis le haut à gauche (pl_shader_va).
constexpr std::array<std::array<float, 2>, 4> kCorners{
    {{0.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 1.0F}}};

}  // namespace

struct EffectPass::Impl {
  pl_gpu gpu = nullptr;
  pl_fmt fmt = nullptr;
  int components = 1;       // le masque va dans la première ; les autres reçoivent la même valeur
  bool floating = false;    // float 32 bits par composante, sinon unorm 8 bits
  pl_fmt vertex = nullptr;  // vec2 des coins (fx_pos)
  bool linear = true;  // faute de format filtrable, lu au plus proche (le flou 3 × 3 adoucit les marches)
  struct Grid {
    pl_tex texture = nullptr;
    std::vector<std::uint8_t> bytes;
    int cols = 0;
    int rows = 0;
    std::array<float, 2> scale{1.0F, 1.0F};
    bool dirty = false;
    bool active() const { return cols > 0 && rows > 0; }
  };
  Grid mask;
  Grid guide;
  pl_hook hook{};

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    pl_tex_destroy(gpu, &mask.texture);
    pl_tex_destroy(gpu, &guide.texture);
  }

  bool upload(Grid& grid);
  void set(Grid& grid, int cols, int rows, std::span<const float> values, double cover_x, double cover_y);
  pl_hook_res run(const pl_hook_params& params);
};

bool EffectPass::Impl::upload(Grid& grid) {
  if (grid.texture == nullptr || grid.texture->params.w != grid.cols || grid.texture->params.h != grid.rows) {
    pl_tex_destroy(gpu, &grid.texture);
    pl_tex_params tp{};
    tp.w = grid.cols;
    tp.h = grid.rows;
    tp.format = fmt;
    tp.sampleable = true;
    tp.host_writable = true;
    tp.debug_tag = PL_DEBUG_TAG;
    grid.texture = pl_tex_create(gpu, &tp);
    if (grid.texture == nullptr) return false;
  }
  pl_tex_transfer_params tp{};
  tp.tex = grid.texture;
  tp.row_pitch = static_cast<std::size_t>(grid.cols) * fmt->texel_size;
  tp.ptr = grid.bytes.data();
  if (!pl_tex_upload(gpu, &tp)) return false;
  grid.dirty = false;
  return true;
}

// Ajouté à la couleur en attente de la sortie (PL_HOOK_SIG_COLOR) : libplacebo le fond dans la passe qui
// rend l'image pour la passe anime, déjà là. Une passe à part recopiait l'image entière (0,41 ms en 4K sur
// une RTX 4070 Ti, passage 9 de la séance du 19/11) pour une seule composante.
pl_hook_res EffectPass::Impl::run(const pl_hook_params& params) {
  pl_hook_res res{};
  if (params.sh == nullptr) return res;
  if (mask.active() && mask.dirty && !upload(mask)) {
    res.failed = true;
    return res;
  }
  // Un refus du guide laisse le masque d'origine disponible. L'alpha neutre vaut 127 / 255.
  if (guide.active() && guide.dirty && !upload(guide)) guide.cols = guide.rows = 0;
  const auto sampling = linear ? PL_TEX_SAMPLE_LINEAR : PL_TEX_SAMPLE_NEAREST;
  std::array<pl_shader_desc, 2> descs{};
  std::array<pl_shader_var, 2> vars{};
  std::size_t bound = 0;
  if (mask.active()) {
    descs[bound] = sampled("AJG_FX", mask.texture, sampling);
    vars[bound++] = {.var = pl_var_vec2("fx_scale"), .data = mask.scale.data(), .dynamic = true};
  }
  if (guide.active()) {
    descs[bound] = sampled("AJG_GLOW", guide.texture, sampling);
    vars[bound++] = {.var = pl_var_vec2("glow_scale"), .data = guide.scale.data(), .dynamic = true};
  }
  pl_shader_va corners{};
  corners.attr.name = "fx_pos";
  corners.attr.fmt = vertex;
  for (std::size_t i = 0; i < kCorners.size(); ++i) corners.data[i] = kCorners[i].data();
  pl_custom_shader custom{};
  custom.body =
      mask.active() ? (guide.active() ? kBothBody : kBody) : (guide.active() ? kGuideBody : kNeutralBody);
  // Sans « AnimeJaNai » : la passe où il se fond n'est pas une passe du grade (placebo_screen.cpp,
  // note_pass).
  custom.description = "masque des effets";
  custom.input = PL_SHADER_SIG_COLOR;
  custom.output = PL_SHADER_SIG_COLOR;
  custom.descriptors = descs.data();
  custom.num_descriptors = static_cast<int>(bound);
  custom.variables = vars.data();
  custom.num_variables = static_cast<int>(bound);
  custom.vertex_attribs = &corners;
  custom.num_vertex_attribs = 1;
  // Une image lue sans passe de mise à l'échelle (aucune, ou fondue dans la sortie) arrive sans taille, et
  // libplacebo refuse alors le résultat (« User hook tried resizing non-resizable stage! ») puis écarte le
  // hook pour toute la vie du renderer : le masque ne tenait que la première image, l'alpha restait à 1, lu
  // « effet sûr partout ». À cette étape, l'image a la taille du rectangle de destination.
  int sized_w = 0;
  int sized_h = 0;
  if (!pl_shader_output_size(params.sh, &sized_w, &sized_h)) {
    custom.output_w = std::abs(pl_rect_w(params.dst_rect));
    custom.output_h = std::abs(pl_rect_h(params.dst_rect));
  }
  if (!pl_shader_custom(params.sh, &custom)) {
    res.failed = true;
    return res;
  }
  res.output = PL_HOOK_SIG_COLOR;
  res.sh = params.sh;
  res.repr = params.repr;
  res.color = params.color;
  res.components = 4;
  res.rect = params.rect;
  return res;
}

std::expected<std::unique_ptr<EffectPass>, Refusal> EffectPass::create(pl_gpu gpu) {
  auto impl = std::make_unique<Impl>();
  impl->gpu = gpu;
  // Le premier format que l'hôte sait écrire : r8, puis rgba8, puis r32f, puis rgba32f. Un pilote
  // peut ne pas offrir r8 filtrable (relevé sur une RTX 4070 Ti, test grade_gpu_effects_test) ; chercher
  // par nom ne voyait que lui.
  constexpr auto kLinear = static_cast<pl_fmt_caps>(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR);
  struct Candidate {
    pl_fmt_type type;
    int components;
    int host_bits;
  };
  constexpr std::array kCandidates{Candidate{PL_FMT_UNORM, 1, 8}, Candidate{PL_FMT_UNORM, 4, 8},
                                   Candidate{PL_FMT_FLOAT, 1, 32}, Candidate{PL_FMT_FLOAT, 4, 32}};
  // D'abord un format filtrable, sinon un format seulement échantillonnable, lu au plus proche.
  for (const pl_fmt_caps caps : {kLinear, PL_FMT_CAP_SAMPLEABLE}) {
    for (const Candidate& c : kCandidates) {
      const pl_fmt fmt = pl_find_fmt(gpu, c.type, c.components, 0, c.host_bits, caps);
      if (fmt == nullptr || fmt->emulated) continue;
      impl->fmt = fmt;
      impl->components = c.components;
      impl->floating = c.type == PL_FMT_FLOAT;
      impl->linear = caps == kLinear;
      break;
    }
    if (impl->fmt != nullptr) break;
  }
  if (impl->fmt == nullptr)
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed,
                                   "masque des effets : aucun format échantillonnable inscriptible"});
  impl->vertex = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2);
  if (impl->vertex == nullptr)
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "masque des effets : pas de sommet vec2"});
  impl->hook.stages = PL_HOOK_OUTPUT;
  impl->hook.input = PL_HOOK_SIG_COLOR;
  impl->hook.priv = impl.get();
  impl->hook.hook = [](void* priv, const pl_hook_params* params) {
    return static_cast<Impl*>(priv)->run(*params);
  };
  impl->hook.signature = 0x414a47465845464bULL;  // « AJGFXEFK »
  return std::unique_ptr<EffectPass>(new EffectPass(std::move(impl)));
}

void EffectPass::Impl::set(Grid& grid, int cols, int rows, std::span<const float> values, double cover_x,
                           double cover_y) {
  const auto count =
      static_cast<std::size_t>(std::max(cols, 0)) * static_cast<std::size_t>(std::max(rows, 0));
  if (count == 0 || values.size() != count) {
    grid.cols = 0;
    grid.rows = 0;
    return;
  }
  const auto comps = static_cast<std::size_t>(components);
  grid.bytes.assign(count * comps * (floating ? sizeof(float) : 1U), 0);
  for (std::size_t i = 0; i < count; ++i) {
    const float v = std::clamp(values[i], 0.0F, 1.0F);
    for (std::size_t c = 0; c < comps; ++c) {
      if (floating) {
        std::memcpy(&grid.bytes[(i * comps + c) * sizeof(float)], &v, sizeof(float));
      } else {
        grid.bytes[i * comps + c] = static_cast<std::uint8_t>(std::lround(v * 255.0F));
      }
    }
  }
  grid.cols = cols;
  grid.rows = rows;
  grid.scale = {static_cast<float>(1.0 / std::max(cover_x, 1.0)),
                static_cast<float>(1.0 / std::max(cover_y, 1.0))};
  grid.dirty = true;
}

void EffectPass::set(int cols, int rows, std::span<const float> values, double cover_x, double cover_y) {
  impl_->set(impl_->mask, cols, rows, values, cover_x, cover_y);
}

void EffectPass::set_guide(int cols, int rows, std::span<const float> values, double cover_x,
                           double cover_y) {
  impl_->set(impl_->guide, cols, rows, values, cover_x, cover_y);
}

const pl_hook* EffectPass::hook() const {
  return &impl_->hook;
}

EffectPass::EffectPass(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
EffectPass::~EffectPass() = default;

}  // namespace kirari::render
