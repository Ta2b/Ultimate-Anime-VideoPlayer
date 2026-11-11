#include "app/play/placebo_imgui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <imgui.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/shaders/sampling.h>

namespace ajp::app::play {
namespace {

constexpr const char* kBackendName = "ajp_play_libplacebo";

Refusal render_failed(const char* call) {
  return Refusal{RefusalCode::VendorCallFailed, call};
}

bool has_caps(pl_fmt_caps have, pl_fmt_caps needed) {
  return (static_cast<unsigned>(have) & static_cast<unsigned>(needed)) == static_cast<unsigned>(needed);
}

// Rectangle de découpe d'ImGui, en pixels de la cible ; vide s'il sort de la cible.
pl_rect2d clip_rect(const ImDrawCmd& cmd, pl_tex target) {
  const auto to_pixel = [](float v, int limit) {
    return std::clamp(static_cast<int>(std::lround(v)), 0, limit);
  };
  return pl_rect2d{.x0 = to_pixel(cmd.ClipRect.x, target->params.w),
                   .y0 = to_pixel(cmd.ClipRect.y, target->params.h),
                   .x1 = to_pixel(cmd.ClipRect.z, target->params.w),
                   .y1 = to_pixel(cmd.ClipRect.w, target->params.h)};
}

}  // namespace

ImGuiLayer::ImGuiLayer(pl_gpu gpu, pl_log log) : gpu_(gpu), dispatch_(pl_dispatch_create(log, gpu)) {}

ImGuiLayer::~ImGuiLayer() {
  for (pl_tex& tex : textures_) pl_tex_destroy(gpu_, &tex);
  pl_tex_destroy(gpu_, &cache_);
  pl_dispatch_destroy(&dispatch_);
}

void ImGuiLayer::configure(ImGuiIO& io) {
  io.BackendRendererName = kBackendName;
  io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures | ImGuiBackendFlags_RendererHasVtxOffset;
}

pl_tex ImGuiLayer::texture_of(std::uint64_t id) const {
  if (id == ImTextureID_Invalid || id > textures_.size()) return nullptr;
  return textures_[static_cast<std::size_t>(id - 1)];
}

std::uint64_t ImGuiLayer::adopt(pl_tex tex) {
  auto free = std::ranges::find(textures_, nullptr);
  if (free == textures_.end()) free = textures_.insert(textures_.end(), nullptr);
  *free = tex;
  return static_cast<std::uint64_t>(free - textures_.begin()) + 1;
}

void ImGuiLayer::destroy(ImTextureData& texture) {
  const std::uint64_t id = texture.GetTexID();
  if (texture_of(id) != nullptr) {
    // libplacebo diffère la destruction réelle après le travail GPU en cours.
    pl_tex_destroy(gpu_, &textures_[static_cast<std::size_t>(id - 1)]);
  }
  texture.SetTexID(ImTextureID_Invalid);
  texture.SetStatus(ImTextureStatus_Destroyed);
}

void ImGuiLayer::release_textures(ImGuiPlatformIO& platform) {
  for (ImTextureData* texture : platform.Textures) {
    if (texture_of(texture->GetTexID()) != nullptr) destroy(*texture);
  }
}

std::expected<void, Refusal> ImGuiLayer::update_texture(ImTextureData& texture) {
  if (texture.Status == ImTextureStatus_WantDestroy) {
    destroy(texture);
    return {};
  }
  if (texture.Status != ImTextureStatus_WantCreate && texture.Status != ImTextureStatus_WantUpdates)
    return {};

  pl_tex tex = texture_of(texture.GetTexID());
  const bool create = texture.Status == ImTextureStatus_WantCreate || tex == nullptr;
  if (create) {
    if (tex != nullptr) destroy(texture);  // recréée (atlas agrandi)
    pl_tex_params params{};
    params.w = texture.Width;
    params.h = texture.Height;
    params.format = pl_find_named_fmt(gpu_, texture.Format == ImTextureFormat_Alpha8 ? "r8" : "rgba8");
    params.sampleable = true;
    params.host_writable = true;
    params.debug_tag = "ajp interface";
    if (params.format == nullptr)
      return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "rgba8 ou r8"});
    tex = pl_tex_create(gpu_, &params);
    if (tex == nullptr) return std::unexpected(render_failed("pl_tex_create (texture de l'interface)"));
    texture.SetTexID(adopt(tex));
  }
  // Envoi depuis l'origine jusqu'au coin bas-droit des rectangles à jour (toute la texture à la
  // création) : le chemin Vulkan de libplacebo 7.360 prend la fin du rectangle pour sa taille
  // (imageExtent = rc.x1), si bien qu'un rectangle qui ne part pas de (0, 0) déborde de l'image.
  int right = texture.Width;
  int bottom = texture.Height;
  if (!create) {
    right = 0;
    bottom = 0;
    for (const ImTextureRect& rect : texture.Updates) {
      right = std::max(right, rect.x + rect.w);
      bottom = std::max(bottom, rect.y + rect.h);
    }
  }
  if (right > 0 && bottom > 0) {
    pl_tex_transfer_params transfer{};
    transfer.tex = tex;
    transfer.rc = {.x0 = 0, .y0 = 0, .z0 = 0, .x1 = right, .y1 = bottom, .z1 = 1};
    transfer.row_pitch = static_cast<std::size_t>(texture.GetPitch());
    transfer.ptr = texture.GetPixelsAt(0, 0);
    if (!pl_tex_upload(gpu_, &transfer))
      return std::unexpected(render_failed("pl_tex_upload (texture de l'interface)"));
  }
  texture.SetStatus(ImTextureStatus_OK);
  return {};
}

std::expected<void, Refusal> ImGuiLayer::draw(const ImDrawData& data, const pl_frame& target,
                                              float white_nits, std::uint64_t serial) {
  pl_dispatch_reset_frame(dispatch_);
  if (data.Textures != nullptr) {
    for (ImTextureData* texture : *data.Textures) {
      if (auto ok = update_texture(*texture); !ok) return ok;
    }
  }
  const pl_tex target_tex = target.planes[0].texture;
  if (!has_caps(target_tex->params.format->caps, PL_FMT_CAP_BLENDABLE))
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "cible de l'interface non mélangeable"});
  if (data.DisplayPos.x != 0.0F || data.DisplayPos.y != 0.0F || data.FramebufferScale.x != 1.0F ||
      data.FramebufferScale.y != 1.0F) {
    return std::unexpected(
        Refusal{RefusalCode::InputMismatch, "interface : DisplayPos (0, 0) et échelle 1 attendus"});
  }
  if (serial == 0) return draw_lists(data, target, target_tex, white_nits, nullptr);
  if (!cache_matches(serial, target, white_nits)) {
    cache_serial_ = 0;
    const int w = target_tex->params.w;
    const int h = target_tex->params.h;
    if (cache_ == nullptr || cache_->params.w != w || cache_->params.h != h) {
      pl_tex_destroy(gpu_, &cache_);
      constexpr auto kNeeded =
          static_cast<pl_fmt_caps>(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLENDABLE |
                                   PL_FMT_CAP_LINEAR | PL_FMT_CAP_BLITTABLE);
      pl_tex_params params{};
      params.w = w;
      params.h = h;
      // Une cible flottante (scRGB sous Windows) porte des valeurs au-delà de 1 : une texture gardée en
      // entiers normalisés les écrêterait et ternirait l'interface. Sinon rgba16, plus précis sur [0, 1].
      const bool float_target = target_tex->params.format->type == PL_FMT_FLOAT;
      const std::array<const char*, 2> names = float_target ? std::array<const char*, 2>{"rgba16f", "rgba32f"}
                                                            : std::array<const char*, 2>{"rgba16", "rgba16f"};
      for (const char* name : names) {
        const pl_fmt fmt = pl_find_named_fmt(gpu_, name);
        if (fmt != nullptr && has_caps(fmt->caps, kNeeded)) {
          params.format = fmt;
          break;
        }
      }
      params.sampleable = true;
      params.renderable = true;
      params.blit_dst = true;  // pl_tex_clear
      params.debug_tag = "ajp interface gardée";
      if (params.format != nullptr) cache_ = pl_tex_create(gpu_, &params);
      // Sans texture gardée, le dessin direct d'avant.
      if (cache_ == nullptr) return draw_lists(data, target, target_tex, white_nits, nullptr);
    }
    pl_tex_clear(gpu_, cache_, std::array<float, 4>{0.0F, 0.0F, 0.0F, 0.0F}.data());
    pl_rect2d bounds{};
    if (auto drawn = draw_lists(data, target, cache_, white_nits, &bounds); !drawn) return drawn;
    cache_serial_ = serial;
    cache_flipped_ = target.planes[0].flipped;
    cache_white_ = white_nits;
    cache_color_ = target.color;
    cache_repr_ = target.repr;
    cache_bounds_ = bounds;
  }
  return blend_cache(target_tex);
}

void ImGuiLayer::drop_cache() {
  pl_tex_destroy(gpu_, &cache_);
  cache_serial_ = 0;
}

bool ImGuiLayer::cache_matches(std::uint64_t serial, const pl_frame& target, float white_nits) const {
  const pl_tex t = target.planes[0].texture;
  return cache_ != nullptr && cache_serial_ == serial && cache_->params.w == t->params.w &&
         cache_->params.h == t->params.h && cache_flipped_ == target.planes[0].flipped &&
         cache_white_ == white_nits && pl_color_space_equal(&cache_color_, &target.color) &&
         pl_color_repr_equal(&cache_repr_, &target.repr);
}

// La texture gardée, déjà prémultipliée par pl_alpha_overlay (couleur × alpha, alpha accumulé), se mélange
// par « over » : cible × (1 − alpha) + couleur. Seul le rectangle couvert par l'interface est touché.
std::expected<void, Refusal> ImGuiLayer::blend_cache(pl_tex target_tex) {
  const pl_rect2d& b = cache_bounds_;
  if (b.x1 <= b.x0 || b.y1 <= b.y0) return {};
  pl_shader sh = pl_dispatch_begin(dispatch_);
  pl_sample_src src{};
  src.tex = cache_;
  src.rect = {static_cast<float>(b.x0), static_cast<float>(b.y0), static_cast<float>(b.x1),
              static_cast<float>(b.y1)};
  if (!pl_shader_sample_direct(sh, &src)) {
    pl_dispatch_abort(dispatch_, &sh);
    return std::unexpected(render_failed("pl_shader_sample_direct (interface gardée)"));
  }
  static constexpr pl_blend_params kPremultiplied{.src_rgb = PL_BLEND_ONE,
                                                  .dst_rgb = PL_BLEND_ONE_MINUS_SRC_ALPHA,
                                                  .src_alpha = PL_BLEND_ONE,
                                                  .dst_alpha = PL_BLEND_ONE_MINUS_SRC_ALPHA};
  pl_dispatch_params params{};
  params.shader = &sh;
  params.target = target_tex;
  params.rect = b;
  params.blend_params = &kPremultiplied;
  if (!pl_dispatch_finish(dispatch_, &params))
    return std::unexpected(render_failed("pl_dispatch_finish (interface gardée)"));
  return {};
}

std::expected<void, Refusal> ImGuiLayer::draw_lists(const ImDrawData& data, const pl_frame& target,
                                                    pl_tex into, float white_nits, pl_rect2d* bounds) {
  const pl_fmt float2 = pl_find_vertex_fmt(gpu_, PL_FMT_FLOAT, 2);
  const pl_fmt rgba8 = pl_find_named_fmt(gpu_, "rgba8");
  if (float2 == nullptr || rgba8 == nullptr || !has_caps(rgba8->caps, PL_FMT_CAP_VERTEX))
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "formats de sommet de l'interface"});
  const std::array<pl_vertex_attrib, 3> attribs{{
      {.name = "ajp_pos", .fmt = float2, .offset = offsetof(ImDrawVert, pos)},
      {.name = "ajp_uv", .fmt = float2, .offset = offsetof(ImDrawVert, uv)},
      {.name = "ajp_color", .fmt = rgba8, .offset = offsetof(ImDrawVert, col)},
  }};
  pl_color_space ui = pl_color_space_srgb;
  if (pl_color_space_is_hdr(&target.color)) ui.hdr.max_luma = white_nits;
  pl_color_repr repr = target.repr;
  repr.alpha = PL_ALPHA_INDEPENDENT;  // alpha droit : le mélange pl_alpha_overlay le veut ainsi
  const bool flipped = target.planes[0].flipped;
  const int height = into->params.h;

  for (const ImDrawList* list : data.CmdLists) {
    for (const ImDrawCmd& cmd : list->CmdBuffer) {
      if (cmd.UserCallback != nullptr) {
        if (cmd.UserCallback != ImDrawCallback_ResetRenderState) cmd.UserCallback(list, &cmd);
        continue;
      }
      const pl_tex tex = texture_of(cmd.GetTexID());
      const pl_rect2d scissors = clip_rect(cmd, into);
      if (tex == nullptr || cmd.ElemCount == 0 || scissors.x1 <= scissors.x0 || scissors.y1 <= scissors.y0)
        continue;

      pl_shader sh = pl_dispatch_begin(dispatch_);
      const pl_shader_desc desc{.desc = {.name = "ajp_ui_tex", .type = PL_DESC_SAMPLED_TEX},
                                .binding = {.object = tex, .sample_mode = PL_TEX_SAMPLE_LINEAR}};
      const bool alpha_only = tex->params.format->num_components == 1;
      pl_custom_shader custom{};
      custom.description = "interface ImGui";
      custom.body = alpha_only ? "color = ajp_color * vec4(1.0, 1.0, 1.0, texture(ajp_ui_tex, ajp_uv).r);"
                               : "color = ajp_color * texture(ajp_ui_tex, ajp_uv);";
      custom.input = PL_SHADER_SIG_NONE;
      custom.output = PL_SHADER_SIG_COLOR;
      custom.descriptors = &desc;
      custom.num_descriptors = 1;
      if (!pl_shader_custom(sh, &custom)) {
        pl_dispatch_abort(dispatch_, &sh);
        return std::unexpected(render_failed("pl_shader_custom (interface)"));
      }
      pl_shader_color_map(sh, nullptr, ui, target.color, nullptr, false);
      pl_shader_encode_color(sh, &repr);

      pl_dispatch_vertex_params params{};
      params.shader = &sh;
      params.target = into;
      params.scissors = scissors;
      params.blend_params = &pl_alpha_overlay;
      params.vertex_attribs = attribs.data();
      params.num_vertex_attribs = static_cast<int>(attribs.size());
      params.vertex_stride = sizeof(ImDrawVert);
      params.vertex_position_idx = 0;
      params.vertex_coords = PL_COORDS_ABSOLUTE;
      params.vertex_flipped = flipped;
      params.vertex_type = PL_PRIM_TRIANGLE_LIST;
      params.vertex_count = static_cast<int>(cmd.ElemCount);
      params.vertex_data = list->VtxBuffer.Data + cmd.VtxOffset;
      params.index_data = list->IdxBuffer.Data + cmd.IdxOffset;
      params.index_fmt = sizeof(ImDrawIdx) == 2 ? PL_INDEX_UINT16 : PL_INDEX_UINT32;
      if (!pl_dispatch_vertex(dispatch_, &params))
        return std::unexpected(render_failed("pl_dispatch_vertex (interface)"));
      if (bounds != nullptr) {
        // libplacebo retourne la découpe avec les sommets (vertex_flipped) : la même, en lignes de la
        // texture.
        const pl_rect2d in_tex{.x0 = scissors.x0,
                               .y0 = flipped ? height - scissors.y1 : scissors.y0,
                               .x1 = scissors.x1,
                               .y1 = flipped ? height - scissors.y0 : scissors.y1};
        *bounds = bounds->x1 <= bounds->x0 ? in_tex
                                           : pl_rect2d{.x0 = std::min(bounds->x0, in_tex.x0),
                                                       .y0 = std::min(bounds->y0, in_tex.y0),
                                                       .x1 = std::max(bounds->x1, in_tex.x1),
                                                       .y1 = std::max(bounds->y1, in_tex.y1)};
      }
    }
  }
  return {};
}

}  // namespace ajp::app::play
