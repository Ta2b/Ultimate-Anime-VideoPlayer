#include "render/imgui_layer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include <imgui.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/custom.h>

#include "render/c_structs.h"

namespace ajp::render {
namespace {

constexpr const char* kBackendName = "ajp_render_libplacebo";

Refusal render_failed(const char* call) {
  return Refusal{RefusalCode::RenderFailed, call};
}

// Rectangle de découpe d'ImGui, en pixels de la cible ; vide s'il sort de la cible.
pl_rect2d clip_rect(const ImDrawCmd& cmd, const ImDrawData& data, pl_tex target) {
  const ImVec2 off = data.DisplayPos;
  const ImVec2 scale = data.FramebufferScale;
  const auto to_pixel = [](float v, int limit) {
    return std::clamp(static_cast<int>(std::lround(v)), 0, limit);
  };
  return pl_rect2d{.x0 = to_pixel((cmd.ClipRect.x - off.x) * scale.x, target->params.w),
                   .y0 = to_pixel((cmd.ClipRect.y - off.y) * scale.y, target->params.h),
                   .x1 = to_pixel((cmd.ClipRect.z - off.x) * scale.x, target->params.w),
                   .y1 = to_pixel((cmd.ClipRect.w - off.y) * scale.y, target->params.h)};
}

}  // namespace

ImGuiLayer::ImGuiLayer(pl_gpu gpu, pl_log log) : gpu_(gpu), dispatch_(pl_dispatch_create(log, gpu)) {}

ImGuiLayer::~ImGuiLayer() {
  for (pl_tex& tex : textures_) pl_tex_destroy(gpu_, &tex);
  pl_dispatch_destroy(&dispatch_);
}

void ImGuiLayer::configure(ImGuiIO& io) {
  io.BackendRendererName = kBackendName;
  io.BackendFlags = static_cast<ImGuiBackendFlags>(detail::flag_bits(
      io.BackendFlags, ImGuiBackendFlags_RendererHasTextures, ImGuiBackendFlags_RendererHasVtxOffset));
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
    auto params = detail::zeroed<pl_tex_params>();
    params.w = texture.Width;
    params.h = texture.Height;
    params.format = pl_find_named_fmt(gpu_, texture.Format == ImTextureFormat_Alpha8 ? "r8" : "rgba8");
    params.sampleable = true;
    params.host_writable = true;
    params.debug_tag = "ajp OSD";
    if (params.format == nullptr)
      return std::unexpected(Refusal{RefusalCode::FeatureUnavailable, "rgba8 ou r8"});
    tex = pl_tex_create(gpu_, &params);
    if (tex == nullptr) return std::unexpected(render_failed("pl_tex_create (texture de l'OSD)"));
    texture.SetTexID(adopt(tex));
  }
  // Toute la texture à la création, sinon les rectangles demandés.
  const ImTextureRect whole{.x = 0,
                            .y = 0,
                            .w = static_cast<unsigned short>(texture.Width),
                            .h = static_cast<unsigned short>(texture.Height)};
  const std::span<const ImTextureRect> rects =
      create ? std::span<const ImTextureRect>(&whole, 1)
             : std::span<const ImTextureRect>(texture.Updates.Data,
                                              static_cast<std::size_t>(texture.Updates.Size));
  for (const ImTextureRect& rect : rects) {
    pl_tex_transfer_params transfer{};
    transfer.tex = tex;
    transfer.rc = {
        .x0 = rect.x, .y0 = rect.y, .z0 = 0, .x1 = rect.x + rect.w, .y1 = rect.y + rect.h, .z1 = 1};
    transfer.row_pitch = static_cast<std::size_t>(texture.GetPitch());
    transfer.ptr = texture.GetPixelsAt(rect.x, rect.y);
    if (!pl_tex_upload(gpu_, &transfer))
      return std::unexpected(render_failed("pl_tex_upload (texture de l'OSD)"));
  }
  texture.SetStatus(ImTextureStatus_OK);
  return {};
}

std::expected<void, Refusal> ImGuiLayer::draw(ImDrawData& data, const pl_frame& target,
                                              float osd_white_nits) {
  pl_dispatch_reset_frame(dispatch_);
  if (data.Textures != nullptr) {
    for (ImTextureData* texture : *data.Textures) {
      if (auto ok = update_texture(*texture); !ok) return ok;
    }
  }
  const pl_tex target_tex = target.planes[0].texture;
  if (!detail::has_all(target_tex->params.format->caps, detail::flag_bits(PL_FMT_CAP_BLENDABLE))) {
    return std::unexpected(Refusal{RefusalCode::FeatureUnavailable, "cible de l'OSD non mélangeable"});
  }
  const pl_fmt float2 = pl_find_vertex_fmt(gpu_, PL_FMT_FLOAT, 2);
  const pl_fmt rgba8 = pl_find_named_fmt(gpu_, "rgba8");
  if (float2 == nullptr || rgba8 == nullptr ||
      !detail::has_all(rgba8->caps, detail::flag_bits(PL_FMT_CAP_VERTEX))) {
    return std::unexpected(Refusal{RefusalCode::FeatureUnavailable, "formats de sommet de l'OSD"});
  }
  const std::array<pl_vertex_attrib, 3> attribs{{
      {.name = "ajp_pos", .fmt = float2, .offset = offsetof(ImDrawVert, pos)},
      {.name = "ajp_uv", .fmt = float2, .offset = offsetof(ImDrawVert, uv)},
      {.name = "ajp_color", .fmt = rgba8, .offset = offsetof(ImDrawVert, col)},
  }};
  pl_color_space osd = pl_color_space_srgb;
  if (pl_color_space_is_hdr(&target.color)) osd.hdr.max_luma = osd_white_nits;
  pl_color_repr repr = target.repr;
  repr.alpha = PL_ALPHA_INDEPENDENT;  // alpha droit : le mélange pl_alpha_overlay le veut ainsi
  const bool unit_scale = data.DisplayPos.x == 0.0F && data.DisplayPos.y == 0.0F &&
                          data.FramebufferScale.x == 1.0F && data.FramebufferScale.y == 1.0F;
  if (!unit_scale) {
    return std::unexpected(
        Refusal{RefusalCode::InputMismatch, "OSD : DisplayPos (0, 0) et échelle 1 attendus"});
  }

  for (const ImDrawList* list : data.CmdLists) {
    for (const ImDrawCmd& cmd : list->CmdBuffer) {
      if (cmd.UserCallback != nullptr) {
        if (cmd.UserCallback != ImDrawCallback_ResetRenderState) cmd.UserCallback(list, &cmd);
        continue;
      }
      const pl_tex tex = texture_of(cmd.GetTexID());
      const pl_rect2d scissors = clip_rect(cmd, data, target_tex);
      if (tex == nullptr || cmd.ElemCount == 0 || scissors.x1 <= scissors.x0 || scissors.y1 <= scissors.y0)
        continue;

      pl_shader sh = pl_dispatch_begin(dispatch_);
      const pl_shader_desc desc{.desc = {.name = "osd_tex", .type = PL_DESC_SAMPLED_TEX},
                                .binding = {.object = tex, .sample_mode = PL_TEX_SAMPLE_LINEAR}};
      const bool alpha_only = tex->params.format->num_components == 1;
      pl_custom_shader custom{};
      custom.description = "OSD ImGui";
      custom.body = alpha_only ? "color = ajp_color * vec4(1.0, 1.0, 1.0, texture(osd_tex, ajp_uv).r);"
                               : "color = ajp_color * texture(osd_tex, ajp_uv);";
      custom.input = PL_SHADER_SIG_NONE;
      custom.output = PL_SHADER_SIG_COLOR;
      custom.descriptors = &desc;
      custom.num_descriptors = 1;
      if (!pl_shader_custom(sh, &custom)) {
        pl_dispatch_abort(dispatch_, &sh);
        return std::unexpected(render_failed("pl_shader_custom (OSD)"));
      }
      pl_shader_color_map(sh, nullptr, osd, target.color, nullptr, false);
      pl_shader_encode_color(sh, &repr);

      pl_dispatch_vertex_params params{};
      params.shader = &sh;
      params.target = target_tex;
      params.scissors = scissors;
      params.blend_params = &pl_alpha_overlay;
      params.vertex_attribs = attribs.data();
      params.num_vertex_attribs = static_cast<int>(attribs.size());
      params.vertex_stride = sizeof(ImDrawVert);
      params.vertex_position_idx = 0;
      params.vertex_coords = PL_COORDS_ABSOLUTE;
      params.vertex_flipped = target.planes[0].flipped;
      params.vertex_type = PL_PRIM_TRIANGLE_LIST;
      params.vertex_count = static_cast<int>(cmd.ElemCount);
      params.vertex_data = list->VtxBuffer.Data + cmd.VtxOffset;
      params.index_data = list->IdxBuffer.Data + cmd.IdxOffset;
      params.index_fmt = sizeof(ImDrawIdx) == 2 ? PL_INDEX_UINT16 : PL_INDEX_UINT32;
      if (!pl_dispatch_vertex(dispatch_, &params))
        return std::unexpected(render_failed("pl_dispatch_vertex (OSD)"));
    }
  }
  return {};
}

}  // namespace ajp::render
