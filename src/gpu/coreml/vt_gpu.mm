#include "gpu/coreml/vt_gpu.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <string>
#include <utility>

namespace ajp::gpu::coreml_rt {
namespace {

std::unexpected<Refusal> refuse_gpu(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{
      .code = code, .detail = std::format("conversions de l'interpolation d'Apple sur le GPU : {}", text)});
}

std::string describe(NSError* error) {
  if (error == nil) return "erreur inconnue";
  const char* text = error.localizedDescription.UTF8String;
  return std::format("{} ({} {})", text != nullptr ? text : "?", error.domain.UTF8String, error.code);
}

// Le plus grand côté d'une texture 2D sur les GPU des Mac (familles Apple et Mac2).
constexpr int kMaxTextureSide = 16384;
// Éléments comparés par fil : les drapeaux s'écrivent une fois par groupe SIMD de 32 fils.
constexpr std::uint32_t kSpan = 8;

// Les noyaux. Les calculs suivent vt_pixels.cpp opération par opération, en mathématiques IEEE
// (sans les approximations rapides de Metal) : mêmes valeurs à l'arrondi près. Les codes d'un
// format YCbCr passent par des textures normalisées (R8Unorm, R16Unorm…) : un code k s'écrit
// k × 2^shift / scale et se relit à l'entier le plus proche, exactement. Un octet de BGRA se relit
// par la table de ses 256 demi-flottants, que vt::unpack calcule : au bit près.
constexpr const char* kSource = R"(
#include <metal_stdlib>
using namespace metal;

struct Frame {
  uint width;
  uint height;
  uint row;  // la première rangée de l'image dans le tenseur de la chaîne
};

struct Yuv {
  float kr, kg, kb, cb_scale, cr_scale, gcb, gcr;
  float y_offset, y_scale, c_mid, c_scale, max;
  float y_unit, c_unit;
  float scale;  // 255 (octets) ou 65535 (mots de 16 bits)
  uint shift;   // 6 : 10 bits dans les poids forts
};

static float quantize(float v, float max) { return round(clamp(v, 0.0f, max)); }
static float stored(float code, constant Yuv& l) { return (code * float(1u << l.shift)) / l.scale; }
static float code_of(float v, constant Yuv& l) { return float(uint(round(v * l.scale)) >> l.shift); }

static float3 rgb_at(texture2d<half, access::read> chain, constant Frame& f, uint x, uint y) {
  const uint r = f.row + y;
  return float3(float(chain.read(uint2(x, r)).r), float(chain.read(uint2(x, r + f.height)).r),
                float(chain.read(uint2(x, r + 2 * f.height)).r));
}

kernel void pack_rgba_half(texture2d<half, access::read> chain [[texture(0)]],
                           texture2d<half, access::write> out [[texture(1)]],
                           constant Frame& f [[buffer(0)]],
                           uint2 p [[thread_position_in_grid]]) {
  if (p.x >= f.width || p.y >= f.height) return;
  const uint r = f.row + p.y;
  out.write(half4(chain.read(uint2(p.x, r)).r, chain.read(uint2(p.x, r + f.height)).r,
                  chain.read(uint2(p.x, r + 2 * f.height)).r, 1.0h), p);
}

kernel void pack_bgra8(texture2d<half, access::read> chain [[texture(0)]],
                       texture2d<float, access::write> out [[texture(1)]],
                       constant Frame& f [[buffer(0)]],
                       uint2 p [[thread_position_in_grid]]) {
  if (p.x >= f.width || p.y >= f.height) return;
  const float3 c = rgb_at(chain, f, p.x, p.y);
  out.write(float4(quantize(c.r * 255.0f, 255.0f) / 255.0f, quantize(c.g * 255.0f, 255.0f) / 255.0f,
                   quantize(c.b * 255.0f, 255.0f) / 255.0f, 1.0f), p);
}

// Un fil par bloc 2 × 2 (largeur et hauteur paires).
kernel void pack_yuv(texture2d<half, access::read> chain [[texture(0)]],
                     texture2d<float, access::write> luma [[texture(1)]],
                     texture2d<float, access::write> chroma [[texture(2)]],
                     constant Frame& f [[buffer(0)]],
                     constant Yuv& l [[buffer(1)]],
                     uint2 p [[thread_position_in_grid]]) {
  const uint x = 2 * p.x;
  const uint y = 2 * p.y;
  if (x >= f.width || y >= f.height) return;
  float3 sum = float3(0.0f);
  for (uint dy = 0; dy < 2; ++dy) {
    for (uint dx = 0; dx < 2; ++dx) {
      const float3 c = rgb_at(chain, f, x + dx, y + dy);
      const float value = (l.kr * c.r) + (l.kg * c.g) + (l.kb * c.b);
      luma.write(float4(stored(quantize(l.y_offset + (value * l.y_scale), l.max), l)), uint2(x + dx, y + dy));
      sum += c;
    }
  }
  const float3 m = sum / 4.0f;
  const float value = (l.kr * m.r) + (l.kg * m.g) + (l.kb * m.b);
  const float cb = (m.b - value) / l.cb_scale;
  const float cr = (m.r - value) / l.cr_scale;
  chroma.write(float4(stored(quantize(l.c_mid + (cb * l.c_scale), l.max), l),
                      stored(quantize(l.c_mid + (cr * l.c_scale), l.max), l), 0.0f, 0.0f), p);
}

kernel void unpack_rgba_half(texture2d<half, access::read> in [[texture(0)]],
                             texture2d<half, access::write> chain [[texture(1)]],
                             constant Frame& f [[buffer(0)]],
                             uint2 p [[thread_position_in_grid]]) {
  if (p.x >= f.width || p.y >= f.height) return;
  const half4 c = in.read(p);
  chain.write(half4(c.r), p);
  chain.write(half4(c.g), uint2(p.x, p.y + f.height));
  chain.write(half4(c.b), uint2(p.x, p.y + 2 * f.height));
}

kernel void unpack_bgra8(texture2d<float, access::read> in [[texture(0)]],
                         texture2d<half, access::write> chain [[texture(1)]],
                         constant Frame& f [[buffer(0)]],
                         constant ushort* halves [[buffer(1)]],
                         uint2 p [[thread_position_in_grid]]) {
  if (p.x >= f.width || p.y >= f.height) return;
  const uint4 c = uint4(round(clamp(in.read(p), 0.0f, 1.0f) * 255.0f));
  chain.write(half4(as_type<half>(halves[c.r])), p);
  chain.write(half4(as_type<half>(halves[c.g])), uint2(p.x, p.y + f.height));
  chain.write(half4(as_type<half>(halves[c.b])), uint2(p.x, p.y + 2 * f.height));
}

static float2 chroma_at(texture2d<float, access::read> chroma, constant Yuv& l, int cx, int cy, int cw,
                        int ch) {
  const float2 v = chroma.read(uint2(uint(clamp(cx, 0, cw - 1)), uint(clamp(cy, 0, ch - 1)))).rg;
  return float2((code_of(v.x, l) - l.c_mid) * l.c_unit, (code_of(v.y, l) - l.c_mid) * l.c_unit);
}

// La chroma bilinéaire au centre de chaque bloc 2 × 2, comme vt::unpack.
kernel void unpack_yuv(texture2d<float, access::read> luma [[texture(0)]],
                       texture2d<float, access::read> chroma [[texture(1)]],
                       texture2d<half, access::write> chain [[texture(2)]],
                       constant Frame& f [[buffer(0)]],
                       constant Yuv& l [[buffer(1)]],
                       uint2 p [[thread_position_in_grid]]) {
  if (p.x >= f.width || p.y >= f.height) return;
  const int cw = int(f.width + 1) / 2;
  const int ch = int(f.height + 1) / 2;
  const int x = int(p.x);
  const int y = int(p.y);
  const int cx = x / 2;
  const int nx = (x % 2 == 0) ? cx - 1 : cx + 1;
  const int cy = y / 2;
  const int ny = (y % 2 == 0) ? cy - 1 : cy + 1;
  const float2 here = (0.75f * chroma_at(chroma, l, cx, cy, cw, ch)) + (0.25f * chroma_at(chroma, l, nx, cy, cw, ch));
  const float2 there = (0.75f * chroma_at(chroma, l, cx, ny, cw, ch)) + (0.25f * chroma_at(chroma, l, nx, ny, cw, ch));
  const float cb = (0.75f * here.x) + (0.25f * there.x);
  const float cr = (0.75f * here.y) + (0.25f * there.y);
  const float value = (code_of(luma.read(p).r, l) - l.y_offset) * l.y_unit;
  chain.write(half4(half(value + (l.cr_scale * cr))), p);
  chain.write(half4(half(value - (l.gcb * cb) - (l.gcr * cr))), uint2(p.x, p.y + f.height));
  chain.write(half4(half(value + (l.cb_scale * cb))), uint2(p.x, p.y + 2 * f.height));
}

// Un fil par SPAN éléments d'une rangée des trois plans ; un drapeau levé par groupe SIMD au plus.
constant uint SPAN = 8;
kernel void differ(texture2d<half, access::read> chain [[texture(0)]],
                   texture2d<half, access::read> kept0 [[texture(1)]],
                   texture2d<half, access::read> kept1 [[texture(2)]],
                   constant Frame& f [[buffer(0)]],
                   device atomic_uint* flags [[buffer(1)]],
                   uint2 p [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = 3 * f.height;
  bool d0 = false;
  bool d1 = false;
  bool d2 = false;
  if (p.y < rows) {
    for (uint i = 0; i < SPAN; ++i) {
      const uint x = (p.x * SPAN) + i;
      if (x >= f.width) break;
      const ushort a = as_type<ushort>(chain.read(uint2(x, p.y)).r);
      const ushort b = as_type<ushort>(chain.read(uint2(x, p.y + rows)).r);
      const ushort k0 = as_type<ushort>(kept0.read(uint2(x, p.y)).r);
      const ushort k1 = as_type<ushort>(kept1.read(uint2(x, p.y)).r);
      d0 = d0 || a != k0;
      d1 = d1 || b != k1;
      d2 = d2 || a != k1;
    }
  }
  d0 = simd_any(d0);
  d1 = simd_any(d1);
  d2 = simd_any(d2);
  if (lane != 0) return;
  if (d0) atomic_store_explicit(&flags[0], 1u, memory_order_relaxed);
  if (d1) atomic_store_explicit(&flags[1], 1u, memory_order_relaxed);
  if (d2) atomic_store_explicit(&flags[2], 1u, memory_order_relaxed);
}
)";

struct Frame {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t row;
};

struct Yuv {
  float kr, kg, kb, cb_scale, cr_scale, gcb, gcr;
  float y_offset, y_scale, c_mid, c_scale, max;
  float y_unit, c_unit;
  float scale;
  std::uint32_t shift;
};

using ByteHalves = std::array<std::uint16_t, 256>;

// Les demi-flottants des 256 octets d'un canal de BGRA, par vt::unpack lui-même.
ByteHalves byte_halves() {
  std::array<std::uint8_t, 256 * 4> bgra{};
  for (std::size_t v = 0; v < 256; ++v) {
    for (std::size_t c = 0; c < 3; ++c) bgra[(v * 4) + c] = static_cast<std::uint8_t>(v);
    bgra[(v * 4) + 3] = 0xFF;
  }
  std::array<std::uint16_t, 3 * 256> rgb{};
  vt::Planes planes;
  planes.data[0] = reinterpret_cast<std::byte*>(bgra.data());
  planes.stride[0] = bgra.size();
  vt::unpack(vt::Pixels::Bgra8, planes, reinterpret_cast<std::byte*>(rgb.data()), TensorType::F16, 256, 1,
             vt::Rows{.first = 0, .last = 1});
  ByteHalves halves{};
  std::copy_n(rgb.begin(), halves.size(), halves.begin());
  return halves;
}

// Les constantes de vt_pixels.cpp, telles que le C++ les calcule.
Yuv yuv_of(vt::Pixels format) {
  const vt::Levels l = vt::levels(format);
  return {.kr = vt::kKr,
          .kg = vt::kKg,
          .kb = vt::kKb,
          .cb_scale = vt::kCbScale,
          .cr_scale = vt::kCrScale,
          .gcb = vt::kGCb,
          .gcr = vt::kGCr,
          .y_offset = l.y_offset,
          .y_scale = l.y_scale,
          .c_mid = l.c_mid,
          .c_scale = l.c_scale,
          .max = l.max,
          .y_unit = 1.0F / l.y_scale,
          .c_unit = 1.0F / l.c_scale,
          .scale = l.wide ? 65535.0F : 255.0F,
          .shift = l.shift};
}

// Les textures d'un tampon de VideoToolbox : un plan RGB, ou la luma et la chroma entrelacée.
struct PlaneFormats {
  MTLPixelFormat first;
  MTLPixelFormat second;  // MTLPixelFormatInvalid : un seul plan
};

PlaneFormats plane_formats(vt::Pixels format) {
  switch (format) {
    case vt::Pixels::RgbaHalf:
      return {MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid};
    case vt::Pixels::Bgra8:
      return {MTLPixelFormatBGRA8Unorm, MTLPixelFormatInvalid};
    case vt::Pixels::Yuv420Full10:
    case vt::Pixels::Yuv420Video10:
      return {MTLPixelFormatR16Unorm, MTLPixelFormatRG16Unorm};
    case vt::Pixels::Yuv420Full8:
    case vt::Pixels::Yuv420Video8:
      break;
  }
  return {MTLPixelFormatR8Unorm, MTLPixelFormatRG8Unorm};
}

class MetalVtGpu final : public VtGpu {
 public:
  MetalVtGpu(vt::Pixels format, int width, int height, id<MTLDevice> device, id<MTLCommandQueue> queue,
             id<MTLComputePipelineState> pack, id<MTLComputePipelineState> unpack,
             id<MTLComputePipelineState> differ, std::array<id<MTLTexture>, 2> kept, id<MTLBuffer> flags)
      : format_(format),
        width_(static_cast<std::uint32_t>(width)),
        height_(static_cast<std::uint32_t>(height)),
        planes_(plane_formats(format)),
        yuv_(yuv_of(format)),
        device_(device),
        queue_(queue),
        pack_(pack),
        unpack_(unpack),
        differ_(differ),
        kept_(std::move(kept)),
        flags_(flags),
        byte_halves_(byte_halves()) {}

  std::expected<void, Refusal> pack(IOSurfaceRef input, std::span<const Pack> packs) override {
    @autoreleasepool {
      id<MTLTexture> chain = chain_texture(input, 7, MTLTextureUsageShaderRead);
      if (chain == nil) return refuse_gpu(RefusalCode::DeviceUnavailable, "texture de l'entrée refusée");
      id<MTLCommandBuffer> commands = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> compute = [commands computeCommandEncoder];
      [compute setComputePipelineState:pack_];
      [compute setTexture:chain atIndex:0];
      for (const Pack& p : packs) {
        const auto target = pixel_textures(p.target, MTLTextureUsageShaderWrite);
        if (!target) {
          [compute endEncoding];
          return refuse_gpu(RefusalCode::DeviceUnavailable, "texture d'un tampon de VideoToolbox refusée");
        }
        const Frame frame{
            .width = width_, .height = height_, .row = static_cast<std::uint32_t>(p.image) * 3 * height_};
        [compute setTexture:target->first atIndex:1];
        [compute setTexture:target->second atIndex:2];
        [compute setBytes:&frame length:sizeof(frame) atIndex:0];
        [compute setBytes:&yuv_ length:sizeof(yuv_) atIndex:1];
        if (vt::is_yuv(format_)) {
          dispatch(compute, pack_, width_ / 2, height_ / 2);
        } else {
          dispatch(compute, pack_, width_, height_);
        }
      }
      [compute endEncoding];
      id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
      for (const Pack& p : packs) {
        [blit copyFromTexture:chain
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, static_cast<NSUInteger>(p.image) * 3 * height_, 0)
                   sourceSize:MTLSizeMake(width_, 3 * height_, 1)
                    toTexture:kept_[static_cast<std::size_t>(p.keep)]
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
      }
      [blit endEncoding];
      return finish(commands, "écriture des tampons");
    }
  }

  std::expected<std::array<bool, 3>, Refusal> differs(IOSurfaceRef input) override {
    @autoreleasepool {
      id<MTLTexture> chain = chain_texture(input, 7, MTLTextureUsageShaderRead);
      if (chain == nil) return refuse_gpu(RefusalCode::DeviceUnavailable, "texture de l'entrée refusée");
      auto* flags = static_cast<std::uint32_t*>(flags_.contents);
      std::fill_n(flags, 3, 0U);
      id<MTLCommandBuffer> commands = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> compute = [commands computeCommandEncoder];
      [compute setComputePipelineState:differ_];
      [compute setTexture:chain atIndex:0];
      [compute setTexture:kept_[0] atIndex:1];
      [compute setTexture:kept_[1] atIndex:2];
      const Frame frame{.width = width_, .height = height_, .row = 0};
      [compute setBytes:&frame length:sizeof(frame) atIndex:0];
      [compute setBuffer:flags_ offset:0 atIndex:1];
      dispatch(compute, differ_, (width_ + kSpan - 1) / kSpan, 3 * height_);
      [compute endEncoding];
      if (auto ok = finish(commands, "comparaison des images"); !ok) return std::unexpected(ok.error());
      return std::array<bool, 3>{flags[0] != 0, flags[1] != 0, flags[2] != 0};
    }
  }

  void swap_kept() override { std::swap(kept_[0], kept_[1]); }

  std::expected<void, Refusal> unpack(CVPixelBufferRef source, IOSurfaceRef output) override {
    @autoreleasepool {
      id<MTLTexture> chain = chain_texture(output, 3, MTLTextureUsageShaderWrite);
      if (chain == nil) return refuse_gpu(RefusalCode::DeviceUnavailable, "texture de la sortie refusée");
      const auto in = pixel_textures(source, MTLTextureUsageShaderRead);
      if (!in) return refuse_gpu(RefusalCode::DeviceUnavailable, "texture du tampon de VideoToolbox refusée");
      id<MTLCommandBuffer> commands = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> compute = [commands computeCommandEncoder];
      [compute setComputePipelineState:unpack_];
      const Frame frame{.width = width_, .height = height_, .row = 0};
      if (vt::is_yuv(format_)) {
        [compute setTexture:in->first atIndex:0];
        [compute setTexture:in->second atIndex:1];
        [compute setTexture:chain atIndex:2];
      } else {
        [compute setTexture:in->first atIndex:0];
        [compute setTexture:chain atIndex:1];
      }
      [compute setBytes:&frame length:sizeof(frame) atIndex:0];
      if (format_ == vt::Pixels::Bgra8) {
        [compute setBytes:byte_halves_.data() length:sizeof(byte_halves_) atIndex:1];
      } else {
        [compute setBytes:&yuv_ length:sizeof(yuv_) atIndex:1];
      }
      dispatch(compute, unpack_, width_, height_);
      [compute endEncoding];
      return finish(commands, "lecture du tampon");
    }
  }

  std::expected<void, Refusal> copy(IOSurfaceRef input, int image, IOSurfaceRef output) override {
    @autoreleasepool {
      id<MTLTexture> from = chain_texture(input, 7, MTLTextureUsageShaderRead);
      id<MTLTexture> to = chain_texture(output, 3, MTLTextureUsageShaderWrite);
      if (from == nil || to == nil) return refuse_gpu(RefusalCode::DeviceUnavailable, "texture refusée");
      id<MTLCommandBuffer> commands = [queue_ commandBuffer];
      id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
      [blit copyFromTexture:from
                sourceSlice:0
                sourceLevel:0
               sourceOrigin:MTLOriginMake(0, static_cast<NSUInteger>(image) * 3 * height_, 0)
                 sourceSize:MTLSizeMake(width_, 3 * height_, 1)
                  toTexture:to
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
      return finish(commands, "copie d'une image");
    }
  }

 private:
  // Le tenseur de la chaîne, planes × H rangées de W éléments FP16, vu comme une texture R16Float.
  id<MTLTexture> chain_texture(IOSurfaceRef surface, std::uint32_t planes, MTLTextureUsage usage) const {
    if (surface == nullptr || IOSurfaceGetWidth(surface) != width_ ||
        IOSurfaceGetHeight(surface) < static_cast<std::size_t>(planes) * height_ ||
        IOSurfaceGetBytesPerRow(surface) != static_cast<std::size_t>(width_) * 2)
      return nil;
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                                 width:width_
                                                                                height:planes * height_
                                                                             mipmapped:NO];
    d.usage = usage;
    d.storageMode = MTLStorageModeShared;
    return [device_ newTextureWithDescriptor:d iosurface:surface plane:0];
  }

  struct Textures {
    id<MTLTexture> first;
    id<MTLTexture> second;  // nil : un seul plan
  };

  std::optional<Textures> pixel_textures(CVPixelBufferRef buffer, MTLTextureUsage usage) const {
    IOSurfaceRef surface = buffer != nullptr ? CVPixelBufferGetIOSurface(buffer) : nullptr;
    if (surface == nullptr) return std::nullopt;
    const auto texture = [&](MTLPixelFormat format, NSUInteger width, NSUInteger height,
                             NSUInteger plane) -> id<MTLTexture> {
      MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                   width:width
                                                                                  height:height
                                                                               mipmapped:NO];
      d.usage = usage;
      d.storageMode = MTLStorageModeShared;
      return [device_ newTextureWithDescriptor:d iosurface:surface plane:plane];
    };
    Textures out{};
    if (planes_.second == MTLPixelFormatInvalid) {
      out.first = texture(planes_.first, width_, height_, 0);
      if (out.first == nil) return std::nullopt;
      return out;
    }
    if (IOSurfaceGetPlaneCount(surface) < 2) return std::nullopt;
    out.first = texture(planes_.first, width_, height_, 0);
    out.second = texture(planes_.second, width_ / 2, height_ / 2, 1);
    if (out.first == nil || out.second == nil) return std::nullopt;
    return out;
  }

  static void dispatch(id<MTLComputeCommandEncoder> compute, id<MTLComputePipelineState> pipeline,
                       std::uint32_t width, std::uint32_t height) {
    const NSUInteger simd = pipeline.threadExecutionWidth;
    const NSUInteger rows = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / simd / 4);
    [compute dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(simd, rows, 1)];
  }

  static std::expected<void, Refusal> finish(id<MTLCommandBuffer> commands, std::string_view what) {
    [commands commit];
    [commands waitUntilCompleted];
    if (commands.status != MTLCommandBufferStatusCompleted)
      return refuse_gpu(RefusalCode::DeviceUnavailable,
                        std::format("{} : {}", what, describe(commands.error)));
    return {};
  }

  vt::Pixels format_;
  std::uint32_t width_;
  std::uint32_t height_;
  PlaneFormats planes_;
  Yuv yuv_;
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> pack_;
  id<MTLComputePipelineState> unpack_;
  id<MTLComputePipelineState> differ_;
  std::array<id<MTLTexture>, 2> kept_;  // a et b de la dernière paire écrite, au GPU seul
  id<MTLBuffer> flags_;                 // les trois drapeaux de differs, partagés avec le CPU
  ByteHalves byte_halves_;              // la relecture d'un octet de BGRA
};

const char* pack_name(vt::Pixels format) {
  if (format == vt::Pixels::RgbaHalf) return "pack_rgba_half";
  if (format == vt::Pixels::Bgra8) return "pack_bgra8";
  return "pack_yuv";
}

const char* unpack_name(vt::Pixels format) {
  if (format == vt::Pixels::RgbaHalf) return "unpack_rgba_half";
  if (format == vt::Pixels::Bgra8) return "unpack_bgra8";
  return "unpack_yuv";
}

}  // namespace

std::expected<std::unique_ptr<VtGpu>, Refusal> VtGpu::open(vt::Pixels format, int width, int height) {
  @autoreleasepool {
    if (width <= 0 || height <= 0 || width > kMaxTextureSide || 7 * height > kMaxTextureSide)
      return refuse_gpu(RefusalCode::FeatureUnavailable,
                        std::format("{}x{} : l'entrée ne tient pas dans une texture", width, height));
    if (vt::is_yuv(format) && (width % 2 != 0 || height % 2 != 0))
      return refuse_gpu(RefusalCode::FeatureUnavailable, std::format("{}x{} impair en 4:2:0", width, height));
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return refuse_gpu(RefusalCode::FeatureUnavailable, "pas de GPU Metal");
    // dispatchThreads : des groupes non uniformes aux bords de l'image.
    if (![device supportsFamily:MTLGPUFamilyApple4] && ![device supportsFamily:MTLGPUFamilyMac2])
      return refuse_gpu(RefusalCode::FeatureUnavailable, "GPU sans groupes de fils non uniformes");
    const NSUInteger alignment = [device minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatR16Float];
    if (alignment == 0 || (static_cast<NSUInteger>(width) * 2) % alignment != 0)
      return refuse_gpu(
          RefusalCode::FeatureUnavailable,
          std::format("rangées de {} o, Metal les veut multiples de {} o", width * 2, alignment));
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_15_0
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
      options.fastMathEnabled = NO;
    }
#else
    options.fastMathEnabled = NO;
#endif
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(kSource) options:options error:&error];
    if (library == nil)
      return refuse_gpu(RefusalCode::FeatureUnavailable,
                        std::format("noyaux non compilés : {}", describe(error)));
    const auto pipeline = [&](const char* name) -> id<MTLComputePipelineState> {
      id<MTLFunction> function = [library newFunctionWithName:@(name)];
      if (function == nil) return nil;
      NSError* failure = nil;
      id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function
                                                                                error:&failure];
      if (state == nil) error = failure;
      return state;
    };
    id<MTLComputePipelineState> pack = pipeline(pack_name(format));
    id<MTLComputePipelineState> unpack = pipeline(unpack_name(format));
    id<MTLComputePipelineState> differ = pipeline("differ");
    if (pack == nil || unpack == nil || differ == nil)
      return refuse_gpu(RefusalCode::FeatureUnavailable, std::format("noyau refusé : {}", describe(error)));
    MTLTextureDescriptor* kept =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:static_cast<NSUInteger>(width)
                                                          height:3 * static_cast<NSUInteger>(height)
                                                       mipmapped:NO];
    kept.usage = MTLTextureUsageShaderRead;
    kept.storageMode = MTLStorageModePrivate;
    std::array<id<MTLTexture>, 2> kept_textures{[device newTextureWithDescriptor:kept],
                                                [device newTextureWithDescriptor:kept]};
    id<MTLBuffer> flags = [device newBufferWithLength:3 * sizeof(std::uint32_t)
                                              options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (kept_textures[0] == nil || kept_textures[1] == nil || flags == nil || queue == nil)
      return refuse_gpu(RefusalCode::GpuMemoryExhausted, "textures des images gardées refusées");
    return std::make_unique<MetalVtGpu>(format, width, height, device, queue, pack, unpack, differ,
                                        std::move(kept_textures), flags);
  }
}

}  // namespace ajp::gpu::coreml_rt
