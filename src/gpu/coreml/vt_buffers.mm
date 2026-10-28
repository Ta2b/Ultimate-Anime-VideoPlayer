#include "gpu/coreml/vt_buffers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>
#include <vector>

#include "gpu/coreml/coreml_direct.h"
#include "gpu/coreml/coreml_surfaces.h"
#include "gpu/coreml/coreml_vt.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::coreml_rt {

using vk::net::HostBinding;

std::unexpected<Refusal> refuse_vt(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::format("interpolation d'Apple : {}", text)});
}

std::expected<BufferPool, Refusal> make_pool(NSDictionary<NSString*, id>* wanted, vt::Pixels format,
                                             int width, int height) {
  NSMutableDictionary<NSString*, id>* attributes =
      wanted != nil ? [wanted mutableCopy] : [NSMutableDictionary dictionary];
  attributes[(__bridge NSString*)kCVPixelBufferPixelFormatTypeKey] = @(vt::four_cc(format));
  attributes[(__bridge NSString*)kCVPixelBufferWidthKey] = @(width);
  attributes[(__bridge NSString*)kCVPixelBufferHeightKey] = @(height);
  if (attributes[(__bridge NSString*)kCVPixelBufferIOSurfacePropertiesKey] == nil)
    attributes[(__bridge NSString*)kCVPixelBufferIOSurfacePropertiesKey] = @{};
  if (attributes[(__bridge NSString*)kCVPixelBufferMetalCompatibilityKey] == nil)
    attributes[(__bridge NSString*)kCVPixelBufferMetalCompatibilityKey] = @YES;
  CVPixelBufferPoolRef pool = nullptr;
  if (const CVReturn status =
          CVPixelBufferPoolCreate(kCFAllocatorDefault, nullptr, (__bridge CFDictionaryRef)attributes, &pool);
      status != kCVReturnSuccess || pool == nullptr) {
    return refuse_vt(RefusalCode::GpuMemoryExhausted, std::format("réserve de tampons refusée ({})", status));
  }
  return BufferPool(pool);
}

void tag_colors(CVPixelBufferRef buffer, bool yuv) {
  CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2,
                        kCVAttachmentMode_ShouldPropagate);
  CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2,
                        kCVAttachmentMode_ShouldPropagate);
  if (!yuv) return;
  CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2,
                        kCVAttachmentMode_ShouldPropagate);
  CVBufferSetAttachment(buffer, kCVImageBufferChromaLocationTopFieldKey, kCVImageBufferChromaLocation_Center,
                        kCVAttachmentMode_ShouldPropagate);
  CVBufferSetAttachment(buffer, kCVImageBufferChromaLocationBottomFieldKey,
                        kCVImageBufferChromaLocation_Center, kCVAttachmentMode_ShouldPropagate);
}

std::expected<void, Refusal> with_planes(CVPixelBufferRef buffer, bool read_only,
                                         const std::function<void(const vt::Planes&)>& use) {
  const CVPixelBufferLockFlags flags = read_only ? kCVPixelBufferLock_ReadOnly : 0;
  if (const CVReturn status = CVPixelBufferLockBaseAddress(buffer, flags); status != kCVReturnSuccess)
    return refuse_vt(RefusalCode::DeviceUnavailable, std::format("tampon non verrouillé ({})", status));
  vt::Planes planes;
  if (CVPixelBufferIsPlanar(buffer)) {
    const std::size_t count = std::min<std::size_t>(CVPixelBufferGetPlaneCount(buffer), planes.data.size());
    for (std::size_t p = 0; p < count; ++p) {
      planes.data[p] = static_cast<std::byte*>(CVPixelBufferGetBaseAddressOfPlane(buffer, p));
      planes.stride[p] = CVPixelBufferGetBytesPerRowOfPlane(buffer, p);
    }
  } else {
    planes.data[0] = static_cast<std::byte*>(CVPixelBufferGetBaseAddress(buffer));
    planes.stride[0] = CVPixelBufferGetBytesPerRow(buffer);
  }
  use(planes);
  CVPixelBufferUnlockBaseAddress(buffer, flags);
  return {};
}

void by_bands(int height, const std::function<void(vt::Rows)>& work) {
  constexpr int kBands = 16;
  const int band = (((height + kBands - 1) / kBands) + 1) & ~1;
  const int count = (height + band - 1) / band;
  const auto* job = &work;
  dispatch_apply(static_cast<std::size_t>(count), DISPATCH_APPLY_AUTO, ^(std::size_t i) {
    const int first = static_cast<int>(i) * band;
    (*job)(vt::Rows{.first = first, .last = std::min(first + band, height)});
  });
}

namespace {

// Les plans de deux tampons au même format égaux : à l'octet pour RGB (sans perte, ou arrondis par
// la même formule), à un code près en YCbCr (le GPU contracte autrement les produits).
bool same_planes(vt::Pixels format, int width, int height, const vt::Planes& a, const vt::Planes& b) {
  const auto w = static_cast<std::size_t>(width);
  const auto h = static_cast<std::size_t>(height);
  if (!vt::is_yuv(format)) {
    const std::size_t row = w * (format == vt::Pixels::RgbaHalf ? 8U : 4U);
    for (std::size_t y = 0; y < h; ++y) {
      if (std::memcmp(a.data[0] + (y * a.stride[0]), b.data[0] + (y * b.stride[0]), row) != 0) return false;
    }
    return true;
  }
  const vt::Levels l = vt::levels(format);
  const auto code = [&](const std::byte* row, std::size_t x) -> int {
    if (!l.wide) return std::to_integer<int>(row[x]);
    std::uint16_t word = 0;
    std::memcpy(&word, row + (x * 2), 2);
    return word >> l.shift;
  };
  const auto close = [&](std::size_t p, std::size_t rows, std::size_t codes) {
    for (std::size_t y = 0; y < rows; ++y) {
      const std::byte* ra = a.data[p] + (y * a.stride[p]);
      const std::byte* rb = b.data[p] + (y * b.stride[p]);
      for (std::size_t x = 0; x < codes; ++x) {
        if (std::abs(code(ra, x) - code(rb, x)) > 1) return false;
      }
    }
    return true;
  };
  return close(0, h, w) && close(1, h / 2, w);
}

std::expected<bool, Refusal> same_buffers(vt::Pixels format, int width, int height, CVPixelBufferRef a,
                                          CVPixelBufferRef b) {
  bool same = false;
  std::expected<void, Refusal> inner;
  auto outer = with_planes(a, true, [&](const vt::Planes& pa) {
    inner = with_planes(b, true,
                        [&](const vt::Planes& pb) { same = same_planes(format, width, height, pa, pb); });
  });
  if (!outer) return std::unexpected(outer.error());
  if (!inner) return std::unexpected(inner.error());
  return same;
}

// Les noyaux de gpu contre vt::pack et vt::unpack, sur une paire de test à la taille des images,
// dans des IOSurface comme celles de la chaîne : a et b écrites dans des tampons de sources et
// gardées, leur comparaison aux gardées, un tampon de destination écrit par le CPU puis relu, b
// copiée. Un écart, et le CPU garde les conversions. Une fois par format et par taille.
std::expected<void, Refusal> verify_gpu(VtGpu& gpu, vt::Pixels format, int width, int height,
                                        const BufferPool& sources, const BufferPool& destinations) {
  static std::mutex mutex;
  static std::set<std::tuple<vt::Pixels, int, int>> verified;  // sous mutex
  const std::scoped_lock lock(mutex);
  if (verified.contains({format, width, height})) return {};
  const auto mismatch = [](std::string_view what) {
    return refuse_vt(RefusalCode::DeviceUnavailable, std::format("GPU : {} différente du CPU", what));
  };
  const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const Shape in_shape{.dims = {1, 7, height, width}};
  const Shape out_shape{.dims = {1, 3, height, width}};
  const std::shared_ptr<Surfaces> surfaces = make_surfaces();
  auto in_memory = allocate_surface(surfaces, 7 * plane * 2, 64, TensorType::F16, in_shape);
  if (!in_memory) return std::unexpected(in_memory.error());
  auto out_memory = allocate_surface(surfaces, 3 * plane * 2, 64, TensorType::F16, out_shape);
  if (!out_memory) return std::unexpected(out_memory.error());
  std::byte* in_bytes = (*in_memory)->data();
  std::byte* out_bytes = (*out_memory)->data();
  // a et b : du bruit de -0,05 à 1,05 (les bornes des formats entiers servent), l'instant t.
  std::vector<std::uint16_t> values(7 * plane, kernels::ref::float_to_half(0.5F));
  for (std::size_t i = 0; i < 6 * plane; ++i) {
    const auto hash = static_cast<std::uint32_t>(i * 2654435761U) >> 8U;
    values[i] = kernels::ref::float_to_half(-0.05F + (1.1F * static_cast<float>(hash % 4096U) / 4095.0F));
  }
  std::memcpy(in_bytes, values.data(), values.size() * 2);
  const HostBinding input{
      .data = in_bytes, .bytes = 7 * plane * 2, .type = TensorType::F16, .shape = in_shape};
  const HostBinding output{
      .data = out_bytes, .bytes = 3 * plane * 2, .type = TensorType::F16, .shape = out_shape};

  const auto cpu_pack = [&](const std::byte* rgb, CVPixelBufferRef target) {
    return with_planes(target, false, [&](const vt::Planes& planes) {
      vt::pack(format, rgb, TensorType::F16, width, height, planes, vt::Rows{.first = 0, .last = height});
    });
  };
  auto a_cpu = sources.take();
  auto a_gpu = sources.take();
  auto b_gpu = sources.take();
  auto read = destinations.take();
  if (!a_cpu || !a_gpu || !b_gpu || !read)
    return refuse_vt(RefusalCode::GpuMemoryExhausted, "tampons de la vérification refusés");
  if (auto ok = cpu_pack(in_bytes, a_cpu->get()); !ok) return ok;
  if (auto ok = cpu_pack(in_bytes, read->get()); !ok) return ok;
  {
    std::optional<LentSurface> in = lend_surface(surfaces, input);
    std::optional<LentSurface> out = lend_surface(surfaces, output);
    if (!in || !out)
      return refuse_vt(RefusalCode::DeviceUnavailable, "IOSurface de la vérification non prêtées");
    const std::array<VtGpu::Pack, 2> packs{VtGpu::Pack{.image = 0, .target = a_gpu->get(), .keep = 0},
                                           VtGpu::Pack{.image = 1, .target = b_gpu->get(), .keep = 1}};
    if (auto ok = gpu.pack(in->get(), packs); !ok) return ok;
    auto differs = gpu.differs(in->get());
    if (!differs) return std::unexpected(differs.error());
    if ((*differs)[0] || (*differs)[1] || !(*differs)[2]) return mismatch("comparaison des images");
    if (auto ok = gpu.unpack(read->get(), out->get()); !ok) return ok;
  }
  auto same = same_buffers(format, width, height, a_cpu->get(), a_gpu->get());
  if (!same) return std::unexpected(same.error());
  if (!*same) return mismatch("écriture d'une image");
  // La relecture du tampon de destination, contre celle du CPU.
  std::vector<std::uint16_t> expected(3 * plane);
  if (auto ok = with_planes(read->get(), true,
                            [&](const vt::Planes& planes) {
                              vt::unpack(format, planes, reinterpret_cast<std::byte*>(expected.data()),
                                         TensorType::F16, width, height,
                                         vt::Rows{.first = 0, .last = height});
                            });
      !ok)
    return ok;
  std::vector<std::uint16_t> got(3 * plane);
  std::memcpy(got.data(), out_bytes, got.size() * 2);
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] == expected[i]) continue;
    const float g = kernels::ref::half_to_float(got[i]);
    const float e = kernels::ref::half_to_float(expected[i]);
    if (!vt::is_yuv(format) || !(std::abs(g - e) <= 2e-3F + (2e-3F * std::abs(e))))
      return mismatch(std::format("lecture d'un tampon ({} au lieu de {} à {})", g, e, i));
  }
  {
    std::optional<LentSurface> in = lend_surface(surfaces, input);
    std::optional<LentSurface> out = lend_surface(surfaces, output);
    if (!in || !out)
      return refuse_vt(RefusalCode::DeviceUnavailable, "IOSurface de la vérification non prêtées");
    if (auto ok = gpu.copy(in->get(), 1, out->get()); !ok) return ok;
  }
  if (std::memcmp(out_bytes, in_bytes + (3 * plane * 2), 3 * plane * 2) != 0)
    return mismatch("copie d'une image");
  verified.insert({format, width, height});
  return {};
}

}  // namespace

std::unique_ptr<VtGpu> open_gpu(vt::Pixels format, int width, int height, const BufferPool& sources,
                                const BufferPool& destinations, std::string& why) {
  auto gpu = VtGpu::open(format, width, height);
  if (!gpu) {
    why = gpu.error().detail;
    return nullptr;
  }
  if (auto ok = verify_gpu(**gpu, format, width, height, sources, destinations); !ok) {
    why = ok.error().detail;
    return nullptr;
  }
  return std::move(*gpu);
}

std::expected<void, Refusal> vt_gpu_check(vt::Pixels format, int width, int height) {
  @autoreleasepool {
    auto sources = make_pool(nil, format, width, height);
    if (!sources) return std::unexpected(sources.error());
    auto destinations = make_pool(nil, format, width, height);
    if (!destinations) return std::unexpected(destinations.error());
    auto gpu = VtGpu::open(format, width, height);
    if (!gpu) return std::unexpected(gpu.error());
    return verify_gpu(**gpu, format, width, height, *sources, *destinations);
  }
}

}  // namespace ajp::gpu::coreml_rt
