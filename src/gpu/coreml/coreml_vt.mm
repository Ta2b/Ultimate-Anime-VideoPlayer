#include "gpu/coreml/coreml_vt.h"

#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <VideoToolbox/VideoToolbox.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <mutex>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

// Le SDK de macOS 15.4 et plus déclare VTFrameProcessor ; avec un SDK plus ancien, l'interpolation
// d'Apple reste absente (refus FeatureUnavailable), comme hors d'un Mac.
#if defined(__MAC_15_4) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_15_4
#define AJP_VT_FRC 1
#if __has_include(<VideoToolbox/VTFrameProcessor.h>)
#import <VideoToolbox/VTFrameProcessor.h>
#endif
#if __has_include(<VideoToolbox/VTFrameProcessor_FrameRateConversion.h>)
#import <VideoToolbox/VTFrameProcessor_FrameRateConversion.h>
#endif
#else
#define AJP_VT_FRC 0
#endif

#include "gpu/coreml/coreml_steps.h"
#include "gpu/coreml/vt_pixels.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::coreml_rt {
namespace {

using vk::net::HostBinding;

std::unexpected<Refusal> refuse_vt(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::format("interpolation d'Apple : {}", text)});
}

#if AJP_VT_FRC

// Les instants des images, en millièmes d'image : l'image n de la vidéo est à n × kTicks.
constexpr std::int32_t kTicks = 1000;
// Une interpolation qui ne rend pas la main en 2 s est perdue : la lecture ne reste pas figée.
constexpr std::int64_t kTimeoutNs = 2'000'000'000;

std::string describe(NSError* error) {
  if (error == nil) return "erreur inconnue";
  const char* text = error.localizedDescription.UTF8String;
  return std::format("{} ({} {})", text != nullptr ? text : "?", error.domain.UTF8String, error.code);
}

// Une exception Objective-C (sélecteur inconnu, argument refusé) devient un refus.
template <class F>
auto guarded(std::string_view where, F&& body) -> decltype(body()) {
  @try {
    return body();
  } @catch (NSException* e) {
    const char* reason = e.reason.UTF8String;
    return refuse_vt(RefusalCode::DeviceUnavailable,
                     std::format("{} : {}", where, reason != nullptr ? reason : "exception"));
  }
}

class PixelBuffer {
 public:
  PixelBuffer() = default;
  explicit PixelBuffer(CVPixelBufferRef ref) : ref_(ref) {}
  PixelBuffer(const PixelBuffer&) = delete;
  PixelBuffer& operator=(const PixelBuffer&) = delete;
  PixelBuffer(PixelBuffer&& other) noexcept : ref_(std::exchange(other.ref_, nullptr)) {}
  PixelBuffer& operator=(PixelBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ref_ = std::exchange(other.ref_, nullptr);
    }
    return *this;
  }
  ~PixelBuffer() { reset(); }
  CVPixelBufferRef get() const { return ref_; }

 private:
  void reset() {
    if (ref_ != nullptr) CVPixelBufferRelease(ref_);
    ref_ = nullptr;
  }
  CVPixelBufferRef ref_ = nullptr;
};

class BufferPool {
 public:
  BufferPool() = default;
  explicit BufferPool(CVPixelBufferPoolRef ref) : ref_(ref) {}
  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;
  BufferPool(BufferPool&& other) noexcept : ref_(std::exchange(other.ref_, nullptr)) {}
  BufferPool& operator=(BufferPool&& other) noexcept {
    if (this != &other) {
      if (ref_ != nullptr) CVPixelBufferPoolRelease(ref_);
      ref_ = std::exchange(other.ref_, nullptr);
    }
    return *this;
  }
  ~BufferPool() {
    if (ref_ != nullptr) CVPixelBufferPoolRelease(ref_);
  }

  std::expected<PixelBuffer, Refusal> take() const {
    CVPixelBufferRef raw = nullptr;
    if (const CVReturn status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, ref_, &raw);
        status != kCVReturnSuccess || raw == nullptr) {
      return refuse_vt(RefusalCode::GpuMemoryExhausted, std::format("tampon de pixels refusé ({})", status));
    }
    return PixelBuffer(raw);
  }

 private:
  CVPixelBufferPoolRef ref_ = nullptr;
};

// Les tampons d'une sorte (source ou destination) : les exigences de l'interpolateur, au format
// et à la taille choisis.
std::expected<BufferPool, Refusal> make_pool(NSDictionary<NSString*, id>* wanted, vt::Pixels format,
                                             int width, int height) {
  NSMutableDictionary<NSString*, id>* attributes =
      wanted != nil ? [wanted mutableCopy] : [NSMutableDictionary dictionary];
  attributes[(__bridge NSString*)kCVPixelBufferPixelFormatTypeKey] = @(vt::four_cc(format));
  attributes[(__bridge NSString*)kCVPixelBufferWidthKey] = @(width);
  attributes[(__bridge NSString*)kCVPixelBufferHeightKey] = @(height);
  if (attributes[(__bridge NSString*)kCVPixelBufferIOSurfacePropertiesKey] == nil)
    attributes[(__bridge NSString*)kCVPixelBufferIOSurfacePropertiesKey] = @{};
  CVPixelBufferPoolRef pool = nullptr;
  if (const CVReturn status =
          CVPixelBufferPoolCreate(kCFAllocatorDefault, nullptr, (__bridge CFDictionaryRef)attributes, &pool);
      status != kCVReturnSuccess || pool == nullptr) {
    return refuse_vt(RefusalCode::GpuMemoryExhausted, std::format("réserve de tampons refusée ({})", status));
  }
  return BufferPool(pool);
}

// Les couleurs de la chaîne pour RIFE : BT.709 (kernels/spec/rife.h), la chroma au centre de chaque
// bloc 2 × 2 (vt_pixels.h).
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

// Les plans du tampon, verrouillés le temps de use.
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

// Les rangées en bandes de hauteur paire, sur les cœurs de la puce.
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

std::size_t element_size(TensorType type) {
  return type == TensorType::F32 ? 4U : 2U;
}

float element_at(const std::byte* data, TensorType type) {
  if (type == TensorType::F32) {
    float v = 0.0F;
    std::memcpy(&v, data, 4);
    return v;
  }
  std::uint16_t h = 0;
  std::memcpy(&h, data, 2);
  return kernels::ref::half_to_float(h);
}

class API_AVAILABLE(macos(15.4)) Interpolator {
 public:
  static std::expected<std::unique_ptr<Interpolator>, Refusal> open(int width, int height) {
    return guarded("ouverture", [&]() -> std::expected<std::unique_ptr<Interpolator>, Refusal> {
      @autoreleasepool {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        // processorSupported et frameSupportedPixelFormats : les noms du SDK 15, dépréciés au 26
        // (isSupported, supportedPixelFormats) mais toujours là.
        if (!VTFrameRateConversionConfiguration.processorSupported)
          return refuse_vt(RefusalCode::FeatureUnavailable, "cette puce ne la prend pas en charge");
        VTFrameRateConversionConfiguration* configuration = [[VTFrameRateConversionConfiguration alloc]
               initWithFrameWidth:width
                      frameHeight:height
               usePrecomputedFlow:NO
            qualityPrioritization:VTFrameRateConversionConfigurationQualityPrioritizationNormal
                         revision:VTFrameRateConversionConfiguration.defaultRevision];
        if (configuration == nil)
          return refuse_vt(RefusalCode::FeatureUnavailable,
                           std::format("images {}x{} refusées", width, height));
        std::vector<std::uint32_t> offered;
        for (NSNumber* code in configuration.frameSupportedPixelFormats)
          offered.push_back(code.unsignedIntValue);
#pragma clang diagnostic pop
        const std::optional<vt::Pixels> format = vt::preferred(offered);
        if (!format) {
          std::string listed;
          for (const std::uint32_t code : offered) listed += std::format(" {:08x}", code);
          return refuse_vt(RefusalCode::FormatUnsupported,
                           std::format("aucun format de pixels connu ({})", listed));
        }
        if (vt::is_yuv(*format) && (width % 2 != 0 || height % 2 != 0))
          return refuse_vt(RefusalCode::FormatUnsupported,
                           std::format("{}x{} impair en 4:2:0", width, height));
        auto source = make_pool(configuration.sourcePixelBufferAttributes, *format, width, height);
        if (!source) return std::unexpected(source.error());
        auto destination = make_pool(configuration.destinationPixelBufferAttributes, *format, width, height);
        if (!destination) return std::unexpected(destination.error());
        VTFrameProcessor* processor = [[VTFrameProcessor alloc] init];
        NSError* error = nil;
        if (![processor startSessionWithConfiguration:configuration error:&error])
          return refuse_vt(RefusalCode::FeatureUnavailable,
                           std::format("session refusée : {}", describe(error)));
        return std::unique_ptr<Interpolator>(
            new Interpolator(width, height, *format, processor, std::move(*source), std::move(*destination)));
      }
    });
  }

  Interpolator(const Interpolator&) = delete;
  Interpolator& operator=(const Interpolator&) = delete;
  Interpolator(Interpolator&&) = delete;
  Interpolator& operator=(Interpolator&&) = delete;
  ~Interpolator() {
    @try {
      [processor_ endSession];
    } @catch (NSException*) {
    }
  }

  std::string_view format_name() const { return vt::name(format_); }

  // in : 1 × 7 × height × width, out : 1 × 3 × height × width, vérifiés par l'appelant.
  std::expected<void, Refusal> interpolate(const HostBinding& in, const HostBinding& out) {
    const std::size_t plane =
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * element_size(in.type);
    const std::size_t image = 3 * plane;
    const std::byte* a = in.data;
    const std::byte* b = in.data + image;
    const float t = element_at(in.data + (6 * plane), in.type);
    // Aux bords de la paire, l'image même : ni conversion ni passage par VideoToolbox.
    if (!(t > 0.0F) || t >= 1.0F) {
      const std::size_t count = image / element_size(in.type);
      return vk::net::copy_converted(t >= 1.0F ? b : a, in.type, out.data, out.type, count);
    }
    return guarded("interpolation", [&]() -> std::expected<void, Refusal> {
      @autoreleasepool {
        const auto same_as = [&](const std::byte* p, const std::vector<std::byte>& kept) {
          return kept.size() == image && std::memcmp(p, kept.data(), image) == 0;
        };
        const bool repeats = known_ && same_as(a, source_rgb_) && same_as(b, next_rgb_);
        const bool follows = known_ && !repeats && same_as(a, next_rgb_);
        if (!repeats) {
          known_ = false;
          if (follows) {
            source_ = std::move(next_);
            std::swap(source_rgb_, next_rgb_);
            index_ += 1;
          } else {
            auto made = frame_of(a, in.type);
            if (!made) return std::unexpected(made.error());
            source_ = std::move(*made);
            source_rgb_.assign(a, a + image);
            index_ += 2;  // un saut : rien ne suit la paire d'avant
          }
          auto made = frame_of(b, in.type);
          if (!made) return std::unexpected(made.error());
          next_ = std::move(*made);
          next_rgb_.assign(b, b + image);
          known_ = true;
        }
        auto target = destinations_.take();
        if (!target) return std::unexpected(target.error());
        const std::int64_t at = index_ * kTicks;
        VTFrameProcessorFrame* source = [[VTFrameProcessorFrame alloc] initWithBuffer:source_.get()
                                                                presentationTimeStamp:CMTimeMake(at, kTicks)];
        VTFrameProcessorFrame* next =
            [[VTFrameProcessorFrame alloc] initWithBuffer:next_.get()
                                    presentationTimeStamp:CMTimeMake(at + kTicks, kTicks)];
        const auto between = at + static_cast<std::int64_t>(std::lround(t * static_cast<float>(kTicks)));
        VTFrameProcessorFrame* destination =
            [[VTFrameProcessorFrame alloc] initWithBuffer:target->get()
                                    presentationTimeStamp:CMTimeMake(between, kTicks)];
        if (source == nil || next == nil || destination == nil)
          return refuse_vt(RefusalCode::DeviceUnavailable, "image refusée par VideoToolbox");
        const VTFrameRateConversionParametersSubmissionMode mode =
            repeats || follows ? VTFrameRateConversionParametersSubmissionModeSequential
                               : VTFrameRateConversionParametersSubmissionModeRandom;
        VTFrameRateConversionParameters* parameters =
            [[VTFrameRateConversionParameters alloc] initWithSourceFrame:source
                                                               nextFrame:next
                                                             opticalFlow:nil
                                                      interpolationPhase:@[ @(t) ]
                                                          submissionMode:mode
                                                       destinationFrames:@[ destination ]];
        if (parameters == nil) return refuse_vt(RefusalCode::DeviceUnavailable, "paramètres refusés");
        __block NSError* failure = nil;
        dispatch_semaphore_t finished = dispatch_semaphore_create(0);
        [processor_ processWithParameters:parameters
                        completionHandler:^(id<VTFrameProcessorParameters> /*done*/, NSError* error) {
                          failure = error;
                          dispatch_semaphore_signal(finished);
                        }];
        if (dispatch_semaphore_wait(finished, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNs)) != 0) {
          known_ = false;
          return refuse_vt(RefusalCode::DeviceUnavailable, "sans réponse depuis 2 s");
        }
        if (failure != nil) {
          known_ = false;
          return refuse_vt(RefusalCode::DeviceUnavailable, describe(failure));
        }
        return with_planes(target->get(), true, [&](const vt::Planes& planes) {
          by_bands(height_, [&](vt::Rows rows) {
            vt::unpack(format_, planes, out.data, out.type, width_, height_, rows);
          });
        });
      }
    });
  }

 private:
  Interpolator(int width, int height, vt::Pixels format, VTFrameProcessor* processor, BufferPool sources,
               BufferPool destinations)
      : width_(width),
        height_(height),
        format_(format),
        processor_(processor),
        sources_(std::move(sources)),
        destinations_(std::move(destinations)) {}

  // Une image de la paire, dans un tampon neuf de la réserve : VideoToolbox peut garder les
  // précédents tant qu'il s'en sert.
  std::expected<PixelBuffer, Refusal> frame_of(const std::byte* rgb, TensorType type) {
    auto buffer = sources_.take();
    if (!buffer) return std::unexpected(buffer.error());
    tag_colors(buffer->get(), vt::is_yuv(format_));
    auto written = with_planes(buffer->get(), false, [&](const vt::Planes& planes) {
      by_bands(height_, [&](vt::Rows rows) { vt::pack(format_, rgb, type, width_, height_, planes, rows); });
    });
    if (!written) return std::unexpected(written.error());
    return std::move(*buffer);
  }

  int width_;
  int height_;
  vt::Pixels format_;
  VTFrameProcessor* processor_;
  BufferPool sources_;
  BufferPool destinations_;
  // La paire soumise en dernier : ses images converties, et ses plans tels que la chaîne les a
  // donnés, pour reconnaître la paire suivante ou la même.
  PixelBuffer source_;
  PixelBuffer next_;
  std::vector<std::byte> source_rgb_;
  std::vector<std::byte> next_rgb_;
  bool known_ = false;
  std::int64_t index_ = 0;  // l'image source_ de la vidéo, en images
};

class API_AVAILABLE(macos(15.4)) VtNet final : public vk::net::RuntimeNet {
 public:
  VtNet(vk::net::IoNames names, std::unique_ptr<Interpolator> interpolator, int width, int height)
      : names_(std::move(names)), interpolator_(std::move(interpolator)), width_(width), height_(height) {}
  VtNet(const VtNet&) = delete;
  VtNet& operator=(const VtNet&) = delete;
  VtNet(VtNet&&) = delete;
  VtNet& operator=(VtNet&&) = delete;
  ~VtNet() override = default;

  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }

  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    if (auto ok = check(inputs, outputs); !ok) return ok;
    const std::scoped_lock lock(mutex_);
    return interpolator_->interpolate(inputs[0], outputs[0]);
  }

  // Sur le fil du réseau : le bloc média calcule pendant que le GPU fait le reste de la chaîne.
  std::expected<void, Refusal> run_deferred(std::span<const HostBinding> inputs,
                                            std::span<const HostBinding> outputs,
                                            std::function<std::expected<void, Refusal>()> ready,
                                            std::function<void(std::expected<void, Refusal>)> done) override {
    if (auto ok = check(inputs, outputs); !ok) return ok;
    return steps_.push(inputs, outputs, std::move(ready), std::move(done));
  }

 private:
  std::expected<void, Refusal> check(std::span<const HostBinding> inputs,
                                     std::span<const HostBinding> outputs) const {
    if (inputs.size() != 1 || outputs.size() != 1)
      return refuse_vt(RefusalCode::InputMismatch, "une entrée et une sortie attendues");
    const Shape in{.dims = {1, 7, height_, width_}};
    const Shape out{.dims = {1, 3, height_, width_}};
    const auto floating = [](TensorType type) { return type == TensorType::F16 || type == TensorType::F32; };
    if (inputs[0].shape != in || outputs[0].shape != out || inputs[0].data == nullptr ||
        outputs[0].data == nullptr || !floating(inputs[0].type) || !floating(outputs[0].type)) {
      return refuse_vt(
          RefusalCode::InputMismatch,
          std::format("tenseurs attendus 1x7x{0}x{1} et 1x3x{0}x{1}, en mémoire hôte", height_, width_));
    }
    return {};
  }

  vk::net::IoNames names_;
  std::unique_ptr<Interpolator> interpolator_;
  int width_;
  int height_;
  std::mutex mutex_;  // interpolator_ : le fil des étapes ou celui de run
  // Dernier membre : arrêté le premier, ses étapes en file faites sur un interpolateur encore là.
  DeferredSteps steps_{"-vt",
                       [this](std::span<const HostBinding> inputs, std::span<const HostBinding> outputs) {
                         const std::scoped_lock lock(mutex_);
                         return interpolator_->interpolate(inputs[0], outputs[0]);
                       }};
};

#endif  // AJP_VT_FRC

}  // namespace

std::expected<std::string, Refusal> vt_probe(int width, int height) {
#if AJP_VT_FRC
  if (@available(macOS 15.4, *)) {
    auto opened = Interpolator::open(width, height);
    if (!opened) return std::unexpected(opened.error());
    // Une paire unie de la même couleur : l'instant du milieu doit la rendre, quel que soit le flot.
    // Un ordre de canaux ou une matrice faux se voient ici, avant la lecture.
    const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    constexpr std::array<float, 3> kColor{0.8F, 0.4F, 0.2F};
    std::vector<std::uint16_t> in(7 * plane);
    std::vector<std::uint16_t> out(3 * plane);
    for (std::size_t c = 0; c < 7; ++c) {
      const float value = c == 6 ? 0.5F : kColor[c % 3];
      std::fill_n(in.begin() + static_cast<std::ptrdiff_t>(c * plane), plane,
                  kernels::ref::float_to_half(value));
    }
    const HostBinding input{.data = reinterpret_cast<std::byte*>(in.data()),
                            .bytes = in.size() * 2,
                            .type = TensorType::F16,
                            .shape = {.dims = {1, 7, height, width}}};
    const HostBinding output{.data = reinterpret_cast<std::byte*>(out.data()),
                             .bytes = out.size() * 2,
                             .type = TensorType::F16,
                             .shape = {.dims = {1, 3, height, width}}};
    if (auto ok = (*opened)->interpolate(input, output); !ok) return std::unexpected(ok.error());
    const std::size_t middle = ((plane / 2) + (static_cast<std::size_t>(width) / 2)) % plane;
    std::array<float, 3> got{};
    for (std::size_t c = 0; c < 3; ++c) got[c] = kernels::ref::half_to_float(out[(c * plane) + middle]);
    for (std::size_t c = 0; c < 3; ++c) {
      if (!(std::abs(got[c] - kColor[c]) <= 0.06F)) {
        return refuse_vt(RefusalCode::DeviceUnavailable,
                         std::format("couleur rendue {:.3f} {:.3f} {:.3f} au lieu de {:.3f} {:.3f} {:.3f}",
                                     got[0], got[1], got[2], kColor[0], kColor[1], kColor[2]));
      }
    }
    return std::format("{}x{}, {}", width, height, (*opened)->format_name());
  }
#endif
  return refuse_vt(RefusalCode::FeatureUnavailable, "macOS 15.4 ou plus requis");
}

std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> vt_net(vk::net::IoNames names, int width,
                                                                    int height) {
#if AJP_VT_FRC
  if (@available(macOS 15.4, *)) {
    auto opened = Interpolator::open(width, height);
    if (!opened) return std::unexpected(opened.error());
    return std::make_unique<VtNet>(std::move(names), std::move(*opened), width, height);
  }
#endif
  return refuse_vt(RefusalCode::FeatureUnavailable, "macOS 15.4 ou plus requis");
}

}  // namespace ajp::gpu::coreml_rt
