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
#include <optional>
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

#include "core/log.h"
#include "gpu/coreml/coreml_direct.h"
#include "gpu/coreml/coreml_steps.h"
#include "gpu/coreml/coreml_surfaces.h"
#include "gpu/coreml/vt_buffers.h"
#include "gpu/coreml/vt_gpu.h"
#include "gpu/coreml/vt_pixels.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::coreml_rt {
namespace {

using vk::net::HostBinding;

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
  // surfaces : les IOSurface de la chaîne, que les conversions prennent sur le GPU ; nul : le CPU.
  static std::expected<std::unique_ptr<Interpolator>, Refusal> open(int width, int height,
                                                                    std::shared_ptr<Surfaces> surfaces) {
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
        std::string why = "IOSurface de la chaîne absentes";
        std::unique_ptr<VtGpu> gpu =
            surfaces != nullptr ? open_gpu(*format, width, height, *source, *destination, why) : nullptr;
        if (gpu == nullptr)
          log_channel("coreml")->info("interpolation d'Apple : conversions sur le CPU ({})", why);
        return std::unique_ptr<Interpolator>(
            new Interpolator(width, height, *format, processor, std::move(*source), std::move(*destination),
                             std::move(surfaces), std::move(gpu), std::move(why)));
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
  // Les conversions sur le GPU ; sinon pourquoi elles restent sur le CPU.
  bool on_gpu() const { return gpu_ != nullptr; }
  const std::string& cpu_reason() const { return cpu_reason_; }

  // in : 1 × 7 × height × width, out : 1 × 3 × height × width, vérifiés par l'appelant. Sur le GPU
  // quand les deux sont des IOSurface de la chaîne ; sinon, ou si Metal échoue (le GPU est alors
  // laissé), par les cœurs du CPU.
  std::expected<void, Refusal> interpolate(const HostBinding& in, const HostBinding& out) {
    const std::size_t plane =
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * element_size(in.type);
    const float t = element_at(in.data + (6 * plane), in.type);
    return guarded("interpolation", [&]() -> std::expected<void, Refusal> {
      @autoreleasepool {
        if (gpu_ != nullptr) {
          std::optional<LentSurface> input = lend_surface(surfaces_, in);
          std::optional<LentSurface> output = input ? lend_surface(surfaces_, out) : std::nullopt;
          if (input && output) {
            bool converting = false;
            auto done = by_gpu(input->get(), output->get(), t, converting);
            if (done || !converting) return done;
            log_channel("coreml")->warn("interpolation d'Apple : {} ; conversions sur le CPU",
                                        done.error().detail);
            cpu_reason_ = done.error().detail;
            gpu_.reset();
            known_ = false;
          }
        }
        return by_cpu(in, out, t);
      }
    });
  }

 private:
  // La paire par le GPU : comparée aux images gardées, écrite dans les tampons sources si elle est
  // nouvelle, l'image rendue relue dans output. converting : l'échec vient de Metal.
  std::expected<void, Refusal> by_gpu(IOSurfaceRef input, IOSurfaceRef output, float t, bool& converting) {
    // Aux bords de la paire, l'image même : ni conversion ni passage par VideoToolbox.
    if (!(t > 0.0F) || t >= 1.0F) {
      auto copied = gpu_->copy(input, t >= 1.0F ? 1 : 0, output);
      converting = !copied;
      return copied;
    }
    bool repeats = false;
    bool follows = false;
    if (known_ && kept_on_gpu_) {
      auto differs = gpu_->differs(input);
      if (!differs) {
        converting = true;
        return std::unexpected(differs.error());
      }
      repeats = !(*differs)[0] && !(*differs)[1];
      follows = !repeats && !(*differs)[2];
    }
    if (!repeats) {
      known_ = false;
      std::array<VtGpu::Pack, 2> packs{};
      std::size_t count = 0;
      PixelBuffer first;
      if (!follows) {
        auto taken = sources_.take();
        if (!taken) return std::unexpected(taken.error());
        first = std::move(*taken);
        tag_colors(first.get(), vt::is_yuv(format_));
        packs[count++] = VtGpu::Pack{.image = 0, .target = first.get(), .keep = 0};
      }
      auto second = sources_.take();
      if (!second) return std::unexpected(second.error());
      tag_colors(second->get(), vt::is_yuv(format_));
      packs[count++] = VtGpu::Pack{.image = 1, .target = second->get(), .keep = 1};
      // L'image a est la b d'avant : sa gardée passe en tête, b prend l'autre place.
      if (follows) gpu_->swap_kept();
      if (auto packed = gpu_->pack(input, std::span<const VtGpu::Pack>(packs.data(), count)); !packed) {
        converting = true;
        return packed;
      }
      if (follows) {
        source_ = std::move(next_);
        index_ += 1;
      } else {
        source_ = std::move(first);
        index_ += 2;  // un saut : rien ne suit la paire d'avant
      }
      next_ = std::move(*second);
      std::vector<std::byte>().swap(source_rgb_);
      std::vector<std::byte>().swap(next_rgb_);
      known_ = true;
      kept_on_gpu_ = true;
    }
    auto target = destinations_.take();
    if (!target) return std::unexpected(target.error());
    if (auto ok = submit(target->get(), t, repeats || follows); !ok) return ok;
    auto read = gpu_->unpack(target->get(), output);
    converting = !read;
    return read;
  }

  // La paire par les cœurs du CPU : comparée à la précédente, convertie dans les tampons sources
  // si elle est nouvelle, l'image rendue relue dans out.
  std::expected<void, Refusal> by_cpu(const HostBinding& in, const HostBinding& out, float t) {
    const std::size_t plane =
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * element_size(in.type);
    const std::size_t image = 3 * plane;
    const std::byte* a = in.data;
    const std::byte* b = in.data + image;
    // Aux bords de la paire, l'image même : ni conversion ni passage par VideoToolbox.
    if (!(t > 0.0F) || t >= 1.0F) {
      const std::size_t count = image / element_size(in.type);
      return vk::net::copy_converted(t >= 1.0F ? b : a, in.type, out.data, out.type, count);
    }
    const auto same_as = [&](const std::byte* p, const std::vector<std::byte>& kept) {
      return kept.size() == image && std::memcmp(p, kept.data(), image) == 0;
    };
    // Des images gardées par le GPU ne se comparent pas ici : la paire est reprise.
    const bool comparable = known_ && !kept_on_gpu_;
    const bool repeats = comparable && same_as(a, source_rgb_) && same_as(b, next_rgb_);
    const bool follows = comparable && !repeats && same_as(a, next_rgb_);
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
      kept_on_gpu_ = false;
    }
    auto target = destinations_.take();
    if (!target) return std::unexpected(target.error());
    if (auto ok = submit(target->get(), t, repeats || follows); !ok) return ok;
    return with_planes(target->get(), true, [&](const vt::Planes& planes) {
      by_bands(height_, [&](vt::Rows rows) {
        vt::unpack(format_, planes, out.data, out.type, width_, height_, rows);
      });
    });
  }

  // L'image à l'instant t entre source_ et next_, écrite par VideoToolbox dans target. sequential :
  // la paire suit ou répète la précédente, VideoToolbox reprend ce qu'il en a calculé.
  std::expected<void, Refusal> submit(CVPixelBufferRef target, float t, bool sequential) {
    const std::int64_t at = index_ * kTicks;
    VTFrameProcessorFrame* source = [[VTFrameProcessorFrame alloc] initWithBuffer:source_.get()
                                                            presentationTimeStamp:CMTimeMake(at, kTicks)];
    VTFrameProcessorFrame* next =
        [[VTFrameProcessorFrame alloc] initWithBuffer:next_.get()
                                presentationTimeStamp:CMTimeMake(at + kTicks, kTicks)];
    const auto between = at + static_cast<std::int64_t>(std::lround(t * static_cast<float>(kTicks)));
    VTFrameProcessorFrame* destination =
        [[VTFrameProcessorFrame alloc] initWithBuffer:target
                                presentationTimeStamp:CMTimeMake(between, kTicks)];
    if (source == nil || next == nil || destination == nil)
      return refuse_vt(RefusalCode::DeviceUnavailable, "image refusée par VideoToolbox");
    const VTFrameRateConversionParametersSubmissionMode mode =
        sequential ? VTFrameRateConversionParametersSubmissionModeSequential
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
    return {};
  }

  Interpolator(int width, int height, vt::Pixels format, VTFrameProcessor* processor, BufferPool sources,
               BufferPool destinations, std::shared_ptr<Surfaces> surfaces, std::unique_ptr<VtGpu> gpu,
               std::string cpu_reason)
      : width_(width),
        height_(height),
        format_(format),
        processor_(processor),
        sources_(std::move(sources)),
        destinations_(std::move(destinations)),
        surfaces_(std::move(surfaces)),
        gpu_(std::move(gpu)),
        cpu_reason_(std::move(cpu_reason)) {}

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
  std::shared_ptr<Surfaces> surfaces_;  // nul : conversions sur le CPU
  std::unique_ptr<VtGpu> gpu_;          // nul : conversions sur le CPU, pour cpu_reason_
  std::string cpu_reason_;
  // La paire soumise en dernier : ses images converties, et ses plans tels que la chaîne les a
  // donnés, pour reconnaître la paire suivante ou la même ; gardés par le GPU (kept_on_gpu_) ou ici.
  PixelBuffer source_;
  PixelBuffer next_;
  std::vector<std::byte> source_rgb_;
  std::vector<std::byte> next_rgb_;
  bool known_ = false;
  bool kept_on_gpu_ = false;
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
    // Les tenseurs dans des IOSurface comme ceux de la chaîne : les conversions passent par le
    // même chemin qu'à la lecture.
    const std::shared_ptr<Surfaces> surfaces = make_surfaces();
    auto opened = Interpolator::open(width, height, surfaces);
    if (!opened) return std::unexpected(opened.error());
    // Une paire unie de la même couleur : l'instant du milieu doit la rendre, quel que soit le flot.
    // Un ordre de canaux ou une matrice faux se voient ici, avant la lecture.
    const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const Shape in_shape{.dims = {1, 7, height, width}};
    const Shape out_shape{.dims = {1, 3, height, width}};
    auto in_memory = allocate_surface(surfaces, 7 * plane * 2, 64, TensorType::F16, in_shape);
    if (!in_memory) return std::unexpected(in_memory.error());
    auto out_memory = allocate_surface(surfaces, 3 * plane * 2, 64, TensorType::F16, out_shape);
    if (!out_memory) return std::unexpected(out_memory.error());
    constexpr std::array<float, 3> kColor{0.8F, 0.4F, 0.2F};
    std::vector<std::uint16_t> in(7 * plane);
    for (std::size_t c = 0; c < 7; ++c) {
      const float value = c == 6 ? 0.5F : kColor[c % 3];
      std::fill_n(in.begin() + static_cast<std::ptrdiff_t>(c * plane), plane,
                  kernels::ref::float_to_half(value));
    }
    std::memcpy((*in_memory)->data(), in.data(), in.size() * 2);
    const HostBinding input{
        .data = (*in_memory)->data(), .bytes = 7 * plane * 2, .type = TensorType::F16, .shape = in_shape};
    const HostBinding output{
        .data = (*out_memory)->data(), .bytes = 3 * plane * 2, .type = TensorType::F16, .shape = out_shape};
    if (auto ok = (*opened)->interpolate(input, output); !ok) return std::unexpected(ok.error());
    std::vector<std::uint16_t> out(3 * plane);
    std::memcpy(out.data(), (*out_memory)->data(), out.size() * 2);
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
    const std::string where = (*opened)->on_gpu()
                                  ? std::string("conversions sur le GPU")
                                  : std::format("conversions sur le CPU : {}", (*opened)->cpu_reason());
    return std::format("{}x{}, {}, {}", width, height, (*opened)->format_name(), where);
  }
#endif
  return refuse_vt(RefusalCode::FeatureUnavailable, "macOS 15.4 ou plus requis");
}

std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> vt_net(vk::net::IoNames names, int width,
                                                                    int height,
                                                                    std::shared_ptr<Surfaces> surfaces) {
#if AJP_VT_FRC
  if (@available(macOS 15.4, *)) {
    auto opened = Interpolator::open(width, height, std::move(surfaces));
    if (!opened) return std::unexpected(opened.error());
    return std::make_unique<VtNet>(std::move(names), std::move(*opened), width, height);
  }
#endif
  return refuse_vt(RefusalCode::FeatureUnavailable, "macOS 15.4 ou plus requis");
}

}  // namespace ajp::gpu::coreml_rt
