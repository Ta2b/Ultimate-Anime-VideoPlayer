// La dérivation native du paquet « .span8 » sous Vulkan (Backend::derive_span_fp8, docs/span-fp8-cutlass.md)
// : ce que tensorrt_span_derive.cpp fait par les noyaux CUTLASS, ici par les réseaux du moteur Vulkan, sur le
// graphe du paquet (net_span8.h). Pour une carte à matrices coopératives FP8 (RDNA 4, Target::matrix_f8).
//
//   1. La référence FP16 (le graphe sans arrondi E4M3) passe les images de calibration et relève le maximum
//      de |valeur| des 1 + 3·blocs tenseurs que le graphe FP8 arrondit ; échelle = maximum × 1,25 / 448
//      (span_fp8::scales_from_peaks, comme le backend TensorRT). Par tuiles de kTile pixels, chacune ne
//      comptant que sa partie loin de ses bords intérieurs (halo) : là, elle calcule ce que l'image entière
//      calcule, et la mémoire reste bornée (les tenseurs relevés sont des sorties du graphe).
//   2. Le graphe FP8 à ces échelles, construit comme la lecture le construira (autotuner, porte de bout en
//      bout), passe les images de validation, mises de côté, et les cas durs ; sa sortie est jugée contre
//      celle de la référence FP16 (gpu/span_fp8/metrics.h). Les maxima se relèvent aussi sur la validation
//      (span_fp8::judge_peaks, C02-04).
//   3. Le temps du graphe FP8 à la taille de travail, contre celui du SPAN FP16 de ce moteur que la requête
//      apporte (SpanFp8Request::fp16), par tours intercalés ; le lecteur exige le gain (B02-01).
//
// Pas de variante à accumulateur f16 (scores_acc16 vides) : les matrices FP8 de RDNA 4 somment en float.
// Rien de tout cela n'a encore tourné sur une carte RDNA 4 : les tests le jouent sous lavapipe, où le noyau
// direct calcule le même arrondi E4M3 en float.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "gpu/span_fp8/derive.h"
#include "gpu/span_fp8/metrics.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/backend/net_measure.h"
#include "gpu/vulkan/backend/net_span8.h"
#include "gpu/vulkan/backend/vulkan_backend.h"
#include "gpu/vulkan/backend/vulkan_backend_detail.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/memory.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::vk {
namespace {

using detail::refuse_backend;

constexpr int kTile = 512;        // côté des tuiles de la calibration, en pixels d'entrée (pair)
constexpr int kTimingRounds = 5;  // tours intercalés FP8, FP16
constexpr int kTimingRuns = 8;    // passages par tour

// Les origines des tuiles de tile points sur extent (en points du demi-plan de travail), et la partie de
// chacune qui compte : à halo points de ses bords intérieurs, rien ; au bord de l'image, tout.
struct Span1d {
  int origin = 0;
  int first = 0;  // dans la tuile
  int last = 0;   // exclu
};

std::vector<Span1d> tiles_1d(int extent, int tile, int halo) {
  if (tile >= extent) return {{.origin = 0, .first = 0, .last = extent}};
  const int stride = std::max(1, tile - 2 * halo);
  std::vector<Span1d> out;
  for (int origin = 0;; origin += stride) {
    const int at = std::min(origin, extent - tile);
    out.push_back(
        {.origin = at, .first = at == 0 ? 0 : halo, .last = at + tile == extent ? tile : tile - halo});
    if (at + tile >= extent) break;
  }
  return out;
}

// Une découpe d'une image 8 bits planaire en FP16 NCHW (valeurs de 0 à 1).
std::vector<std::byte> still_half(const SpanStill& still, int x0, int y0, int width, int height) {
  std::vector<std::byte> out(std::size_t{3} * static_cast<std::size_t>(width) *
                             static_cast<std::size_t>(height) * 2);
  std::size_t at = 0;
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const std::size_t from = (static_cast<std::size_t>(c) * static_cast<std::size_t>(still.height) +
                                  static_cast<std::size_t>(y0 + y)) *
                                     static_cast<std::size_t>(still.width) +
                                 static_cast<std::size_t>(x0 + x);
        const std::uint16_t h = kernels::ref::float_to_half(static_cast<float>(still.rgb[from]) / 255.0F);
        std::memcpy(out.data() + at, &h, 2);
        at += 2;
      }
  return out;
}

// La sortie FP16 d'un passage, rapportée à [0, 1].
std::vector<float> half_unit(std::span<const std::byte> bytes) {
  std::vector<float> out(bytes.size() / 2);
  for (std::size_t i = 0; i < out.size(); ++i) {
    std::uint16_t h = 0;
    std::memcpy(&h, bytes.data() + 2 * i, 2);
    out[i] = std::clamp(kernels::ref::half_to_float(h), 0.0F, 1.0F);
  }
  return out;
}

span_fp8::ImageScore score(const std::vector<float>& expected, const std::vector<float>& got, int width,
                           int height) {
  return {.psnr_db = span_fp8::psnr(expected, got),
          .ssim = span_fp8::ssim(width, height, expected, got),
          .error_p999 = span_fp8::abs_error_quantile(expected, got, span_fp8::kErrorQuantile),
          .dark_error_p999 =
              span_fp8::dark_abs_error_quantile(width, height, expected, got, span_fp8::kErrorQuantile)};
}

SpanFp8ImageScore backend_score(const span_fp8::ImageScore& s) {
  return {
      .psnr_db = s.psnr_db, .ssim = s.ssim, .error_p999 = s.error_p999, .dark_error_p999 = s.dark_error_p999};
}

// Les tailles et le paquet d'une requête vérifiée.
struct Span8Plan {
  span_fp8::Package package;
  int width = 0;
  int height = 0;
  int timing_width = 0;
  int timing_height = 0;
};

std::expected<Span8Plan, Refusal> check_request(const SpanFp8Request& request) {
  if ((request.calibration.empty() && !request.validate_only) || request.validation.empty())
    return refuse_backend(RefusalCode::ShapeInvalid,
                          "dérivation du SPAN FP8 : images de calibration et de validation attendues");
  Span8Plan plan;
  plan.width = request.validation.front().width;
  plan.height = request.validation.front().height;
  plan.timing_width = request.timing_width > 0 ? request.timing_width : plan.width;
  plan.timing_height = request.timing_height > 0 ? request.timing_height : plan.height;
  if (const std::string why = span_fp8::size_refusal(plan.height, plan.width); !why.empty())
    return refuse_backend(RefusalCode::FormatUnsupported, "dérivation du SPAN FP8 : " + why);
  if (const std::string why = span_fp8::size_refusal(plan.timing_height, plan.timing_width); !why.empty())
    return refuse_backend(RefusalCode::FormatUnsupported, "temps du SPAN FP8 : " + why);
  const std::size_t plane = static_cast<std::size_t>(plan.height) * static_cast<std::size_t>(plan.width);
  for (const auto* group : {&request.calibration, &request.validation, &request.hard})
    for (const SpanStill& still : *group)
      if (still.width != plan.width || still.height != plan.height || still.rgb.size() != 3 * plane)
        return refuse_backend(RefusalCode::ShapeInvalid,
                              "dérivation du SPAN FP8 : images de tailles différentes");
  auto parsed = span_fp8::parse(request.package);
  if (!parsed) return std::unexpected(std::move(parsed.error()));
  if (parsed->fp16)
    return refuse_backend(RefusalCode::FormatUnsupported,
                          "dérivation du SPAN FP8 : paquet FP16, rien à quantifier");
  plan.package = std::move(*parsed);
  return plan;
}

// Étape 3.6 : même estimation que le backend TensorRT (zones des réseaux FP8 et FP16, entrée et sorties,
// poids), contre la VRAM libre.
std::expected<void, Refusal> check_memory(const VkContext& context, const Span8Plan& plan,
                                          double min_free_ratio) {
  if (min_free_ratio <= 0.0) return {};
  const std::optional<HeapUse> heap = device_heap_use(context);
  if (!heap) return {};
  const double plane = static_cast<double>(plan.width) * static_cast<double>(plan.height);
  const double timed = static_cast<double>(plan.timing_width) * static_cast<double>(plan.timing_height);
  const double need = std::max(plane / 4.0 * (768.0 + 1120.0) + plane * 3.0 * 2.0 * 9.0,
                               timed / 4.0 * 768.0 + timed * 3.0 * 2.0 * 9.0) +
                      8.0e6;
  const auto free_bytes = static_cast<double>(heap->total - std::min(heap->used, heap->total));
  if (free_bytes >= min_free_ratio * need) return {};
  return refuse_backend(
      RefusalCode::GpuMemoryExhausted,
      std::format("dérivation du SPAN FP8 : {:.0f} Mo de VRAM libres, {:.0f} Mo demandés ({} × le besoin)",
                  free_bytes / 1048576.0, min_free_ratio * need / 1048576.0, min_free_ratio));
}

std::unexpected<Refusal> stopped() {
  return refuse_backend(RefusalCode::Cancelled, "dérivation du SPAN FP8 interrompue");
}

template <typename T>
std::unexpected<Refusal> failed(const std::expected<T, Refusal>& result) {
  return std::unexpected(result.error());
}

}  // namespace

// Une dérivation en cours : ses moteurs, ses tenseurs, ses tampons de transfert, et chacune de ses étapes.
struct Span8Run {
  VulkanBackend& backend;
  const SpanFp8Request& request;
  Progress& progress;
  Span8Plan plan;
  std::size_t blocks = 0;
  int tile_w = 0;
  int tile_h = 0;
  std::vector<Span1d> rows;
  std::vector<Span1d> cols;
  std::unique_ptr<Engine> peaks_net;  // la référence FP16 d'une tuile, ses tenseurs relevés en sorties
  std::unique_ptr<Engine> reference;  // la référence FP16 de l'image
  std::unique_ptr<Engine> fp8;        // le graphe FP8 de l'image, tel que la lecture le construira
  std::unique_ptr<TensorBuffer> tile_in, tile_out, image_in, ref_out, fp8_out;
  std::vector<std::unique_ptr<TensorBuffer>> taps;
  std::vector<Tensor> tap_tensors;
  std::unique_ptr<Buffer> upload, readback, peak_words;  // posés par make_tensors
  float done = 0.0F;                                     // images passées de l'étape en cours

  void report(float fraction, const char* stage) const {
    if (progress.report) progress.report(fraction, stage);
  }
  std::expected<std::unique_ptr<TensorBuffer>, Refusal> tensor(int c, int h, int w) {
    return backend.alloc_tensor(TensorType::F16,
                                Shape{.dims = {1, std::int64_t{c}, std::int64_t{h}, std::int64_t{w}}});
  }

  std::expected<void, Refusal> plan_tiles(int tile) {
    blocks = plan.package.blocks.size();
    tile_w = std::min(tile, plan.width);
    tile_h = std::min(tile, plan.height);
    // Le tenseur relevé le plus profond (sortie du dernier bloc) voit 1 + 3·blocs points du demi-plan
    // autour de lui : un de plus par prudence.
    const int halo = 2 + 3 * static_cast<int>(blocks);
    rows = tiles_1d(plan.height / 2, tile_h / 2, halo);
    cols = tiles_1d(plan.width / 2, tile_w / 2, halo);
    if ((rows.size() > 1 && tile_h / 2 <= 2 * halo) || (cols.size() > 1 && tile_w / 2 <= 2 * halo))
      return refuse_backend(
          RefusalCode::ShapeInvalid,
          std::format("dérivation du SPAN FP8 : tuiles de {} pixels trop petites pour {} blocs", tile,
                      blocks));
    return {};
  }

  std::expected<void, Refusal> make_references() {
    auto tiles = backend.span8_engine(plan.package, {.quantize = false, .peaks = true}, tile_w, tile_h, false,
                                      progress);
    if (!tiles) return failed(tiles);
    peaks_net = std::move(*tiles);
    auto whole = backend.span8_engine(plan.package, {.quantize = false, .peaks = false}, plan.width,
                                      plan.height, false, progress);
    if (!whole) return failed(whole);
    reference = std::move(*whole);
    return {};
  }

  std::expected<void, Refusal> make_tensors() {
    const int channels = plan.package.real_channels;
    auto a = tensor(3, tile_h, tile_w);
    auto b = tensor(3, 2 * tile_h, 2 * tile_w);
    auto c = tensor(3, plan.height, plan.width);
    auto d = tensor(3, 2 * plan.height, 2 * plan.width);
    auto e = tensor(3, 2 * plan.height, 2 * plan.width);
    for (auto* t : {&a, &b, &c, &d, &e})
      if (!*t) return failed(*t);
    tile_in = std::move(*a);
    tile_out = std::move(*b);
    image_in = std::move(*c);
    ref_out = std::move(*d);
    fp8_out = std::move(*e);
    for (std::size_t k = 0; k < span_fp8::scale_count(blocks); ++k) {
      auto t = tensor(channels, tile_h / 2, tile_w / 2);
      if (!t) return failed(t);
      tap_tensors.push_back((*t)->tensor());
      taps.push_back(std::move(*t));
    }
    const std::size_t in_bytes =
        std::size_t{3} * static_cast<std::size_t>(plan.width) * static_cast<std::size_t>(plan.height) * 2;
    auto up = Buffer::create(backend.context(), backend.memory(), in_bytes, MemoryUse::Upload);
    auto back = Buffer::create(backend.context(), backend.memory(), in_bytes * 4, MemoryUse::Readback);
    auto words = Buffer::create(backend.context(), backend.memory(),
                                span_fp8::kScaleCount * sizeof(std::uint32_t), MemoryUse::Readback);
    for (auto* buffer : {&up, &back, &words})
      if (!*buffer) return failed(*buffer);
    upload = std::make_unique<Buffer>(std::move(*up));
    readback = std::make_unique<Buffer>(std::move(*back));
    peak_words = std::make_unique<Buffer>(std::move(*words));
    return {};
  }

  std::expected<void, Refusal> put(const Tensor& to, std::span<const std::byte> bytes) {
    if (auto ok = backend.finish_on_host(); !ok) return ok;  // l'envoi précédent n'est plus lu
    std::memcpy(upload->mapped(), bytes.data(), bytes.size());
    return backend.copy_on_host(upload->handle(), 0, detail::to_buffer(to.handle), to.offset, bytes.size());
  }

  std::expected<std::vector<float>, Refusal> take(const Tensor& from) {
    const VkDeviceSize size = readback->size();
    if (auto ok =
            backend.copy_on_host(detail::to_buffer(from.handle), from.offset, readback->handle(), 0, size);
        !ok)
      return failed(ok);
    return half_unit(std::span(readback->mapped(), static_cast<std::size_t>(size)));
  }

  std::expected<void, Refusal> run(Engine& engine, const Tensor& in, std::span<Tensor> outs) {
    return backend.enqueue(engine, std::span(&in, 1), outs);
  }

  std::expected<void, Refusal> clear_peaks() {
    if (auto ok = backend.finish_on_host(); !ok) return ok;
    std::memset(peak_words->mapped(), 0, static_cast<std::size_t>(peak_words->size()));
    return {};
  }

  std::expected<void, Refusal> read_peaks(std::array<float, span_fp8::kScaleCount>& into) {
    if (auto ok = backend.finish_on_host(); !ok) return ok;
    into.fill(0.0F);
    std::memcpy(into.data(), peak_words->mapped(), span_fp8::scale_count(blocks) * sizeof(float));
    return {};
  }

  // Les maxima d'une image, tuile par tuile (le demi-plan de travail : là où vivent les tenseurs relevés).
  std::expected<void, Refusal> peaks_of(const SpanStill& still) {
    std::vector<Tensor> outs{tile_out->tensor()};
    outs.insert(outs.end(), tap_tensors.begin(), tap_tensors.end());
    for (const Span1d& r : rows)
      for (const Span1d& c : cols) {
        if (progress.stop.stop_requested()) return stopped();
        if (auto ok = put(tile_in->tensor(), still_half(still, 2 * c.origin, 2 * r.origin, tile_w, tile_h));
            !ok)
          return ok;
        if (auto ok = run(*peaks_net, tile_in->tensor(), outs); !ok) return ok;
        const kernels::Window area{
            .x = c.first, .y = r.first, .width = c.last - c.first, .height = r.last - r.first};
        if (auto ok = backend.kernels_->tensor_peaks(tap_tensors, area, peak_words->view()); !ok) return ok;
      }
    return {};
  }

  std::expected<void, Refusal> calibrate(SpanFp8Derivation& result) {
    if (auto ok = clear_peaks(); !ok) return ok;
    if (request.validate_only) {
      result.scales = plan.package.scales;
      return {};
    }
    for (const SpanStill& still : request.calibration) {
      if (auto ok = peaks_of(still); !ok) return ok;
      if (auto ok = backend.finish_on_host(); !ok) return ok;
      done += 1.0F;
      report(0.05F + 0.4F * done / static_cast<float>(request.calibration.size()), "calibration");
    }
    if (auto ok = read_peaks(result.peaks); !ok) return ok;
    auto scales = span_fp8::scales_from_peaks(result.peaks, blocks);
    if (!scales) return std::unexpected(std::move(scales.error()));
    result.scales = *scales;
    return {};
  }

  // Une image jugée : la référence FP16 et le graphe FP8 sur la même entrée ; ses maxima relevés si track.
  std::expected<SpanFp8ImageScore, Refusal> judge(const SpanStill& still, bool track) {
    if (track) {
      if (auto ok = peaks_of(still); !ok) return failed(ok);
    }
    if (auto ok = put(image_in->tensor(), still_half(still, 0, 0, plan.width, plan.height)); !ok)
      return failed(ok);
    Tensor ref_tensor = ref_out->tensor();
    Tensor fp8_tensor = fp8_out->tensor();
    if (auto ok = run(*reference, image_in->tensor(), std::span(&ref_tensor, 1)); !ok) return failed(ok);
    if (auto ok = run(*fp8, image_in->tensor(), std::span(&fp8_tensor, 1)); !ok) return failed(ok);
    auto expected = take(ref_tensor);
    if (!expected) return failed(expected);
    auto got = take(fp8_tensor);
    if (!got) return failed(got);
    return backend_score(score(*expected, *got, 2 * plan.width, 2 * plan.height));
  }

  // La validation, puis les cas durs ; les maxima de la validation s'ajoutent à ceux de la calibration
  // (SpanFp8Derivation::validation_peaks), pas ceux des cas durs.
  std::expected<void, Refusal> judge_all(SpanFp8Derivation& result) {
    done = 0.0F;
    for (const bool hard : {false, true})
      for (const SpanStill& still : hard ? request.hard : request.validation) {
        if (progress.stop.stop_requested()) return stopped();
        auto scored = judge(still, !hard);
        if (!scored) return failed(scored);
        (hard ? result.hard_scores : result.scores).push_back(*scored);
        done += 1.0F;
        report(0.5F + 0.1F * done / static_cast<float>(request.validation.size() + request.hard.size()),
               "validation");
      }
    return read_peaks(result.validation_peaks);
  }

  // La mémoire des références et des tuiles rendue avant la mesure.
  std::expected<void, Refusal> release_references() {
    if (auto ok = backend.finish_on_host(); !ok) return ok;
    taps.clear();
    tap_tensors.clear();
    tile_in.reset();
    tile_out.reset();
    peaks_net.reset();
    reference.reset();
    return {};
  }

  // Le temps moyen d'un passage, sur kTimingRuns passages attendus sur l'hôte.
  std::expected<double, Refusal> timed_runs(Engine& engine, const Tensor& in, Tensor out) {
    if (auto ok = backend.finish_on_host(); !ok) return failed(ok);
    const auto started = std::chrono::steady_clock::now();
    for (int k = 0; k < kTimingRuns; ++k)
      if (auto ok = run(engine, in, std::span(&out, 1)); !ok) return failed(ok);
    if (auto ok = backend.finish_on_host(); !ok) return failed(ok);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count() /
           kTimingRuns;
  }

  // Le graphe FP8 à la taille de travail (celui de l'image s'il y est déjà) et le SPAN FP16 de la requête.
  std::expected<std::unique_ptr<Engine>, Refusal> timing_engine(const span_fp8::Package& quantized) {
    if (plan.timing_width == plan.width && plan.timing_height == plan.height) return std::move(fp8);
    fp8.reset();
    return backend.span8_engine(quantized, {.quantize = true, .peaks = false}, plan.timing_width,
                                plan.timing_height, true, progress);
  }

  std::expected<void, Refusal> time(const span_fp8::Package& quantized, SpanFp8Derivation& result) {
    auto timed_fp8 = timing_engine(quantized);
    if (!timed_fp8) return failed(timed_fp8);
    std::unique_ptr<Engine> fp16;
    if (request.fp16) {
      auto loaded = backend.load(*request.fp16);
      if (!loaded) return failed(loaded);
      fp16 = std::move(*loaded);
    }
    auto in = tensor(3, plan.timing_height, plan.timing_width);
    auto out = tensor(3, 2 * plan.timing_height, 2 * plan.timing_width);
    if (!in || !out) return std::unexpected(!in ? in.error() : out.error());
    const Tensor in_tensor = (*in)->tensor();
    const Tensor out_tensor = (*out)->tensor();
    // Une passe de chauffe chacun (pipelines, fréquences), puis les tours intercalés.
    if (auto warm = timed_runs(**timed_fp8, in_tensor, out_tensor); !warm) return failed(warm);
    if (fp16) {
      if (auto warm = timed_runs(*fp16, in_tensor, out_tensor); !warm)
        return refuse_backend(warm.error().code, "temps du SPAN FP16 : " + warm.error().detail);
    }
    std::vector<double> fp8_ms;
    std::vector<double> fp16_ms;
    for (int round = 0; round < kTimingRounds; ++round) {
      if (progress.stop.stop_requested()) return stopped();
      auto a = timed_runs(**timed_fp8, in_tensor, out_tensor);
      if (!a) return failed(a);
      fp8_ms.push_back(*a);
      if (fp16) {
        auto b = timed_runs(*fp16, in_tensor, out_tensor);
        if (!b) return failed(b);
        fp16_ms.push_back(*b);
      }
      report(0.6F + 0.38F * static_cast<float>(round + 1) / kTimingRounds, "mesure du temps");
    }
    result.kernels_ms = net::median(fp8_ms);
    result.trt_fp16_ms = fp16_ms.empty() ? 0.0 : net::median(fp16_ms);
    result.timed_size = std::format("{}x{}", plan.timing_width, plan.timing_height);
    return {};
  }

  std::expected<SpanFp8Derivation, Refusal> derive(int tile) {
    SpanFp8Derivation result;
    result.gpu_arch = backend.capabilities().gpu_arch;
    result.kernels_version = backend.capabilities().span_fp8_kernels;
    result.reference = "vulkan-fp16";
    report(0.0F, "référence FP16");
    if (auto ok = plan_tiles(tile); !ok) return failed(ok);
    if (auto ok = make_references(); !ok) return failed(ok);
    if (auto ok = make_tensors(); !ok) return failed(ok);
    if (auto ok = calibrate(result); !ok) return failed(ok);
    span_fp8::Package quantized = plan.package;
    quantized.scales = result.scales;
    quantized.acc16 = false;
    report(0.5F, "noyaux FP8");
    auto made = backend.span8_engine(quantized, {.quantize = true, .peaks = false}, plan.width, plan.height,
                                     true, progress);
    if (!made) return failed(made);
    fp8 = std::move(*made);
    if (auto ok = judge_all(result); !ok) return failed(ok);
    if (auto ok = release_references(); !ok) return failed(ok);
    if (auto ok = time(quantized, result); !ok) return failed(ok);
    report(1.0F, "terminé");
    return result;
  }
};

std::expected<SpanFp8Derivation, Refusal> VulkanBackend::derive_span_fp8(const SpanFp8Request& request,
                                                                         Progress& progress) {
  if (runtime_ || !(profile_.target.matrix_f8 || options_.span_fp8_emulated))
    return refuse_backend(
        RefusalCode::FormatUnsupported,
        std::format("SPAN FP8 sous Vulkan : {} n'a pas de matrices coopératives FP8 (RDNA 4)",
                    context_->caps().device_name));
  auto plan = check_request(request);
  if (!plan) return failed(plan);
  if (auto ok = check_memory(*context_, *plan, request.min_free_ratio); !ok) return failed(ok);
  Span8Run run{.backend = *this, .request = request, .progress = progress, .plan = std::move(*plan)};
  return run.derive(options_.span_fp8_tile > 0 ? options_.span_fp8_tile / 2 * 2 : kTile);
}

}  // namespace ajp::gpu::vk
