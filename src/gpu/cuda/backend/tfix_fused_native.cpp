// L'essai au chargement de TemporalFix natif (--tfix-native, docs/tfix-natif.md) : l'encodeur et la fenêtre
// passent une entrée de test avec leurs moteurs TensorRT, puis avec les noyaux natifs (sommes f16, mixtes,
// puis f32). Chacun des deux garde le natif seulement si sa sortie tient le seuil contre celle de TensorRT et
// s'il le bat en temps (tours intercalés, gmfss_trial::times) : l'un peut être natif, l'autre TensorRT.
// L'entrée : un dessin lisse (ondes, des carrés aux bords francs) qui glisse de (2, 1) pixels d'une image à
// l'autre et scintille (un bruit de ±3 % propre à chaque image), ce que TemporalFix voit en lecture.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <format>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/gmfss_trial.h"
#include "gpu/cuda/backend/tfix_fused_engine.h"
#include "gpu/cuda/backend/tfix_fused_kernels.h"

namespace kirari::gpu::cuda {
namespace {

int work(int net, bool half) {
  return half ? 2 * net : net;
}

std::size_t size(int value) {
  return static_cast<std::size_t>(value);
}

// Les sommes essayées, de la plus rapide à la plus juste (comme RIFE natif).
constexpr RifeNativeSums kSums[] = {RifeNativeSums::kF16, RifeNativeSums::kMixed, RifeNativeSums::kF32};

const char* sums_name(RifeNativeSums sums) {
  switch (sums) {
    case RifeNativeSums::kF16:
      return "f16";
    case RifeNativeSums::kMixed:
      return "mixtes";
    case RifeNativeSums::kF32:
      return "f32";
  }
  return "?";
}

// Le temps d'une passe, TensorRT puis natif ; pass(0) met en file une passe TensorRT, pass(1) une native.
struct Speed {
  double trt = 0.0;
  double native = 0.0;
  bool native_wins() const { return native <= trt * (1.0 - gmfss_trial::kMinGain); }
  std::string text() const {
    return std::format("une passe : TensorRT {:.2f} ms, natif {:.2f} ms", trt, native);
  }
};

template <typename Pass>
std::expected<Speed, Refusal> speed(cudaStream_t stream, Pass&& pass) {
  auto ms = gmfss_trial::times(stream, 2, std::forward<Pass>(pass));
  if (!ms) return std::unexpected(std::move(ms.error()));
  return Speed{.trt = (*ms)[0], .native = (*ms)[1]};
}

// Le canal c de l'image r de la fenêtre au point (x, y), de 0 à 1.
float drawing(int x, int y, int c, int r) {
  const auto fx = static_cast<float>(x - 2 * (r - kernels::kTfixCenter));
  const auto fy = static_cast<float>(y - (r - kernels::kTfixCenter));
  const float wave = 0.5F + 0.22F * std::sin(0.071F * fx + 0.043F * fy + static_cast<float>(c)) +
                     0.12F * std::sin(0.017F * fx - 0.029F * fy);
  const bool square = std::fmod(std::abs(fx), 96.F) < 40.F && std::fmod(std::abs(fy), 72.F) < 28.F;
  // Le scintillement : un bruit fixe par point, par canal et par image.
  std::uint32_t h = static_cast<std::uint32_t>(x) * 73856093U ^ static_cast<std::uint32_t>(y) * 19349663U ^
                    static_cast<std::uint32_t>(c * 7 + r) * 83492791U;
  h = (h ^ (h >> 13U)) * 0x5bd1e995U;
  const float noise = 0.06F * (static_cast<float>(h >> 8U) / 16777216.F - 0.5F);
  return std::clamp((square ? 0.85F - 0.2F * static_cast<float>(c) : wave) + noise, 0.F, 1.F);
}

// Des traits de test pour l'image r (4 plans à la taille du réseau) : lisses et propres à chaque image.
float trait(int x, int y, int c, int r) {
  const auto f = [](int v) { return static_cast<float>(v); };
  return 0.5F + 0.4F * std::sin(0.05F * f(x) * f(c + 1) + 0.07F * f(y) + 0.4F * f(r) + f(c));
}

std::vector<__half> planes(int h, int w, int channels, float (*f)(int, int, int, int), int r) {
  const std::size_t plane = size(h) * size(w);
  std::vector<__half> out(plane * size(channels));
  for (int c = 0; c < channels; ++c)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        out[size(c) * plane + size(y) * size(w) + size(x)] = __float2half(f(x, y, c, r));
  return out;
}

// Le PSNR de got contre want sur [begin, end), au pic max(1, |want| le plus grand) ; une valeur non finie,
// d'un côté ou de l'autre : -infini (refusé).
double psnr(const std::vector<float>& want, const std::vector<float>& got, std::size_t begin,
            std::size_t end) {
  double peak = 1.0;
  double sum = 0.0;
  for (std::size_t i = begin; i < end; ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) return -std::numeric_limits<double>::infinity();
    peak = std::max(peak, static_cast<double>(std::abs(want[i])));
    const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
    sum += d * d;
  }
  const double mse = sum / static_cast<double>(std::max<std::size_t>(end - begin, 1));
  return mse == 0.0 ? 200.0 : 10.0 * std::log10(peak * peak / mse);
}

std::unexpected<Refusal> failed(cudaError_t error, const char* step) {
  std::ignore = cudaGetLastError();
  return refuse(RefusalCode::VendorCallFailed,
                std::format("TemporalFix natif : essai ({}) : {}", step, cudaGetErrorString(error)));
}

// Copie de count demi-flottants de l'appareil en flottants.
std::expected<std::vector<float>, Refusal> fetch(const void* device, std::size_t count) {
  std::vector<__half> half(count);
  if (cudaError_t e = cudaMemcpy(half.data(), device, count * 2, cudaMemcpyDeviceToHost); e != cudaSuccess)
    return failed(e, "copie");
  std::vector<float> out(count);
  std::ranges::transform(half, out.begin(), [](__half v) { return __half2float(v); });
  return out;
}

const std::byte* bytes_at(const void* base, std::size_t elements) {
  return static_cast<const std::byte*>(base) + elements * 2U;
}

// Une fenêtre de test dans scratch : les sept images (buffers[0]), leurs traits de test et leurs pyramides
// (buffers[1], celles de l'encodeur fusionné), la sortie (buffers[2]). Les tenseurs de TfixWindowEngine.
struct TrialWindow {
  std::vector<Tensor> inputs;
  Tensor out;
};

std::expected<TrialWindow, Refusal> trial_window(FusedScratch& scratch, int h, int w, bool half,
                                                 const Shape& packed) {
  const int wh = work(h, half);
  const int ww = work(w, half);
  const std::size_t image_bytes = size(kernels::kTfixImage) * size(wh) * size(ww) * 2;
  const std::size_t packed_bytes = static_cast<std::size_t>(packed.dims[1]) * size(wh) * size(ww) * 2;
  const kernels::TfixPacked spots = kernels::tfix_packed(h, w, half);
  const std::size_t frames = size(kernels::kTfixFrames);
  if (cudaError_t e = cudaMalloc(&scratch.buffers[0], frames * image_bytes); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMalloc(&scratch.buffers[1], frames * packed_bytes); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMalloc(&scratch.buffers[2], image_bytes); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  auto* const rgb = static_cast<std::byte*>(scratch.buffers[0]);
  auto* const kept = static_cast<std::byte*>(scratch.buffers[1]);
  // Les sept images, leurs traits de test et leurs pyramides (celles de l'encodeur fusionné). Les copies
  // sur le flux de l'essai, avant les noyaux qui les lisent (un cudaMemcpy irait sur le flux par défaut,
  // que ce flux n'attend pas).
  for (int r = 0; r < kernels::kTfixFrames; ++r) {
    std::byte* const image = rgb + size(r) * image_bytes;
    std::byte* const own = kept + size(r) * packed_bytes;
    const std::vector<__half> host = planes(wh, ww, kernels::kTfixImage, drawing, r);
    const std::vector<__half> traits = planes(h, w, kernels::kTfixTraits, trait, r);
    if (cudaError_t e =
            cudaMemcpyAsync(image, host.data(), image_bytes, cudaMemcpyHostToDevice, scratch.stream);
        e != cudaSuccess)
      return failed(e, "cudaMemcpyAsync");
    if (cudaError_t e = cudaMemcpyAsync(own + spots.traits * 2, traits.data(), traits.size() * 2,
                                        cudaMemcpyHostToDevice, scratch.stream);
        e != cudaSuccess)
      return failed(e, "cudaMemcpyAsync");
    if (half) {
      if (cudaError_t e = launch_tfix_pool(image, own, h, w, scratch.stream); e != cudaSuccess)
        return failed(e, "tfix_pool");
    }
    if (cudaError_t e = launch_tfix_pyramid(image, own, h, w, half, scratch.stream); e != cudaSuccess)
      return failed(e, "tfix_pyramid");
  }
  // Le travail en vol sur les voies finit avant : l'essai écrit dans la zone partagée.
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return failed(e, "cudaDeviceSynchronize");
  std::vector<Tensor> inputs;
  inputs.reserve(2 * frames);
  for (std::size_t r = 0; r < frames; ++r)
    inputs.push_back(
        fused_tensor(reinterpret_cast<std::uintptr_t>(rgb), r * image_bytes, kernels::kTfixImage, wh, ww));
  for (std::size_t r = 0; r < frames; ++r)
    inputs.push_back({.type = TensorType::F16,
                      .shape = packed,
                      .handle = reinterpret_cast<std::uintptr_t>(kept),
                      .offset = r * packed_bytes});
  const Tensor out =
      fused_tensor(reinterpret_cast<std::uintptr_t>(scratch.buffers[2]), 0, kernels::kTfixImage, wh, ww);
  return TrialWindow{.inputs = std::move(inputs), .out = out};
}

}  // namespace

// --- L'encodeur ---

std::expected<std::vector<float>, Refusal> TfixEncoderEngine::trial_traits(
    cudaStream_t stream, const std::shared_ptr<SharedActivations>& shared, const Tensor& input,
    const Tensor& output) {
  if (auto ok = enqueue(stream, shared, std::span(&input, 1), std::span(&output, 1)); !ok)
    return std::unexpected(std::move(ok.error()));
  if (cudaError_t e = cudaStreamSynchronize(stream); e != cudaSuccess) return failed(e, "exécution");
  const kernels::TfixPacked packed = kernels::tfix_packed(height_, width_, half_);
  return fetch(bytes_at(fused_address(output), packed.traits),
               size(kernels::kTfixTraits) * size(height_) * size(width_));
}

std::expected<std::string, Refusal> TfixEncoderEngine::try_native(
    const std::vector<RifeNativePiece>& pieces, const std::shared_ptr<SharedActivations>& shared,
    double min_db) {
  const int wh = work(height_, half_);
  const int ww = work(width_, half_);
  const std::vector<__half> image = planes(wh, ww, kernels::kTfixImage, drawing, kernels::kTfixCenter);
  const Shape out_shape = output_shape();
  const auto out_bytes = static_cast<std::size_t>(out_shape.dims[1]) * size(wh) * size(ww) * 2;
  FusedScratch scratch;
  if (cudaError_t e = cudaMalloc(&scratch.buffers[0], image.size() * 2); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMalloc(&scratch.buffers[1], out_bytes); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  if (cudaError_t e = cudaMemcpyAsync(scratch.buffers[0], image.data(), image.size() * 2,
                                      cudaMemcpyHostToDevice, scratch.stream);
      e != cudaSuccess)
    return failed(e, "cudaMemcpyAsync");
  // Le travail en vol sur les voies finit avant : l'essai écrit dans la zone partagée.
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return failed(e, "cudaDeviceSynchronize");
  const Tensor input =
      fused_tensor(reinterpret_cast<std::uintptr_t>(scratch.buffers[0]), 0, kernels::kTfixImage, wh, ww);
  const Tensor output{.type = TensorType::F16,
                      .shape = out_shape,
                      .handle = reinterpret_cast<std::uintptr_t>(scratch.buffers[1]),
                      .offset = 0};
  native_on_ = false;
  auto reference = trial_traits(scratch.stream, shared, input, output);
  if (!reference) return std::unexpected(std::move(reference.error()));
  // Les passes natives sur une zone à elles : la zone de la voie ne grandit que pour un natif retenu.
  const auto own = std::make_shared<SharedActivations>();
  const auto back_to_trt = [this] {
    native_on_ = false;
    native_.reset();
  };
  std::string tried;
  for (const RifeNativeSums sums : kSums) {
    std::string error;
    native_on_ = false;
    native_ = TfixNativeNet::create(pieces, height_, width_, sums, error);
    if (!native_) return refuse(RefusalCode::FormatUnsupported, "TemporalFix natif : encodeur : " + error);
    native_on_ = true;
    auto got = trial_traits(scratch.stream, own, input, output);
    if (!got) {
      back_to_trt();
      return std::unexpected(std::move(got.error()));
    }
    const double db = psnr(*reference, *got, 0, got->size());
    tried += std::format("{}sommes {} : {:.1f} dB", tried.empty() ? "" : ", ", sums_name(sums), db);
    if (db < min_db) continue;
    auto timed = speed(scratch.stream, [&](std::size_t v) {
      native_on_ = v == 1;
      return enqueue(scratch.stream, v == 1 ? own : shared, std::span(&input, 1), std::span(&output, 1));
    });
    native_on_ = true;
    if (!timed) {
      back_to_trt();
      return std::unexpected(std::move(timed.error()));
    }
    if (!timed->native_wins()) {
      back_to_trt();
      return std::format(
          "TemporalFix natif écarté pour l'encodeur, TensorRT plus rapide ({}×{}, {} contre TensorRT ; {})",
          width_, height_, tried, timed->text());
    }
    if (auto ok = shared->reserve(native_->encoder_workspace_bytes()); !ok) {
      back_to_trt();
      return std::unexpected(std::move(ok.error()));
    }
    return std::format(
        "TemporalFix natif retenu pour l'encodeur ({}×{}, {} contre TensorRT, seuil {:.0f} dB ; {})", width_,
        height_, tried, min_db, timed->text());
  }
  back_to_trt();
  return std::format(
      "TemporalFix natif écarté pour l'encodeur, TensorRT le garde ({}×{}, {} contre TensorRT, seuil {:.0f} "
      "dB)",
      width_, height_, tried, min_db);
}

// --- La fenêtre ---

std::expected<void, Refusal> TfixWindowEngine::use_native(bool on,
                                                          const std::shared_ptr<SharedActivations>& shared) {
  native_on_ = on && native_ != nullptr;
  layout_ = kernels::tfix_layout(height_, width_, half_, native_on_);
  // La zone de travail des noyaux s'ajoute aux activations : les tampons de la fenêtre montent au-dessus.
  own_activation_ = native_on_ ? std::max(trt_activation_, native_->workspace_bytes()) : trt_activation_;
  graphs_.clear();  // capturés sur l'autre disposition
  std::ignore = graphs_.take_notice();
  scratch_offset_ = scratch_offset_for(*shared);
  return use_zone(shared);
}

std::expected<std::vector<float>, Refusal> TfixWindowEngine::trial_output(
    cudaStream_t stream, const std::shared_ptr<SharedActivations>& shared, std::span<const Tensor> inputs,
    const Tensor& out) {
  // Trois passes : directe, capture du graphe, rejeu ; la dernière est comparée.
  for (int pass = 0; pass < 3; ++pass) {
    if (auto ok = enqueue(stream, shared, inputs, std::span(&out, 1)); !ok)
      return std::unexpected(std::move(ok.error()));
    if (cudaError_t e = cudaStreamSynchronize(stream); e != cudaSuccess) return failed(e, "exécution");
  }
  // La sortie (3 plans de travail), puis Y2 [canal][appui][point] et la tête [appui][point], quelle que soit
  // la disposition : les groupes de canaux comparés à part se suivent.
  const std::size_t image =
      size(kernels::kTfixImage) * size(work(height_, half_)) * size(work(width_, half_));
  auto result = fetch(fused_address(out), image);
  if (!result) return result;
  const void* zone = static_cast<const std::byte*>(shared->data()) + scratch_offset_;
  const kernels::TfixDims y2 = kernels::tfix_block_y(2, height_, width_);
  const kernels::TfixDims l2 = kernels::tfix_level(2, height_, width_, 1);
  const struct {
    std::size_t offset;
    int channels;
    int ld;
    std::int64_t plane;
  } parts[] = {{layout_.y[2], kernels::kTfixY, kernels::tfix_out_ld(kernels::kTfixY, native_on_), y2.plane()},
               {layout_.conf_out, 1, kernels::tfix_out_ld(1, native_on_), l2.plane()}};
  for (const auto& part : parts) {
    const auto plane = static_cast<std::size_t>(part.plane);
    const auto ld = static_cast<std::size_t>(part.ld);
    const std::size_t supports = size(kernels::kTfixSupports);
    auto raw = fetch(bytes_at(zone, part.offset), supports * ld * plane);
    if (!raw) return raw;
    for (std::size_t c = 0; c < size(part.channels); ++c)
      for (std::size_t k = 0; k < supports; ++k)
        for (std::size_t p = 0; p < plane; ++p)
          result->push_back((*raw)[k * ld * plane + (native_on_ ? p * ld + c : c * plane + p)]);
  }
  return result;
}

std::expected<std::string, Refusal> TfixWindowEngine::try_native(
    const std::vector<RifeNativePiece>& pieces, const std::shared_ptr<SharedActivations>& shared,
    double min_db) {
  const int h = height_;
  const int w = width_;
  FusedScratch scratch;
  auto window = trial_window(scratch, h, w, half_, packed_shape());
  if (!window) return std::unexpected(std::move(window.error()));
  const std::vector<Tensor>& inputs = window->inputs;
  const Tensor& out = window->out;
  const auto back_to_trt = [&] {
    native_.reset();
    return use_native(false, shared);
  };
  if (auto ok = use_native(false, shared); !ok) return std::unexpected(std::move(ok.error()));
  auto reference = trial_output(scratch.stream, shared, inputs, out);
  if (!reference) return std::unexpected(std::move(reference.error()));
  // Les groupes comparés, chacun à son pic : la sortie, le flot de Y2 (canaux 0 et 1, en pixels), le reste
  // de Y2 (masque et attributs, autour de 1), la tête.
  const std::size_t image = size(kernels::kTfixImage) * size(work(h, half_)) * size(work(w, half_));
  const std::size_t y2 =
      size(kernels::kTfixSupports) * static_cast<std::size_t>(kernels::tfix_block_y(2, h, w).plane());
  const std::size_t ends[] = {image, image + 2 * y2, image + size(kernels::kTfixY) * y2, reference->size()};
  // Les passes natives sur une zone à elles : la zone de la voie ne grandit que pour un natif retenu.
  const auto own = std::make_shared<SharedActivations>();
  // Le temps contre TensorRT. Passer d'un côté à l'autre vide les graphes : la première passe d'un tour (de
  // chauffe, hors mesure) bascule, puis passe deux fois (directe, capturée) avant celle du tour.
  const auto time_native = [&]() {
    std::size_t current = 1;
    return speed(scratch.stream, [&](std::size_t v) -> std::expected<void, Refusal> {
      const std::shared_ptr<SharedActivations>& zone = v == 1 ? own : shared;
      if (v != current) {
        current = v;
        if (auto ok = use_native(v == 1, zone); !ok) return ok;
        for (int pass = 0; pass < 2; ++pass)
          if (auto ok = enqueue(scratch.stream, zone, inputs, std::span(&out, 1)); !ok) return ok;
      }
      return enqueue(scratch.stream, zone, inputs, std::span(&out, 1));
    });
  };
  std::string tried;
  for (const RifeNativeSums sums : kSums) {
    std::string error;
    native_ = TfixNativeNet::create(pieces, h, w, sums, error);
    if (!native_) {
      std::ignore = back_to_trt();
      return refuse(RefusalCode::FormatUnsupported, "TemporalFix natif : fenêtre : " + error);
    }
    if (auto ok = use_native(true, own); !ok) {
      std::ignore = back_to_trt();
      return std::unexpected(std::move(ok.error()));
    }
    auto got = trial_output(scratch.stream, own, inputs, out);
    if (!got || got->size() != reference->size()) {
      std::ignore = back_to_trt();
      if (!got) return std::unexpected(std::move(got.error()));
      return refuse(RefusalCode::GraphInvalid, "TemporalFix natif : essai de taille différente");
    }
    double db = std::numeric_limits<double>::infinity();
    std::size_t begin = 0;
    for (const std::size_t end : ends) {
      db = std::min(db, psnr(*reference, *got, begin, end));
      begin = end;
    }
    tried += std::format("{}sommes {} : {:.1f} dB", tried.empty() ? "" : ", ", sums_name(sums), db);
    if (db < min_db) continue;
    auto timed = time_native();
    if (!timed || !timed->native_wins()) {
      if (auto ok = back_to_trt(); !ok) return std::unexpected(std::move(ok.error()));
      if (!timed) return std::unexpected(std::move(timed.error()));
      return std::format(
          "TemporalFix natif écarté pour la fenêtre, TensorRT plus rapide ({}×{}, {} contre TensorRT ; {})",
          w, h, tried, timed->text());
    }
    // Retenu : sur la zone de la voie (use_native y prend la place des tampons, et vide les graphes).
    if (auto ok = use_native(true, shared); !ok) {
      std::ignore = back_to_trt();
      return std::unexpected(std::move(ok.error()));
    }
    return std::format(
        "TemporalFix natif retenu pour la fenêtre ({}×{}, {} contre TensorRT, seuil {:.0f} dB ; {})", w, h,
        tried, min_db, timed->text());
  }
  if (auto ok = back_to_trt(); !ok) return std::unexpected(std::move(ok.error()));
  return std::format(
      "TemporalFix natif écarté pour la fenêtre, TensorRT la garde ({}×{}, {} contre TensorRT, seuil {:.0f} "
      "dB)",
      w, h, tried, min_db);
}

}  // namespace kirari::gpu::cuda
