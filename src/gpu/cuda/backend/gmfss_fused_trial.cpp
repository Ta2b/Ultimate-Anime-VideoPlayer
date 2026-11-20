// L'essai au chargement du GMFSS natif pour la fusion (GmfssFusedEngine::try_native, docs/gmfss-natif.md) :
// le GridNet natif réglé au banc, puis des entrées de test passent par le moteur TensorRT de la fusion
// entière et par l'IFNet et le GridNet natif ; le choix des sommes de chaque morceau (les trois lignes, la
// queue) est celui de gmfss_trial.h. Les entrées : deux images lisses décalées de quelques pixels (l'IFNet y
// a du travail), leurs plans recopiés en tête des warped*_1, et des caractéristiques en ondes de l'ordre de
// celles de FeatureNet.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <format>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/fused_graphs.h"
#include "gpu/cuda/backend/gmfss_fused_engine.h"
#include "gpu/cuda/backend/gmfss_mix.h"
#include "gpu/cuda/backend/gmfss_trial.h"

namespace kirari::gpu::cuda {
namespace {

// Les plans d'une entrée de test : images aux trois premiers plans des entrées half* et warped*_1, ondes
// au-delà (de pente et de phase propres au plan), l'instant à 0,5.
std::vector<__half> test_input(const Shape& shape, int input, float shift) {
  const auto c_count = static_cast<int>(shape.dims[1]);
  const auto h = static_cast<int>(shape.dims[2]);
  const auto w = static_cast<int>(shape.dims[3]);
  std::vector<__half> out(static_cast<std::size_t>(c_count) * static_cast<std::size_t>(h) *
                          static_cast<std::size_t>(w));
  std::size_t at = 0;
  for (int c = 0; c < c_count; ++c)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        float v = 0.5F;
        if (h > 1 && c < 3) {
          v = gmfss_trial::drawing(static_cast<float>(x), static_cast<float>(y), c, shift);
        } else if (h > 1) {
          const float k = 0.05F + 0.003F * static_cast<float>((c * 7 + input) % 17);
          v = 0.8F * std::sin(k * static_cast<float>(x) + 0.7F * static_cast<float>(c)) *
              std::cos(0.6F * k * static_cast<float>(y) - static_cast<float>(input));
        }
        out[at++] = __float2half(v);
      }
  return out;
}

template <class Switch>
std::expected<void, Refusal> verify_tail_speed(GmfssNativeNet& native, const Switch& use,
                                               std::string_view name, std::string_view quality,
                                               cudaStream_t stream, void* out,
                                               std::vector<std::string>& lines) {
  FusedGraphs graphs(2, "GMFSS natif, essai du PixelShuffle");
  const auto timed = gmfss_trial::times(
      stream, 4,
      [&](std::size_t variant) {
        use((variant & 1U) != 0);
        const auto queue = [&] {
          return cuda_check(native.run_piece(stream, prepare::kGmfssQueuePiece, out),
                            "GMFSS natif : queue du GridNet");
        };
        return variant < 2 ? queue() : graphs.launch(stream, FusedGraphs::Key{variant}, queue);
      },
      gmfss_trial::kPieceRounds);
  const auto& captures = graphs.counters();
  const bool keep = timed && captures.captures == 2 && captures.replays >= 2 &&
                    captures.capture_failures == 0 && captures.launch_failures == 0 &&
                    gmfss_mix::beats((*timed)[0], (*timed)[1]) && gmfss_mix::beats((*timed)[2], (*timed)[3]);
  use(keep);
  if (!timed && timed.error().code == RefusalCode::DeviceLost) return std::unexpected(timed.error());
  if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
    return std::unexpected(lost_refusal(lost, "GMFSS natif : essai du PixelShuffle"));
  if (timed) {
    lines.push_back(std::format(
        "GMFSS natif, {} {} ({}) : {:.3f} à {:.3f} ms en direct, {:.3f} à {:.3f} ms sous graphe", name,
        keep ? "retenu" : "écarté", quality, (*timed)[0], (*timed)[1], (*timed)[2], (*timed)[3]));
  } else {
    lines.push_back(std::format("GMFSS natif, {} écarté : {}", name, timed.error().detail));
  }
  return {};
}

template <class Pass, class Full>
std::expected<void, Refusal> verify_grid_variants(GmfssNativeNet& native, const Pass& pass, const Full& full,
                                                  double min_db, std::size_t bytes, cudaStream_t stream,
                                                  void* out, std::vector<std::string>& lines) {
  const auto reference = pass(0xff);
  if (!reference) {
    if (reference.error().code == RefusalCode::DeviceLost) return std::unexpected(reference.error());
    if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "GMFSS natif : essai du GridNet"));
    lines.push_back("GMFSS natif, référence des variantes du GridNet indisponible : " +
                    reference.error().detail);
  } else {
    const auto verify = [&](auto switch_on, std::string_view kept,
                            std::string_view refused) -> std::expected<bool, Refusal> {
      switch_on(true);
      const auto candidate = pass(0x7e);
      if (candidate && std::memcmp(reference->data(), candidate->data(), bytes) == 0) {
        if (!kept.empty()) lines.emplace_back(kept);
        return true;
      }
      switch_on(false);
      if (!candidate && candidate.error().code == RefusalCode::DeviceLost)
        return std::unexpected(candidate.error());
      if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
        return std::unexpected(lost_refusal(lost, "GMFSS natif : essai du GridNet"));
      lines.push_back(std::string(refused) +
                      (candidate ? "sortie FP16 différente" : candidate.error().detail));
      return false;
    };
    if (native.join_ready()) {
      auto checked = verify([&](bool on) { native.use_join(on); },
                            "GMFSS natif, anneaux C64 avec addition retenus (FP16 identique)",
                            "GMFSS natif, anneaux C64 avec addition écartés : ");
      if (!checked) return std::unexpected(std::move(checked.error()));
    }
    if (native.planar_head_ready()) {
      auto checked = verify([&](bool on) { native.use_planar_head(on); },
                            "GMFSS natif, lecture de la tête en neuf plans retenue (FP16 identique)",
                            "GMFSS natif, lecture de la tête en neuf plans écartée : ");
      if (!checked) return std::unexpected(std::move(checked.error()));
    }
    if (native.tail_ready()) {
      auto checked =
          verify([&](bool on) { native.use_tail(on); }, {}, "GMFSS natif, anneau du PixelShuffle écarté : ");
      if (!checked) return std::unexpected(std::move(checked.error()));
      if (*checked) {
        auto speed = verify_tail_speed(
            native, [&](bool on) { native.use_tail(on); }, "anneau du PixelShuffle", "FP16 identique", stream,
            out, lines);
        if (!speed) return speed;
      }
    }
    if (native.folded_tail_ready()) {
      const auto trt = full();
      if (!trt && trt.error().code == RefusalCode::DeviceLost) return std::unexpected(trt.error());
      const auto use = [&](bool on) { native.use_folded_tail(on); };
      use(true);
      const auto candidate = pass(0x7e);
      use(false);
      if (!candidate && candidate.error().code == RefusalCode::DeviceLost)
        return std::unexpected(candidate.error());
      if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
        return std::unexpected(lost_refusal(lost, "GMFSS natif : essai de la queue composée"));
      if (trt && candidate) {
        const double db = gmfss_trial::output_db(*reference, *candidate);
        const double trt_db = gmfss_trial::output_db(*trt, *candidate);
        // La comparaison avec le chemin natif borne également les écarts locaux, aux bords compris.
        float peak = 0.F;
        for (std::size_t i = 0; i < candidate->size(); ++i)
          peak = std::max(peak, std::fabs(__half2float((*candidate)[i]) - __half2float((*reference)[i])));
        const auto quality =
            std::format("écart {:.2f} dB au natif et {:.2f} dB à TensorRT, pic {:.6f}", db, trt_db, peak);
        if (db >= 60. && trt_db >= min_db && peak <= 0.004F)
          return verify_tail_speed(native, use, "queue composée", quality, stream, out, lines);
        lines.push_back(std::format("GMFSS natif, queue composée écartée ({})", quality));
      } else {
        lines.push_back("GMFSS natif, queue composée écartée : " +
                        (trt ? candidate.error().detail : trt.error().detail));
      }
    }
  }
  return {};
}

}  // namespace

std::expected<std::string, Refusal> GmfssFusedEngine::try_ifnet(
    const std::array<RifeNativePiece, 4>& pieces, cudaStream_t stream,
    const std::shared_ptr<SharedActivations>& shared, std::span<const Tensor> inputs, const Tensor& out,
    double min_db) {
  std::string error;
  native_ifnet_ = GmfssIfnetNative::create(pieces, height_, width_, stream, error);
  if (!native_ifnet_) {
    if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "GMFSS natif : réglage de l'IFNet"));
    return "GMFSS natif, IFNet : " + error;
  }
  const auto down = 3U * static_cast<std::size_t>(height_) * static_cast<std::size_t>(width_);
  const auto pass = [&](std::size_t v) { return enqueue_ifnet(stream, shared, inputs, v != 0); };
  const auto fetch = [&](const void* addr, std::size_t count) -> std::expected<std::vector<__half>, Refusal> {
    if (auto ok = cuda_check(cudaStreamSynchronize(stream), "GMFSS natif : essai IFNet"); !ok)
      return std::unexpected(std::move(ok.error()));
    std::vector<__half> host(count);
    if (auto ok = cuda_check(cudaMemcpy(host.data(), addr, count * sizeof(__half), cudaMemcpyDeviceToHost),
                             "GMFSS natif : copie IFNet");
        !ok)
      return std::unexpected(std::move(ok.error()));
    return host;
  };
  if (auto ok = pass(0); !ok) return std::unexpected(std::move(ok.error()));
  auto reference = fetch(rife_, down);
  if (!reference) return std::unexpected(std::move(reference.error()));
  if (auto ok = pass(1); !ok) return std::unexpected(std::move(ok.error()));
  auto candidate = fetch(rife_, down);
  if (!candidate) return std::unexpected(std::move(candidate.error()));
  const double db = gmfss_trial::output_db(*reference, *candidate);
  const auto ms = gmfss_trial::times(stream, 2, pass);
  if (!ms) return std::unexpected(ms.error());

  double fusion_db = db;
  if (db >= min_db) {
    const Tensor outputs[]{out};
    if (auto ok = full_->enqueue(stream, shared, inputs, outputs); !ok)
      return std::unexpected(std::move(ok.error()));
    auto full = fetch(fused_address(out), 4 * down);
    if (!full) return std::unexpected(std::move(full.error()));
    if (auto ok = enqueue_native(stream, shared, inputs, out); !ok)
      return std::unexpected(std::move(ok.error()));
    auto fused = fetch(fused_address(out), 4 * down);
    if (!fused) return std::unexpected(std::move(fused.error()));
    fusion_db = gmfss_trial::output_db(*full, *fused);
  }
  const bool keep = db >= min_db && fusion_db >= min_db && gmfss_mix::beats((*ms)[0], (*ms)[1]);
  if (!keep) native_ifnet_.reset();
  return std::format(
      "GMFSS natif, IFNet {} sur {}×{} : {:.2f} dB seul, {:.2f} dB en fusion "
      "(seuil {:.0f} dB) ; TensorRT {:.3f} ms, CUDA {:.3f} ms",
      keep ? "retenu" : "écarté", width_, height_, db, fusion_db, min_db, (*ms)[0], (*ms)[1]);
}

std::expected<std::vector<std::string>, Refusal> GmfssFusedEngine::try_native(
    const prepare::GmfssGrid& grid, const std::shared_ptr<SharedActivations>& shared, double min_db,
    const std::array<RifeNativePiece, 4>* ifnet) {
  gmfss_trial::Scratch scratch;
  const auto upload = [&](const std::vector<__half>& host) -> std::expected<Tensor, Refusal> {
    void* device = nullptr;
    if (auto ok = cuda_check(cudaMalloc(&device, host.size() * 2), "GMFSS natif : essai"); !ok)
      return std::unexpected(std::move(ok.error()));
    scratch.buffers.push_back(device);
    if (auto ok = cuda_check(cudaMemcpy(device, host.data(), host.size() * 2, cudaMemcpyHostToDevice),
                             "GMFSS natif : essai");
        !ok)
      return std::unexpected(std::move(ok.error()));
    return Tensor{.type = TensorType::F16, .shape = {}, .handle = reinterpret_cast<std::uintptr_t>(device)};
  };
  std::vector<Tensor> inputs(input_names().size());
  for (std::size_t i = 0; i < kInputs; ++i) {
    const Shape shape = full_->input_shape(at_[i]);
    // B (half1, warped1_*) est A décalée de 4 pixels.
    const float shift = i == kHalf1 || i >= kW11 ? 4.F : 0.F;
    auto made = upload(test_input(shape, static_cast<int>(i), shift));
    if (!made) return std::unexpected(std::move(made.error()));
    made->shape = shape;
    inputs[at_[i]] = *made;
  }
  const Shape out_shape = full_->output_shape(0);
  const auto elements = static_cast<std::size_t>(3 * out_shape.dims[2] * out_shape.dims[3]);
  auto out = upload(std::vector<__half>(elements));
  if (!out) return std::unexpected(std::move(out.error()));
  out->shape = out_shape;
  if (auto ok =
          cuda_check(cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking), "cudaStreamCreate");
      !ok)
    return std::unexpected(std::move(ok.error()));
  // Le travail en vol sur les voies finit avant : l'essai écrit dans la zone partagée (comme load).
  if (auto ok = cuda_check(cudaDeviceSynchronize(), "GMFSS natif : essai"); !ok)
    return std::unexpected(std::move(ok.error()));
  // Quoi qu'il arrive, l'essai laisse la fusion à TensorRT ou au natif retenu.
  native_.reset();
  native_ifnet_.reset();
  native_on_ = false;
  const auto started = std::chrono::steady_clock::now();
  std::string error;
  const char* packed = std::getenv("KIRARI_AB_GMFSS_PACKED_SPLATS");  // env: banc
  features_c8_ = packed != nullptr && std::string_view(packed) == "2";
  std::unique_ptr<GmfssNativeNet> net =
      GmfssNativeNet::create(grid, height_, width_, scratch.stream, error, features_c8_);
  if (!net) {
    if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "GMFSS natif : réglage de la fusion"));
    return refuse(RefusalCode::FormatUnsupported, "GMFSS natif : " + error);
  }
  const double tuning = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const Tensor outputs[]{*out};
  native_ = std::move(net);
  gmfss_trial::Unit unit{
      .name = "fusion",
      .width = 2 * width_,
      .height = 2 * height_,
      .parts = native_.get(),
      .stream = scratch.stream,
      .outputs = 1,
      .pass =
          [&](bool native) {
            native_on_ = native;
            auto ok = enqueue(scratch.stream, shared, inputs, outputs);
            native_on_ = false;
            return ok;
          },
      .piece =
          [&](int piece) { return cuda_check(native_->run_piece(scratch.stream, piece), "GMFSS natif"); },
      .fetch = [&](std::size_t) -> std::expected<std::vector<__half>, Refusal> {
        std::vector<__half> host(elements);
        if (auto ok =
                cuda_check(cudaMemcpy(host.data(), fused_address(*out), elements * 2, cudaMemcpyDeviceToHost),
                           "GMFSS natif : copie de l'essai");
            !ok)
          return std::unexpected(std::move(ok.error()));
        return host;
      }};
  auto outcome = gmfss_trial::try_mix(unit, min_db, tuning);
  if (!outcome || !outcome->keep) native_.reset();
  native_on_ = native_ != nullptr;
  if (!outcome) return std::unexpected(std::move(outcome.error()));
  if (native_on_ && ifnet != nullptr) {
    auto tested = try_ifnet(*ifnet, scratch.stream, shared, inputs, *out, min_db);
    if (tested) {
      outcome->lines.push_back(std::move(*tested));
    } else {
      native_ifnet_.reset();
      if (tested.error().code == RefusalCode::DeviceLost) return std::unexpected(std::move(tested.error()));
      outcome->lines.push_back("GMFSS natif, IFNet : " + tested.error().detail);
    }
  }
  if (native_on_ && (native_->join_ready() || native_->planar_head_ready() || native_->tail_ready() ||
                     native_->folded_tail_ready())) {
    const auto bytes = elements * sizeof(__half);
    const auto pass = [&](int marker) -> std::expected<std::vector<__half>, Refusal> {
      if (auto ok = cuda_check(cudaMemsetAsync(fused_address(*out), marker, bytes, scratch.stream),
                               "GMFSS natif : préparation de la sortie du GridNet");
          !ok)
        return std::unexpected(std::move(ok.error()));
      auto launched = enqueue_native(scratch.stream, shared, inputs, *out);
      if (auto ok = cuda_check(cudaStreamSynchronize(scratch.stream), "GMFSS natif : essai du GridNet"); !ok)
        return std::unexpected(std::move(ok.error()));
      if (!launched) return std::unexpected(std::move(launched.error()));
      return unit.fetch(0);
    };
    const auto full = [&]() -> std::expected<std::vector<__half>, Refusal> {
      const bool was_native = native_on_;
      native_on_ = false;
      auto launched = enqueue(scratch.stream, shared, inputs, outputs);
      native_on_ = was_native;
      if (auto ok = cuda_check(cudaStreamSynchronize(scratch.stream), "GMFSS natif : référence de la queue");
          !ok)
        return std::unexpected(std::move(ok.error()));
      if (!launched) return std::unexpected(std::move(launched.error()));
      return unit.fetch(0);
    };
    if (auto checked = verify_grid_variants(*native_, pass, full, min_db, bytes, scratch.stream,
                                            fused_address(*out), outcome->lines);
        !checked)
      return std::unexpected(std::move(checked.error()));
  }
  return std::move(outcome->lines);
}

}  // namespace kirari::gpu::cuda
