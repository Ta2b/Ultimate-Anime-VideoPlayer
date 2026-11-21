// Façade des noyaux SPAN FP8 et FP16 natifs : choix du corps, poids et lancements.
#include "gpu/cuda/backend/span_fp8_net.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/span_fp8_body.h"

namespace kirari::gpu::cuda {
namespace {

constexpr bool kSpanFp8 = KIRARI_SPAN_FP8_KERNELS;
constexpr bool kSpanFp16 = KIRARI_SPAN_FP16_KERNELS;
constexpr int kStored = 64;  // canaux du paquet (gpu/span_fp8/package.h)

// Poids rangés [outs][taps][groups × kStored] (le paquet) -> [outs][taps][groups × ch] : les ch premières
// entrées de chaque groupe. Les sorties gardées sont les outs premières, qui se suivent déjà.
std::vector<std::uint16_t> narrow(const std::uint16_t* from, std::size_t outs, std::size_t taps,
                                  std::size_t groups, std::size_t ch) {
  std::vector<std::uint16_t> out(outs * taps * groups * ch);
  std::size_t at = 0;
  for (std::size_t o = 0; o < outs; ++o)
    for (std::size_t t = 0; t < taps; ++t)
      for (std::size_t g = 0; g < groups; ++g)
        for (std::size_t c = 0; c < ch; ++c) out[at++] = from[((o * taps + t) * groups + g) * kStored + c];
  return out;
}

// Les poids d'un modèle à ch canaux réels rangé en 64 avec des zéros, réduits à ch. Les biais et conv_1 (16
// entrées) se lisent déjà au début de leurs tableaux : leurs vues restent celles du paquet.
struct Narrowed {
  std::vector<std::vector<std::uint16_t>> store;
  SpanFp8Weights weights;

  Narrowed(const SpanFp8Weights& from, std::size_t ch) : weights(from) {
    const auto keep = [&](SpanFp8ConvView& view, std::size_t outs, std::size_t taps, std::size_t groups) {
      store.push_back(narrow(view.weights, outs, taps, groups, ch));
      view.weights = store.back().data();
    };
    store.reserve(4 * SpanFp8Weights::kBlocks + 2);
    for (int b = 0; b < from.blocks; ++b) {
      for (auto& conv : weights.conv[b]) keep(conv, ch, 9, 1);
      keep(weights.attention[b], ch, 1, 1);
    }
    keep(weights.concat, ch, 1, 4);
    keep(weights.pixel, 48, 9, 1);
  }
};

std::unique_ptr<SpanFp8Body> make_body(const SpanFp8Weights& weights, bool narrow48, int height, int width) {
  if (weights.fp16) {
    if constexpr (kSpanFp16)
      return make_span_fp8_body16(height, width, narrow48 ? 48 : kStored, weights.sums);
  } else if constexpr (kSpanFp8) {
    return narrow48 ? make_span_fp8_body48(height, width, weights.width48 == SpanFp8Width::Native)
                    : make_span_fp8_body64(height, width);
  }
  return {};
}

}  // namespace

struct SpanFp8Net::Impl {
  std::unique_ptr<SpanFp8Body> body;
};

SpanFp8Net::SpanFp8Net(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SpanFp8Net::~SpanFp8Net() = default;

bool SpanFp8Net::supported_device(int major, int minor) {
  // Ada (sm 8.9) seulement. Blackwell grand public (sm 12.0, RTX 50, étape 10.2) a la même boucle CUTLASS
  // (garde >= 890) et la bibliothèque est compilée pour lui, mais aucun noyau n'y a jamais tourné : pas de
  // FP8 sur RTX 50 avant une mesure sur la carte (décision 14 des audits, A37-01 ; docs/span-fp8-cutlass.md).
  return kSpanFp8 && major == 8 && minor == 9;
}

bool SpanFp8Net::supported_fp16_device(int major, int minor) {
  return kSpanFp16 && ((major == 8 && minor == 9) || (major == 12 && minor == 0));
}

std::int64_t SpanFp8Net::workspace_bytes() const {
  return impl_->body->workspace_bytes();
}

void SpanFp8Net::bind(void* workspace) {
  impl_->body->bind(workspace);
}

std::unique_ptr<SpanFp8Net> SpanFp8Net::create(const SpanFp8Weights& weights, int height, int width,
                                               SpanFp8Status& status, std::string& error) {
  // Deux blocs au moins : le premier et le dernier écrivent chacun leur tranche de la concaténation.
  if (weights.blocks < 2 || weights.blocks > SpanFp8Weights::kBlocks) {
    status = SpanFp8Status::Failed;
    error = "nombre de blocs hors de 2 à 8";
    return nullptr;
  }
  if (weights.channels != 48 && weights.channels != kStored) {
    status = SpanFp8Status::Failed;
    error = "canaux réels hors de 48 et 64";
    return nullptr;
  }
  const bool narrow48 = weights.channels == 48 && weights.width48 != SpanFp8Width::Padded;
  const bool native_relu_shape =
      weights.channels == kStored || (narrow48 && weights.width48 == SpanFp8Width::Native);
  if (weights.native_relu && (weights.fp16 || !native_relu_shape)) {
    status = SpanFp8Status::Failed;
    error = "les ReLU natives demandent du FP8 C64 ou C48 en K à plat";
    return nullptr;
  }
  // K par position à 48 canaux et accumulateur f16 : 7 % de NaN sur la RTX 4070 Ti, poids à queue lourde,
  // alors que K à plat et le calcul en 64 restent finis. Ce chemin ne sert qu'à la mesure et n'y gagne rien
  // (kirari_span_tile_bench : 48 canaux sans gain sur 64 en FP8) : refusé plutôt que corrigé.
  if (narrow48 && weights.width48 == SpanFp8Width::PerTap && weights.acc16 && !weights.fp16) {
    status = SpanFp8Status::Failed;
    error = "48 canaux en K par position : pas d'accumulateur f16 (NaN mesurés)";
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->body = make_body(weights, narrow48, height, width);
  if (!impl->body) {
    status = SpanFp8Status::Failed;
    error = weights.fp16 ? "noyaux FP16 natifs absents" : "noyaux FP8 absents";
    return nullptr;
  }
  bool sent = false;
  if (narrow48) {
    const Narrowed narrowed(weights, 48);
    sent = impl->body->upload(narrowed.weights, status, error);
  } else {
    sent = impl->body->upload(weights, status, error);
  }
  if (!sent) return nullptr;
  // Les poids paginés partent sur la voie par défaut. La première inférence peut être enfilée
  // sur une voie non bloquante qui ne rejoint pas automatiquement le transfert.
  if (const auto e = cudaStreamSynchronize(nullptr); e != cudaSuccess) {
    status = e == cudaErrorMemoryAllocation ? SpanFp8Status::OutOfMemory : SpanFp8Status::Failed;
    error = cudaGetErrorString(e);
    return nullptr;
  }
  return std::unique_ptr<SpanFp8Net>(new SpanFp8Net(std::move(impl)));
}

bool SpanFp8Net::launch(cudaStream_t stream, const void* input, void* output, std::string& error) {
  static_cast<void>(poll_accumulator());  // pas de graphe ici
  return launch_body(stream, input, error) && launch_tail(stream, output, {}, error);
}

SpanFp8AccChange SpanFp8Net::poll_accumulator() {
  return impl_->body->poll_accumulator();
}

int SpanFp8Net::overflow_frames() const {
  return impl_->body->overflow_frames();
}

bool SpanFp8Net::launch_body(cudaStream_t stream, const void* input, std::string& error) {
  return impl_->body->launch_body(stream, input, error);
}

bool SpanFp8Net::launch_tail(cudaStream_t stream, void* output, const SpanFp8Ring& ring, std::string& error) {
  return impl_->body->launch_tail(stream, output, ring, error);
}

kernels::SpanTileGrid SpanFp8Net::tile_grid() const {
  return impl_->body->tile_grid();
}

void SpanFp8Net::pack_tile_layers(std::span<const std::uint8_t> core, std::uint32_t* body) const {
  impl_->body->pack_tile_layers(core, body);
}

void SpanFp8Net::set_tile_masks(const std::uint32_t* region, const std::uint32_t* core, bool layered) {
  impl_->body->set_tile_masks(region, core, layered);
}

bool SpanFp8Net::shift_output(cudaStream_t stream, const void* from, void* to, int dx, int dy,
                              std::string& error) {
  return impl_->body->shift_output(stream, from, to, dx, dy, error);
}

}  // namespace kirari::gpu::cuda
