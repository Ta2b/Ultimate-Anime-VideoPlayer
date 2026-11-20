#include "gpu/cuda/backend/gmflow_transformer_net.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

#include "prepare/gmflow_attention_native.h"
#include "prepare/gmflow_mlp.h"

namespace kirari::gpu::cuda {
namespace {

constexpr auto kBlocks = static_cast<std::size_t>(prepare::kGmflowBlocks);
constexpr std::size_t kBytesPerToken = prepare::kGmflowModel * 2;

bool aligned(const void* p) {
  return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

bool overlaps(const void* a, std::size_t na, const void* b, std::size_t nb) {
  const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
  return x <= y ? y - x < na : x - y < nb;
}

bool check_plans(const prepare::GmflowAttention& attention, const prepare::GmflowMlp& mlp,
                 std::string& error) {
  if (attention.tensors != 1 + 3 * prepare::kGmflowBlocks || mlp.layers.size() != kBlocks ||
      mlp.uses.size() != 2 * kBlocks) {
    error = "transformeur de GMFlow : six blocs attendus aux deux échelles";
    return false;
  }
  if (auto checked = prepare::gmflow_attention_check(attention); !checked) {
    error = std::move(checked.error().detail);
    return false;
  }
  if (auto checked = prepare::gmflow_mlp_check(mlp); !checked) {
    error = std::move(checked.error().detail);
    return false;
  }
  int in = attention.input;
  for (std::size_t block = 0; block < kBlocks; ++block) {
    if (attention.ops[3 * block].query != in) {
      error = "transformeur de GMFlow : un bloc ne lit pas la sortie du précédent";
      return false;
    }
    in = attention.ops[3 * block + 2].out;
  }
  return true;
}

}  // namespace

GmflowTransformerNet::GmflowTransformerNet(std::unique_ptr<GmflowAttentionNet> attention,
                                           std::unique_ptr<GmflowMlpNet> mlp, long tokens,
                                           std::size_t token_bytes)
    : attention_(std::move(attention)), mlp_(std::move(mlp)), tokens_(tokens), token_bytes_(token_bytes) {}

std::unique_ptr<GmflowTransformerNet> GmflowTransformerNet::create(const prepare::GmflowAttention& attention,
                                                                   const prepare::GmflowMlp& mlp, int scale,
                                                                   int height, int width,
                                                                   GmflowMlpBlock block, std::string& error,
                                                                   bool fast) {
  const std::uint64_t batch = scale == 0 ? 2 : 4;
  const auto area = static_cast<std::uint64_t>(height) * static_cast<std::uint64_t>(width);
  // La longueur d'une fenêtre est un int, les jetons du MLP sont des long et
  // ses lancements les arrondissent.
  const auto max_tokens = std::min<std::uint64_t>(
      std::numeric_limits<long>::max() - 128, std::numeric_limits<std::size_t>::max() / (3 * kBytesPerToken));
  if (scale < 0 || scale > 1 || height < 1 || width < 1 || area > std::numeric_limits<int>::max() ||
      area > max_tokens / batch) {
    error = "transformeur de GMFlow : échelle ou grille de jetons invalide";
    return nullptr;
  }
  if (!check_plans(attention, mlp, error)) return nullptr;
  auto attn =
      GmflowAttentionNet::create(attention, scale, static_cast<int>(batch), height, width, error, fast);
  if (!attn) return nullptr;
  const auto tokens = static_cast<long>(batch * area);
  auto mlp_net = GmflowMlpNet::create(mlp, tokens, block, error, fast);
  if (!mlp_net) return nullptr;
  const auto bytes = static_cast<std::size_t>(tokens) * kBytesPerToken;
  return std::unique_ptr<GmflowTransformerNet>(
      new GmflowTransformerNet(std::move(attn), std::move(mlp_net), tokens, bytes));
}

cudaError_t GmflowTransformerNet::run(cudaStream_t stream, const void* in, void* out, void* scratch) const {
  if (!aligned(in) || !aligned(out) || !aligned(scratch) || overlaps(in, token_bytes_, out, token_bytes_) ||
      overlaps(scratch, scratch_bytes(), in, token_bytes_) ||
      overlaps(scratch, scratch_bytes(), out, token_bytes_))
    return cudaErrorInvalidValue;

  void* const self = scratch;
  void* const message = static_cast<std::byte*>(scratch) + token_bytes_;
  const void* current = in;
  for (int block = 0; block < prepare::kGmflowBlocks; ++block) {
    if (const auto e = attention_->run_layer(stream, 2 * block, current, current, self); e != cudaSuccess)
      return e;
    if (const auto e = attention_->run_layer(stream, 2 * block + 1, self, current, message); e != cudaSuccess)
      return e;
    // out peut porter l'entrée du bloc : les deux couches d'attention ont fini
    // de la lire sur ce flux.
    if (const auto e = mlp_->run(stream, block, self, message, out, tokens_); e != cudaSuccess) return e;
    current = out;
  }
  return cudaSuccess;
}

}  // namespace kirari::gpu::cuda
