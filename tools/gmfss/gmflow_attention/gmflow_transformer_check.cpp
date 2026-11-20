// Le transformeur GMFlow assemblé contre les mêmes couches avec des sorties alternées.
// gmflow_transformer_check PAIRE.onnx [0|1] HAUTEUR LARGEUR (grille des jetons).
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/gmflow_transformer_net.h"
#include "kernels/ref/half.h"
#include "prepare/gmflow_attention_native.h"
#include "prepare/gmflow_mlp.h"

namespace {

using namespace kirari::gpu::cuda;

int number(std::string_view value) {
  int n = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), n);
  if (error != std::errc{} || end != value.data() + value.size() || n < 0)
    throw std::runtime_error("entier positif ou nul attendu");
  return n;
}

std::vector<std::byte> read_pair(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  const auto bytes = in.tellg();
  if (bytes <= 0) throw std::runtime_error("paire ONNX illisible");
  std::vector<std::byte> pair(static_cast<std::size_t>(bytes));
  in.seekg(0);
  if (!in.read(reinterpret_cast<char*>(pair.data()), bytes))
    throw std::runtime_error("paire ONNX incomplète");
  return pair;
}

void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct Device {
  explicit Device(std::size_t bytes) { checked(cudaMalloc(&p, bytes)); }
  ~Device() { static_cast<void>(cudaFree(p)); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void* p = nullptr;
};

void run(const std::filesystem::path& path, int scale, int height, int width) {
  const auto pair = read_pair(path);
  const auto attention = kirari::prepare::gmflow_attention(pair);
  if (!attention) throw std::runtime_error(attention.error().detail);
  const auto mlp = kirari::prepare::gmflow_mlp(pair);
  if (!mlp) throw std::runtime_error(mlp.error().detail);
  constexpr auto kBlock = GmflowMlpBlock::k64;
  std::string error;
  auto transformer = GmflowTransformerNet::create(*attention, *mlp, scale, height, width, kBlock, error);
  if (!transformer) throw std::runtime_error(error);
  const int batch = scale == 0 ? 2 : 4;
  const long tokens = static_cast<long>(height) * width * batch;
  const auto bytes = transformer->scratch_bytes() / 2;
  auto attn_ref = GmflowAttentionNet::create(*attention, scale, batch, height, width, error);
  if (!attn_ref) throw std::runtime_error(error);
  auto mlp_ref = GmflowMlpNet::create(*mlp, tokens, kBlock, error);
  if (!mlp_ref) throw std::runtime_error(error);
  std::vector<std::uint16_t> values(bytes / sizeof(std::uint16_t));
  for (std::size_t i = 0; i < values.size(); ++i) {
    const float value = static_cast<float>(static_cast<int>(i % 97) - 48) / 64.0F;
    values[i] = kirari::kernels::ref::float_to_half(value);
  }
  const Device input(bytes), tested(bytes), scratch(transformer->scratch_bytes());
  const Device self(bytes), message(bytes), ref0(bytes), ref1(bytes);
  checked(cudaMemcpy(input.p, values.data(), bytes, cudaMemcpyHostToDevice));
  checked(transformer->run(nullptr, input.p, tested.p, scratch.p));

  const void* current = input.p;
  void* dest = ref0.p;
  void* alternate = ref1.p;
  for (int block = 0; block < kirari::prepare::kGmflowBlocks; ++block) {
    checked(attn_ref->run_layer(nullptr, 2 * block, current, current, self.p));
    checked(attn_ref->run_layer(nullptr, 2 * block + 1, self.p, current, message.p));
    checked(mlp_ref->run(nullptr, block, self.p, message.p, dest, tokens));
    current = dest;
    std::swap(dest, alternate);
  }
  std::vector<std::uint16_t> want(values.size()), got(values.size());
  checked(cudaMemcpy(want.data(), current, bytes, cudaMemcpyDeviceToHost));
  checked(cudaMemcpy(got.data(), tested.p, bytes, cudaMemcpyDeviceToHost));
  if (want != got) throw std::runtime_error("écart de bits du transformeur assemblé");
  for (const auto value : got)
    if ((value & 0x7c00U) == 0x7c00U) throw std::runtime_error("sortie FP16 non finie");
  std::cout << "six blocs, " << tokens << " jetons, sortie identique, " << transformer->scratch_bytes()
            << " octets pour les deux tampons\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage : gmflow_transformer_check PAIRE.onnx ECHELLE HAUTEUR LARGEUR\n";
    return 2;
  }
  try {
    run(argv[1], number(argv[2]), number(argv[3]), number(argv[4]));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
