// RIFE fusionné sur TensorRT : l'encodage gardé d'un appel à l'autre (gpu::Engine::next_times_only, un autre
// instant de la même paire ; next_pair_follows, B reprise comme A à la paire suivante) rend les octets exacts
// de la chaîne qui encode les deux images à chaque appel. Trois images, deux paires dans deux tampons
// d'entrée, deux instants par paire ; le parcours trois fois de suite (exécution directe, capture du graphe
// CUDA, rejeu). Un autre moteur fusionné passé entre deux appels, dans la même zone d'activation, efface
// l'encodage gardé : l'annonce qui suit encode de nouveau, la sortie reste exacte. Test caché, sur la carte
// NVIDIA et les modèles du lecteur : ajp_backend_tensorrt_tests "[.rife_fused]".
#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "prepare/rife_split.h"
#include "prepare/sha256.h"

namespace ajp::test {
namespace {

namespace fs = std::filesystem;

std::vector<float> drawing(int h, int w, int shift) {
  std::vector<float> plane(static_cast<std::size_t>(3 * h * w));
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int u = std::max(0, x - shift);
        float v = 0.25F + 0.1F * static_cast<float>(c);
        if (u > w / 3 && u < 2 * w / 3 && y > h / 4) v = 0.7F - 0.1F * static_cast<float>(c);
        if (y > h / 2) v += 0.25F * static_cast<float>(u) / static_cast<float>(w);
        if ((u + 2 * y) % 97 < 2) v = 0.05F;
        plane[static_cast<std::size_t>((c * h + y) * w + x)] = std::clamp(v, 0.F, 1.F);
      }
    }
  }
  return plane;
}

// Les octets FP16 d'un tenseur, relus sur l'hôte.
std::vector<std::uint16_t> bytes_of(const gpu::Tensor& tensor) {
  std::size_t count = 1;
  for (const auto d : tensor.shape.dims) count *= static_cast<std::size_t>(d);
  std::vector<std::uint16_t> half(count);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaMemcpy(half.data(), reinterpret_cast<const void*>(tensor.handle), count * 2,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return half;
}

std::unique_ptr<gpu::Engine> fused_engine(gpu::Backend& backend, const fs::path& onnx, const fs::path& dir,
                                          int h, int w, bool motion) {
  const auto manifest = prepare::derive_rife_fused(
      onnx, dir, h, w, motion ? prepare::RifeFusedOutput::Motion : prepare::RifeFusedOutput::Image);
  REQUIRE(manifest);
  const auto sha = prepare::sha256_file(*manifest);
  REQUIRE(sha);
  const gpu::Shape shape{.dims = {1, 11, h, w}};
  gpu::Progress progress;
  auto artifact =
      backend.build(gpu::ModelSpec{.id = manifest->stem().string(), .onnx_path = *manifest, .sha256 = *sha},
                    gpu::ShapeSet{.min = shape, .opt = shape, .max = shape}, progress);
  REQUIRE(artifact);
  auto engine = backend.load(*artifact);
  REQUIRE(engine);
  return std::move(*engine);
}

void kept_encoding_is_exact(const std::string& file, bool motion) {
  const fs::path onnx = fs::path(AJP_RIFE_MODEL_DIR) / file;
  if (!fs::is_regular_file(onnx)) SKIP("modèle absent : " << onnx.string());
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  REQUIRE(loaded.backend->capabilities().rife_fused);
  gpu::Backend& backend = *loaded.backend;
  constexpr int h = 384;
  constexpr int w = 640;
  const fs::path scratch = fs::temp_directory_path() / "ajp-rife-fused-encode-test";
  fs::create_directories(scratch);
  const std::unique_ptr<gpu::Engine> fused = fused_engine(backend, onnx, scratch, h, w, motion);
  // L'autre variante, qui partage la zone d'activation de la voie.
  const std::unique_ptr<gpu::Engine> other = fused_engine(backend, onnx, scratch, h, w, !motion);

  const gpu::Shape rgb{.dims = {1, 3, h, w}};
  std::array<std::unique_ptr<gpu::TensorBuffer>, 3> images;
  for (std::size_t i = 0; i < images.size(); ++i) {
    auto made = backend.alloc_tensor(gpu::TensorType::F16, rgb);
    REQUIRE(made);
    images[i] = std::move(*made);
    upload(images[i]->tensor(), drawing(h, w, 6 * static_cast<int>(i)));
  }
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> inputs;
  for (auto& input : inputs) {
    auto made = backend.alloc_tensor(gpu::TensorType::F16, gpu::Shape{.dims = {1, 11, h, w}});
    REQUIRE(made);
    input = std::move(*made);
  }
  const auto outputs_for = [&](bool flow) {
    std::vector<std::unique_ptr<gpu::TensorBuffer>> buffers;
    const std::vector<gpu::Shape> shapes =
        flow ? std::vector<gpu::Shape>{{.dims = {1, 4, h, w}}, {.dims = {1, 1, h, w}}}
             : std::vector<gpu::Shape>{rgb};
    for (const gpu::Shape& shape : shapes) {
      auto made = backend.alloc_tensor(gpu::TensorType::F16, shape);
      REQUIRE(made);
      buffers.push_back(std::move(*made));
    }
    return buffers;
  };
  const auto outs = outputs_for(motion);
  const auto other_outs = outputs_for(!motion);
  std::vector<gpu::Tensor> out_tensors;
  for (const auto& b : outs) out_tensors.push_back(b->tensor());
  std::vector<gpu::Tensor> other_tensors;
  for (const auto& b : other_outs) other_tensors.push_back(b->tensor());
  const auto read = [&] {
    std::vector<std::uint16_t> all;
    for (const gpu::Tensor& t : out_tensors) {
      const auto part = bytes_of(t);
      all.insert(all.end(), part.begin(), part.end());
    }
    return all;
  };

  enum class Say { Nothing, TimesOnly, Follows };
  // La paire (first, first + 1) dans le tampon buffer à l'instant t ; whole réécrit tout, sinon rife_time
  // n'écrit que l'instant.
  const auto call = [&](std::size_t first, std::size_t buffer, float t, bool whole, Say say) {
    const gpu::Tensor& input = inputs[buffer]->tensor();
    if (whole) {
      REQUIRE(backend.kernels()->rife_input(images[first]->tensor(), images[first + 1]->tensor(), t, input));
    } else {
      REQUIRE(backend.kernels()->rife_time(input, t));
    }
    if (say == Say::TimesOnly) fused->next_times_only();
    if (say == Say::Follows) fused->next_pair_follows();
    const std::array ins{input};
    REQUIRE(backend.enqueue(*fused, ins, out_tensors));
    return read();
  };
  struct Step {
    std::size_t first;
    std::size_t buffer;
    float t;
    bool whole;
    Say say;
  };
  const std::array<Step, 4> steps{
      Step{0, 0, 0.25F, true, Say::Nothing}, Step{0, 0, 0.75F, false, Say::TimesOnly},
      Step{1, 1, 0.5F, true, Say::Follows}, Step{1, 1, 0.25F, false, Say::TimesOnly}};
  std::vector<std::vector<std::uint16_t>> want;
  for (const Step& s : steps) want.push_back(call(s.first, s.buffer, s.t, true, Say::Nothing));
  CHECK(want[0] != want[1]);                 // l'instant change bien la sortie
  for (int round = 0; round < 3; ++round) {  // direct, capture, rejeu
    for (std::size_t i = 0; i < steps.size(); ++i) {
      const Step& s = steps[i];
      INFO("tour " << round << ", appel " << i);
      CHECK(call(s.first, s.buffer, s.t, s.whole, s.say) == want[i]);
    }
  }
  // L'autre moteur écrit dans la zone : l'encodage gardé ne vaut plus, l'annonce encode de nouveau.
  const std::array other_in{inputs[0]->tensor()};
  REQUIRE(backend.kernels()->rife_input(images[0]->tensor(), images[1]->tensor(), 0.5F, inputs[0]->tensor()));
  REQUIRE(backend.enqueue(*other, other_in, other_tensors));
  CHECK(call(1, 1, 0.5F, true, Say::Follows) == want[2]);
  REQUIRE(backend.enqueue(*other, other_in, other_tensors));
  CHECK(call(1, 1, 0.25F, false, Say::TimesOnly) == want[3]);
}

}  // namespace

TEST_CASE("RIFE fusionné : l'encodage repris d'un appel à l'autre rend les mêmes octets sur TensorRT",
          "[.rife_fused][backend_tensorrt][cuda]") {
  SECTION("4.22 lite, image") {
    kept_encoding_is_exact("rife_v4.22_lite.onnx", false);
  }
  SECTION("4.22 lite, flot et masque") {
    kept_encoding_is_exact("rife_v4.22_lite.onnx", true);
  }
  SECTION("4.25, image") {
    kept_encoding_is_exact("rife_v4.25.onnx", false);
  }
}

}  // namespace ajp::test
