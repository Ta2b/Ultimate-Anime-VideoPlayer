// RIFE fusionné sur TensorRT : l'encodage gardé d'un appel à l'autre (gpu::Engine::next_times_only, un autre
// instant de la même paire ; next_pair_follows, B reprise comme A à la paire suivante) rend les octets exacts
// de la chaîne qui encode les deux images à chaque appel. Trois images, deux paires dans deux tampons
// d'entrée, deux instants par paire ; le parcours trois fois de suite (exécution directe, capture du graphe
// CUDA, rejeu). Un autre moteur fusionné passé entre deux appels, dans la même zone d'activation, efface
// l'encodage gardé : l'annonce qui suit encode de nouveau, la sortie reste exacte. Test caché, sur la carte
// NVIDIA et les modèles du lecteur : ajp_backend_tensorrt_tests "[.rife_fused]". Un SPAN sur la même voie,
// lui, n'écrit que sous les tampons gardés (C03-01) : ils servent encore.
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
#include "gpu/cuda/backend/rife_fused_engine.h"
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
  // L'autre moteur fusionné de la zone a sa propre région (SharedActivations::region_offset) : il passe entre
  // deux appels sans toucher à l'encodage gardé, que l'appel suivant reprend tel quel (encodeur sauté).
  auto* engine = static_cast<gpu::cuda::RifeFusedEngine*>(fused.get());
  const std::array other_in{inputs[0]->tensor()};
  REQUIRE(backend.kernels()->rife_input(images[0]->tensor(), images[1]->tensor(), 0.5F, inputs[0]->tensor()));
  for (const float t : {0.5F, 0.25F}) {
    REQUIRE(backend.enqueue(*other, other_in, other_tensors));
    const auto before = engine->encode_counts();
    INFO("après l'autre moteur, instant " << t);
    CHECK(call(1, 1, t, false, Say::TimesOnly) == want[t == 0.5F ? 2 : 3]);
    CHECK(engine->encode_counts().none == before.none + 1);
  }
}

// C03-01 : sur une seule voie, SPAN et RIFE partagent la zone d'activation. Un SPAN passé entre deux appels
// n'écrit que sous les tampons de RIFE : l'encodage gardé sert encore (compté par le moteur), et les octets
// restent ceux de la chaîne qui encode tout.
void kept_encoding_survives_span(const std::string& file) {
  const fs::path onnx = fs::path(AJP_RIFE_MODEL_DIR) / file;
  const fs::path span_model(AJP_SPAN_MODEL);
  if (!fs::is_regular_file(onnx)) SKIP("modèle absent : " << onnx.string());
  if (!fs::is_regular_file(span_model)) SKIP("SPAN absent : " << span_model.string());
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  REQUIRE(loaded.backend->capabilities().rife_fused);
  gpu::Backend& backend = *loaded.backend;
  backend.use_lane(gpu::Lane::Upscale);  // tout sur la voie Upscale : une seule zone, comme --one-lane
  constexpr int h = 384;
  constexpr int w = 640;
  const fs::path scratch = fs::temp_directory_path() / "ajp-rife-fused-encode-test";
  fs::create_directories(scratch);
  const std::unique_ptr<gpu::Engine> fused = fused_engine(backend, onnx, scratch, h, w, false);
  const auto span_sha = prepare::sha256_file(span_model);
  REQUIRE(span_sha);
  const gpu::Shape span_in{.dims = {1, 3, h / 2, w / 2}};
  gpu::Progress progress;
  auto span_artifact = backend.build(
      gpu::ModelSpec{.id = span_model.stem().string(), .onnx_path = span_model, .sha256 = *span_sha},
      gpu::ShapeSet{.min = span_in, .opt = span_in, .max = span_in}, progress);
  REQUIRE(span_artifact);
  auto span = backend.load(*span_artifact);
  REQUIRE(span);
  auto span_input = backend.alloc_tensor(gpu::TensorType::F16, span_in);
  auto span_output = backend.alloc_tensor(gpu::TensorType::F16, gpu::Shape{.dims = {1, 3, h, w}});
  REQUIRE(span_input);
  REQUIRE(span_output);
  const std::array span_ins{(*span_input)->tensor()};
  std::array span_outs{(*span_output)->tensor()};
  const auto run_span = [&] { REQUIRE(backend.enqueue(**span, span_ins, span_outs)); };

  const gpu::Shape rgb{.dims = {1, 3, h, w}};
  std::array<std::unique_ptr<gpu::TensorBuffer>, 3> images;
  for (std::size_t i = 0; i < images.size(); ++i) {
    auto made = backend.alloc_tensor(gpu::TensorType::F16, rgb);
    REQUIRE(made);
    images[i] = std::move(*made);
    upload(images[i]->tensor(), drawing(h, w, 6 * static_cast<int>(i)));
  }
  auto input = backend.alloc_tensor(gpu::TensorType::F16, gpu::Shape{.dims = {1, 11, h, w}});
  auto output = backend.alloc_tensor(gpu::TensorType::F16, rgb);
  REQUIRE(input);
  REQUIRE(output);
  const std::array ins{(*input)->tensor()};
  std::array outs{(*output)->tensor()};
  const auto call = [&](float t, bool times_only) {
    if (times_only) {
      REQUIRE(backend.kernels()->rife_time((*input)->tensor(), t));
      fused->next_times_only();
    } else {
      REQUIRE(backend.kernels()->rife_input(images[0]->tensor(), images[1]->tensor(), t, (*input)->tensor()));
    }
    REQUIRE(backend.enqueue(*fused, ins, outs));
    return bytes_of((*output)->tensor());
  };
  // Le SPAN d'abord : ses activations déclarées, les tampons de RIFE montent au-dessus.
  run_span();
  const auto want = call(0.75F, false);
  auto* engine = static_cast<gpu::cuda::RifeFusedEngine*>(fused.get());
  for (int round = 0; round < 3; ++round) {  // direct, capture, rejeu
    INFO("tour " << round);
    static_cast<void>(call(0.25F, false));
    run_span();  // l'image source agrandie, entre deux instants de la même paire
    const auto before = engine->encode_counts();
    CHECK(call(0.75F, true) == want);
    CHECK(engine->encode_counts().none == before.none + 1);  // l'encodeur sauté malgré le SPAN
  }
}

}  // namespace

TEST_CASE("RIFE fusionné : un SPAN sur la même voie n'efface pas l'encodage gardé (C03-01)",
          "[.rife_fused][backend_tensorrt][cuda]") {
  kept_encoding_survives_span("rife_v4.22_lite.onnx");
}

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
  // C03-09 : les autres modèles des niveaux d'interpolation (4.25 lite : réduction 32, remplissage 128).
  SECTION("4.25 lite, image") {
    kept_encoding_is_exact("rife_v4.25_lite.onnx", false);
  }
  SECTION("4.26, image") {
    kept_encoding_is_exact("rife_v4.26.onnx", false);
  }
}

}  // namespace ajp::test
