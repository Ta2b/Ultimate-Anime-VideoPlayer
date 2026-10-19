// Runtime TensorRT sur la RTX : moteur construit dans le test depuis un petit
// ONNX, exécuté par enqueueV3 et comparé à l'interpréteur CPU de gpu_ir.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/cuda/trt/cuda_objects.h"
#include "support.h"

namespace trt = ajp::gpu::trt;
namespace ir = ajp::gpu::ir;
using trt::test::must;

namespace {

// Exécute le moteur sur une entrée NCHW et rend la sortie en float.
std::vector<float> run_engine(trt::TrtEngine& engine, const trt::CudaStream& stream, trt::DataType type,
                              const trt::Dims& dims, const std::vector<float>& input) {
  must(engine.set_input_shape("input", dims));
  const trt::Dims out_dims = must(engine.shape("output"));
  const std::vector<std::byte> host = trt::test::to_bytes(type, input);
  const trt::DeviceBuffer in = must(trt::DeviceBuffer::allocate(host.size()));
  const trt::DeviceBuffer out = must(trt::DeviceBuffer::allocate(must(trt::byte_count(type, out_dims))));
  must(trt::test::upload(in.data(), host));
  must(engine.bind("input", in.data()));
  must(engine.bind("output", out.data()));
  const trt::CudaEvent consumed = must(trt::CudaEvent::create());
  const trt::CudaEvent done = must(trt::CudaEvent::create());
  must(engine.enqueue(stream.get(), {.input_consumed = consumed.get(), .done = done.get()}));
  must(done.synchronize());
  CHECK(must(consumed.done()));
  return trt::test::to_floats(type, must(trt::test::download(out.data(), out.size())));
}

std::vector<float> run_reference(std::int32_t type, const ir::Dims& dims, const std::vector<float>& input) {
  const ir::Graph graph = ir::test::build(trt::test::conv_model(type), {{.name = "input", .dims = dims}});
  const ir::DataType value_type = type == ir::onnx::kFloat16 ? ir::DataType::F16 : ir::DataType::F32;
  return ir::test::execute1(graph, {{.name = "input", .value = ir::test::floats(dims, input, value_type)}}).f;
}

}  // namespace

TEST_CASE("le moteur décrit ses entrées, ses sorties et son profil", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  const trt::TrtRuntime runtime = trt::test::make_runtime();
  CHECK(runtime.version().major >= 11);
  const trt::TrtEngine engine = must(runtime.load(trt::test::conv_plan(runtime, ir::onnx::kFloat)));
  REQUIRE(engine.tensors().size() == 2);
  const trt::TensorDesc* input = engine.find("input");
  const trt::TensorDesc* output = engine.find("output");
  REQUIRE(input != nullptr);
  REQUIRE(output != nullptr);
  CHECK(engine.find("absent") == nullptr);
  CHECK(input->io == trt::TensorIo::Input);
  CHECK(output->io == trt::TensorIo::Output);
  CHECK(input->type == trt::DataType::F32);
  CHECK(input->dims == trt::Dims{1, 3, -1, -1});
  CHECK(engine.profile_count() == 1);
  CHECK(engine.profile() == 0);
  REQUIRE(input->profiles.size() == 1);
  CHECK(input->profiles[0].min == trt::test::kConvProfile.min);
  CHECK(input->profiles[0].opt == trt::test::kConvProfile.opt);
  CHECK(input->profiles[0].max == trt::test::kConvProfile.max);
  CHECK(output->profiles.empty());
  CHECK(engine.device_memory_bytes() >= 0);
}

TEST_CASE("FP32 : la sortie égale celle de l'interpréteur gpu_ir", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  const trt::TrtRuntime runtime = trt::test::make_runtime();
  trt::TrtEngine engine = must(runtime.load(trt::test::conv_plan(runtime, ir::onnx::kFloat)));
  const trt::CudaStream stream = must(trt::CudaStream::create());
  // Deux formes du profil, hors de opt : la forme se pose à chaque changement.
  for (const ir::Dims& dims : {ir::Dims{1, 3, 16, 24}, ir::Dims{1, 3, 8, 32}}) {
    const std::vector<float> input = ir::test::sequence(ir::test::count_of(dims), -1.0F, 1.0F, 7);
    const std::vector<float> got = run_engine(engine, stream, trt::DataType::F32, dims, input);
    const std::vector<float> expected = run_reference(ir::onnx::kFloat, dims, input);
    INFO("forme " << trt::dims_text(dims));
    CHECK(must(engine.shape("output")) == trt::Dims{1, 4, dims[2], dims[3]});
    CHECK(ir::test::max_abs_diff(got, expected) <= 1e-4F);
  }
}

TEST_CASE("FP16 : la sortie suit l'interpréteur à la précision du demi-flottant", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  const trt::TrtRuntime runtime = trt::test::make_runtime();
  trt::TrtEngine engine = must(runtime.load(trt::test::conv_plan(runtime, ir::onnx::kFloat16)));
  REQUIRE(engine.find("input") != nullptr);
  CHECK(engine.find("input")->type == trt::DataType::F16);
  const trt::CudaStream stream = must(trt::CudaStream::create());
  const ir::Dims dims{1, 3, 32, 32};
  // Entrée déjà au demi-flottant le plus proche : seul le calcul diffère.
  std::vector<float> input = ir::test::sequence(ir::test::count_of(dims), 0.0F, 1.0F, 3);
  input = trt::test::to_floats(trt::DataType::F16, trt::test::to_bytes(trt::DataType::F16, input));
  const std::vector<float> got = run_engine(engine, stream, trt::DataType::F16, dims, input);
  const std::vector<float> expected = run_reference(ir::onnx::kFloat16, dims, input);
  // Sorties de l'ordre de 1 : quelques ulp F16 (2^-10) d'accumulation.
  CHECK(ir::test::max_abs_diff(got, expected) <= 8e-3F);
}

TEST_CASE("chargement depuis un fichier, puis exécution", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  const trt::TrtRuntime runtime = trt::test::make_runtime();
  const std::vector<std::byte> plan = trt::test::conv_plan(runtime, ir::onnx::kFloat);
  const std::filesystem::path file = std::filesystem::temp_directory_path() / "ajp_cuda_trt_file_test.engine";
  {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(plan.data()), static_cast<std::streamsize>(plan.size()));
  }
  auto engine = runtime.load_file(file);
  std::filesystem::remove(file);
  REQUIRE(engine);
  const trt::CudaStream stream = must(trt::CudaStream::create());
  const ir::Dims dims{1, 3, 8, 8};
  const std::vector<float> input(ir::test::count_of(dims), 0.25F);
  CHECK(ir::test::max_abs_diff(run_engine(*engine, stream, trt::DataType::F32, dims, input),
                               run_reference(ir::onnx::kFloat, dims, input)) <= 1e-4F);
}

TEST_CASE("les moteurs survivent à leur runtime", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  auto engine = [] {
    const trt::TrtRuntime runtime = trt::test::make_runtime();
    return must(runtime.load(trt::test::conv_plan(runtime, ir::onnx::kFloat)));
  }();
  const trt::CudaStream stream = must(trt::CudaStream::create());
  const ir::Dims dims{1, 3, 8, 8};
  const std::vector<float> input(ir::test::count_of(dims), -0.5F);
  CHECK(ir::test::max_abs_diff(run_engine(engine, stream, trt::DataType::F32, dims, input),
                               run_reference(ir::onnx::kFloat, dims, input)) <= 1e-4F);
}

TEST_CASE("un seul runtime vit à la fois : TensorRT n'a qu'un journal", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  {
    const trt::TrtRuntime first = trt::test::make_runtime();
    auto second = trt::TrtRuntime::create(trt::make_logger());
    REQUIRE_FALSE(second);
    CHECK(second.error().code == ajp::RefusalCode::TensorRtFailed);
    CHECK(second.error().detail.find("un par processus") != std::string::npos);
  }
  // Le premier détruit, un nouveau runtime reprend le journal.
  CHECK(trt::TrtRuntime::create(trt::make_logger()).has_value());
}
