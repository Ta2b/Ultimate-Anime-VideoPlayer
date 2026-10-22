// Le runtime Core ML sur un Mac, sans GPU (la machine virtuelle de la CI : Core ML y calcule sur
// le CPU) : un petit réseau construit, chargé et exécuté en direct par Core ML (coreml_direct.h),
// qui retrouve l'interpréteur de gpu_ir et le même artefact exécuté par ONNX Runtime, sur la
// mémoire ordinaire puis sur les IOSurface que le runtime donne à la chaîne. Les noms d'entrée et
// de sortie ne sont pas des identifiants du programme ML : Core ML direct les retrouve sous les
// noms qu'ONNX Runtime leur donne.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/test_util.h"
#include "gpu/coreml/coreml_runtime.h"
#include "gpu/vulkan/backend/net_runtime.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::coreml_rt {
namespace {

namespace irt = ir::test;

// Des lignes de 128 o : ni le système ni Core ML n'ont à bourrer celles des IOSurface.
constexpr std::int64_t kSide = 64;
constexpr std::size_t kCount = 3 * kSide * kSide;

struct TempDir {
  explicit TempDir(const char* test)
      : path(std::filesystem::temp_directory_path() / std::format("ajp-coreml-rt-{}", test)) {
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::filesystem::path path;
};

// Conv 3 × 3 de 3 vers 3 canaux, bord compris, puis Relu, en FP32 : « input.1 » vers « out:0 ».
irt::ModelBuilder small_net() {
  irt::ModelBuilder m;
  m.input("input.1", {1, 3, kSide, kSide})
      .weight("w", {3, 3, 3, 3}, irt::sequence(81, -0.3F, 0.3F, 7))
      .weight("b", {3}, {0.05F, -0.02F, 0.1F})
      .node("Conv", {"input.1", "w", "b"}, {"t"},
            {irt::attr_ints("kernel_shape", {3, 3}), irt::attr_ints("pads", {1, 1, 1, 1})})
      .node("Relu", {"t"}, {"out:0"})
      .output("out:0");
  return m;
}

std::string read_text(const std::filesystem::path& file) {
  std::ifstream in(file);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool has_line(const std::string& text, const std::string& line) {
  std::istringstream lines(text);
  for (std::string l; std::getline(lines, l);) {
    if (l == line) return true;
  }
  return false;
}

const Shape kShape{.dims = {1, 3, kSide, kSide}};

// Une exécution du réseau chargé, de l'entrée FP16 à in vers la sortie FP16 à out : la sortie.
std::vector<std::uint16_t> run_at(vk::net::RuntimeNet& net, std::byte* in_data, std::byte* out_data) {
  std::memset(out_data, 0xFF, kCount * 2);
  const vk::net::HostBinding in{
      .data = in_data, .bytes = kCount * 2, .type = TensorType::F16, .shape = kShape};
  const vk::net::HostBinding out{
      .data = out_data, .bytes = kCount * 2, .type = TensorType::F16, .shape = kShape};
  auto ok = net.run({{in}}, {{out}});
  if (!ok) FAIL(ok.error().detail);
  std::vector<std::uint16_t> output(kCount);
  std::memcpy(output.data(), out_data, kCount * 2);
  return output;
}

// La même, sur la mémoire ordinaire.
std::vector<std::uint16_t> run_once(vk::net::RuntimeNet& net, std::vector<std::uint16_t>& input) {
  std::vector<std::uint16_t> output(kCount);
  return run_at(net, reinterpret_cast<std::byte*>(input.data()), reinterpret_cast<std::byte*>(output.data()));
}

std::vector<float> as_floats(const std::vector<std::uint16_t>& half) {
  std::vector<float> out(half.size());
  for (std::size_t i = 0; i < half.size(); ++i) out[i] = kernels::ref::half_to_float(half[i]);
  return out;
}

}  // namespace

TEST_CASE("Core ML : un réseau exécuté en direct retrouve l'interpréteur et ONNX Runtime", "[coreml]") {
  TempDir dir("direct");
  const irt::ModelBuilder model = small_net();
  const std::filesystem::path onnx = dir.path / "petit.onnx";
  {
    const std::vector<std::byte> bytes = model.bytes();
    std::ofstream(onnx, std::ios::binary)
        .write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  vk::DeviceCaps caps;
  caps.vendor_id = 0x106B;
  caps.device_name = "Mac de test";
  auto runtime = make_runtime(caps);
  if (!runtime) FAIL(runtime.error().detail);

  const ModelSpec spec{.id = "petit", .onnx_path = onnx, .sha256 = {}, .precision = TensorType::F16};
  const std::vector<vk::net::InputShape> shapes{{.name = "input.1", .dims = {1, 3, kSide, kSide}}};
  const std::filesystem::path artifact = dir.path / "petit.coreml";
  Progress progress;
  if (auto built = (*runtime)->build(spec, shapes, artifact, progress); !built) FAIL(built.error().detail);
  const std::vector<std::string> inputs{"input.1"};
  const std::vector<std::string> outputs{"out:0"};
  REQUIRE(vk::net::write_io_names(artifact, inputs, outputs));
  // Un seul morceau confié à Core ML, sa sortie conforme, sur la mémoire ordinaire et sur des
  // IOSurface : le chemin direct est retenu, avec elles.
  const std::string manifest = read_text(artifact);
  CHECK(has_line(manifest, "direct"));
  CHECK(has_line(manifest, "surfaces"));
  CHECK(std::filesystem::is_directory(std::filesystem::path(artifact.string() + ".mlmodelc")));

  // L'entrée dans [0, 1[, en FP16 ; la référence de l'interpréteur sur les mêmes valeurs.
  std::vector<float> values = irt::sequence(kCount, 0.0F, 1.0F, 3);
  std::vector<std::uint16_t> input(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    input[i] = kernels::ref::float_to_half(values[i]);
    values[i] = kernels::ref::half_to_float(input[i]);
  }
  const ir::Graph graph = irt::build(model);
  const ir::Value expected = irt::execute1(graph, {{"input.1", irt::floats({1, 3, kSide, kSide}, values)}});

  auto direct = (*runtime)->load(artifact);
  if (!direct) FAIL(direct.error().detail);
  const std::vector<std::uint16_t> first = run_once(**direct, input);
  CHECK(irt::max_abs_diff(as_floats(first), expected.f) <= 2e-2F);
  // Les enveloppes gardées d'un appel à l'autre : la même sortie.
  CHECK(run_once(**direct, input) == first);

  // Sur les IOSurface du runtime, comme la chaîne les reçoit (Runtime::allocate_host) : la même
  // sortie, deux fois de suite.
  {
    auto in_surface = (*runtime)->allocate_host(kCount * 2, 16384, TensorType::F16, kShape);
    if (!in_surface) FAIL(in_surface.error().detail);
    auto out_surface = (*runtime)->allocate_host(kCount * 2, 16384, TensorType::F16, kShape);
    if (!out_surface) FAIL(out_surface.error().detail);
    std::memcpy((*in_surface)->data(), input.data(), kCount * 2);
    for (int pass = 0; pass < 2; ++pass) {
      const std::vector<std::uint16_t> on_surfaces =
          run_at(**direct, (*in_surface)->data(), (*out_surface)->data());
      CHECK(irt::max_abs_diff(as_floats(on_surfaces), as_floats(first)) <= 1e-3F);
    }
  }

  // Le même artefact sans le chemin direct : ONNX Runtime, la même sortie.
  {
    std::string rewritten;
    std::istringstream lines(manifest);
    for (std::string l; std::getline(lines, l);) {
      if (l != "direct") rewritten += l + '\n';
    }
    std::ofstream(artifact, std::ios::trunc) << rewritten;
  }
  auto through_ort = (*runtime)->load(artifact);
  if (!through_ort) FAIL(through_ort.error().detail);
  const std::vector<std::uint16_t> by_ort = run_once(**through_ort, input);
  CHECK(irt::max_abs_diff(as_floats(by_ort), as_floats(first)) <= 1e-3F);
}

}  // namespace ajp::gpu::coreml_rt
