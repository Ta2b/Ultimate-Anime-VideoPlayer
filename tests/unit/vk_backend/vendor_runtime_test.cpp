// Sorties de référence par modèle. L'interpréteur de gpu_ir calcule une fois la sortie de chaque
// modèle de référence (SPAN et RIFE en petit, le vrai SPAN d'AnimeJaNai s'il est installé) pour
// une entrée déterministe, l'enregistre sous le dossier des références du build
// (AJP_TEST_REFERENCE_DIR) et la relit ensuite. Chaque runtime des réseaux la retrouve à la
// tolérance près : le moteur Vulkan sur l'appareil du système (lavapipe en CI), puis MIGraphX,
// OpenVINO et Core ML quand ce build les a et que leur matériel est là ; sans lui, le cas le dit
// et se saute. C'est la seule vérification numérique de ces runtimes, qui ne tournent que sur
// leur GPU.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "core/model_index.h"
#include "gpu/ir/ir.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_runtime.h"
#include "kernels/ref/half.h"
#include "net_rig.h"
#include "test_models.h"

#if defined(AJP_TEST_WITH_MIGRAPHX)
#include "gpu/migraphx/migraphx_runtime.h"
#endif
#if defined(AJP_TEST_WITH_OPENVINO)
#include "gpu/openvino/openvino_runtime.h"
#endif
#if defined(AJP_TEST_WITH_COREML)
#include "gpu/coreml/coreml_runtime.h"
#include "gpu/coreml/coreml_units.h"
#endif
#if defined(AJP_TEST_WITH_WINML)
#include "gpu/winml/winml_runtime.h"
#endif

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;

namespace {

// Les runtimes des fabricants n'accumulent pas comme l'interpréteur ni comme le moteur Vulkan
// (ordre des sommes, fusions, float16 partout) : 2 % relatifs et un centième absolu sur des
// images dans [0, 1], soit deux pas et demi de 8 bits.
constexpr Tolerance kVendor{.rtol = 2e-2F, .atol = 1e-2F};

struct ReferenceModel {
  ModelSpec spec;
  ir::Dims input;  // NCHW de l'entrée unique
  std::size_t seed = 0;
};

std::vector<ReferenceModel> reference_models(const TempDir& dir) {
  std::vector<ReferenceModel> out;
  out.push_back({.spec = write_model(dir.path(), "span_like_f16", span_like(true).bytes(), TensorType::F16),
                 .input = {1, 3, 24, 32},
                 .seed = 101});
  out.push_back({.spec = write_model(dir.path(), "rife_like_f16", rife_like(true).bytes(), TensorType::F16),
                 .input = {1, 6, 16, 24},
                 .seed = 102});
  const std::filesystem::path real(AJP_SPAN_MODEL);
  std::error_code error;
  if (std::filesystem::is_regular_file(real, error)) {
    const auto sha256 = core::sha256_file(real);
    REQUIRE(sha256);
    out.push_back({.spec = {.id = real.stem().string(), .onnx_path = real, .sha256 = *sha256},
                   .input = {1, 3, 32, 48},
                   .seed = 103});
  }
  return out;
}

std::size_t count_of(const ir::Dims& dims) {
  std::size_t count = 1;
  for (const std::int64_t d : dims) count *= static_cast<std::size_t>(d);
  return count;
}

std::string dims_text(const ir::Dims& dims) {
  std::string out;
  for (const std::int64_t d : dims) out += std::format("{}{}", out.empty() ? "" : "x", d);
  return out;
}

Shape shape_of(const ir::Dims& dims) {
  REQUIRE(dims.size() == 4);
  return Shape{{dims[0], dims[1], dims[2], dims[3]}};
}

// La référence d'un modèle : son graphe préparé comme le backend le prépare, l'entrée
// déterministe dans [0, 1[, et la sortie de l'interpréteur, enregistrée ou relue.
struct Reference {
  ir::Graph graph;
  ir::Value input;
  ir::Value output;
  bool recorded = false;  // calculée et écrite par cet appel
};

std::filesystem::path reference_path(const ReferenceModel& m) {
  return std::filesystem::path(AJP_TEST_REFERENCE_DIR) /
         std::format("{}-{}-{}-s{}-{}.f32", m.spec.id, m.spec.sha256.substr(0, 8), dims_text(m.input), m.seed,
                     m.spec.precision == TensorType::F16 ? "f16" : "f32");
}

Reference reference_for(const ReferenceModel& m, const vk::net::Target& target) {
  const bool fp16 = m.spec.precision == TensorType::F16;
  const vk::net::BuildRequest request{.onnx = m.spec.onnx_path,
                                      .model_id = m.spec.id,
                                      .model_sha256 = m.spec.sha256,
                                      .input_shape = m.input,
                                      .fp16 = fp16};
  auto prepared = vk::net::prepare_graph(request, target);
  if (!prepared && prepared.error().code == RefusalCode::FeatureUnavailable) SKIP(describe(prepared.error()));
  Reference ref{.graph = require_value(std::move(prepared))};
  REQUIRE(ref.graph.inputs.size() == 1);
  REQUIRE(ref.graph.outputs.size() == 1);
  const ir::Tensor& in = ref.graph.tensors[ref.graph.inputs[0]];
  const ir::Tensor& out = ref.graph.tensors[ref.graph.outputs[0]];
  REQUIRE(out.known);
  ref.input = test_values(in.type, m.input, m.seed, 0.0F, 1.0F);
  ref.output = ir::Value{.type = out.type, .dims = out.dims};
  const std::size_t count = count_of(out.dims);
  const std::filesystem::path path = reference_path(m);
  if (auto bytes = vk::net::read_file(path); bytes && bytes->size() == count * sizeof(float)) {
    ref.output.f.resize(count);
    std::memcpy(ref.output.f.data(), bytes->data(), bytes->size());
    return ref;
  }
  const std::vector<ir::NamedValue> inputs{{in.name, ref.input}};
  ref.output = run_reference(ref.graph, inputs)[0];
  REQUIRE(ref.output.f.size() == count);
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  std::vector<std::byte> bytes(count * sizeof(float));
  std::memcpy(bytes.data(), ref.output.f.data(), bytes.size());
  require_ok(vk::net::write_file(path, bytes));
  ref.recorded = true;
  return ref;
}

std::vector<std::byte> host_bytes(const ir::Value& value, TensorType type) {
  const std::size_t element = type == TensorType::F16 ? 2 : 4;
  std::vector<std::byte> out(value.f.size() * element);
  if (type == TensorType::F32) {
    std::memcpy(out.data(), value.f.data(), out.size());
    return out;
  }
  for (std::size_t k = 0; k < value.f.size(); ++k) {
    const std::uint16_t h = kernels::ref::float_to_half(value.f[k]);
    std::memcpy(out.data() + 2 * k, &h, 2);
  }
  return out;
}

ir::Value host_value(std::span<const std::byte> bytes, TensorType type, const ir::Tensor& tensor) {
  ir::Value out{.type = tensor.type, .dims = tensor.dims};
  const std::size_t element = type == TensorType::F16 ? 2 : 4;
  out.f.resize(bytes.size() / element);
  for (std::size_t k = 0; k < out.f.size(); ++k) {
    if (type == TensorType::F32) {
      std::memcpy(&out.f[k], bytes.data() + 4 * k, 4);
    } else {
      std::uint16_t h = 0;
      std::memcpy(&h, bytes.data() + 2 * k, 2);
      out.f[k] = kernels::ref::half_to_float(h);
    }
  }
  return out;
}

// Le modèle construit, chargé et exécuté deux fois par le runtime, sur des tenseurs hôte ; les
// deux passages doivent rendre les mêmes octets.
ir::Value run_on_vendor(vk::net::Runtime& runtime, const ReferenceModel& m, const Reference& ref,
                        const std::filesystem::path& dir) {
  const ir::Tensor& in = ref.graph.tensors[ref.graph.inputs[0]];
  const ir::Tensor& out = ref.graph.tensors[ref.graph.outputs[0]];
  const std::filesystem::path artifact =
      dir / std::format("{}-{}.{}", m.spec.id, runtime.name(), runtime.extension());
  Progress progress;
  const std::vector<vk::net::InputShape> shapes{{.name = in.name, .dims = m.input}};
  require_ok(runtime.build(m.spec, shapes, artifact, progress));
  auto net = require_value(runtime.load(artifact));
  REQUIRE(net->input_names().size() == 1);
  REQUIRE(net->output_names().size() == 1);
  const TensorType type = m.spec.precision;
  std::vector<std::byte> in_bytes = host_bytes(ref.input, type);
  std::vector<std::byte> out_bytes(count_of(out.dims) * (type == TensorType::F16 ? 2U : 4U));
  const vk::net::HostBinding input{
      .data = in_bytes.data(), .bytes = in_bytes.size(), .type = type, .shape = shape_of(m.input)};
  const vk::net::HostBinding output{
      .data = out_bytes.data(), .bytes = out_bytes.size(), .type = type, .shape = shape_of(out.dims)};
  require_ok(net->run({{input}}, {{output}}));
  const std::vector<std::byte> first = out_bytes;
  std::ranges::fill(out_bytes, std::byte{0});
  require_ok(net->run({{input}}, {{output}}));
  CHECK(out_bytes == first);
  return host_value(out_bytes, type, out);
}

struct Vendor {
  std::string_view name;
  std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> (*make)(const vk::DeviceCaps& caps);
};

std::vector<Vendor> vendors() {
  std::vector<Vendor> out;
#if defined(AJP_TEST_WITH_MIGRAPHX)
  out.push_back({"migraphx", &migraphx_rt::make_runtime});
#endif
#if defined(AJP_TEST_WITH_OPENVINO)
  out.push_back({"openvino", &openvino_rt::make_runtime});
#endif
#if defined(AJP_TEST_WITH_COREML)
  out.push_back({"coreml", &coreml_rt::make_runtime});
#endif
#if defined(AJP_TEST_WITH_WINML)
  out.push_back({"winml", &winml_rt::make_runtime});
#endif
  return out;
}

}  // namespace

TEST_CASE("Référence : le moteur Vulkan retrouve la sortie enregistrée de chaque modèle",
          "[vk_backend][reference]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  TempDir dir("reference");
  for (const ReferenceModel& m : reference_models(dir)) {
    INFO(m.spec.id);
    const Reference ref = reference_for(m, rig.target);
    // Enregistrée au premier passage, relue ensuite à l'identique.
    CHECK(std::filesystem::is_regular_file(reference_path(m)));
    const Reference again = reference_for(m, rig.target);
    CHECK_FALSE(again.recorded);
    CHECK(again.output.f == ref.output.f);
    const ir::Tensor& in = ref.graph.tensors[ref.graph.inputs[0]];
    const std::vector<ir::NamedValue> inputs{{in.name, ref.input}};
    const std::vector<ir::Value> got = run_on_device(rig, ref.graph, inputs);
    require_close(got[0], ref.output, kFp16, m.spec.id);
  }
}

TEST_CASE("Référence : chaque runtime du fabricant retrouve la sortie enregistrée de chaque modèle",
          "[vk_backend][reference][vendor]") {
  const std::vector<Vendor> candidates = vendors();
  if (candidates.empty())
    SKIP("aucun runtime de fabricant dans ce build (MIGraphX, OpenVINO, Core ML, Windows ML)");
  NetRig rig = make_net_rig();
  require_f16(rig);
  TempDir dir("reference-vendor");
  const std::vector<ReferenceModel> models = reference_models(dir);
  std::string absent;
  std::size_t ran = 0;
  for (const Vendor& vendor : candidates) {
    auto runtime = vendor.make(rig.context->caps());
    if (!runtime) {
      const RefusalCode code = runtime.error().code;
      if (code == RefusalCode::DeviceMismatch || code == RefusalCode::DeviceUnavailable ||
          code == RefusalCode::LibraryUnavailable) {
        absent +=
            std::format("{}{} : {}", absent.empty() ? "" : " ; ", vendor.name, describe(runtime.error()));
        continue;
      }
      FAIL(describe(runtime.error()));
    }
    for (const ReferenceModel& m : models) {
      INFO(vendor.name << " : " << m.spec.id << " sur " << (*runtime)->device_name());
      const Reference ref = reference_for(m, rig.target);
      const ir::Value got = run_on_vendor(**runtime, m, ref, dir.path());
      require_close(got, ref.output, kVendor, std::format("{} {}", vendor.name, m.spec.id));
      ++ran;
    }
  }
  if (ran == 0) SKIP("aucun matériel pour les runtimes de ce build : " << absent);
}

#if defined(AJP_TEST_WITH_COREML)
// Les valeurs de MLComputeUnits qu'ONNX Runtime accepte (coreml_options.cc) : un nom d'Apple
// (« MLComputeUnitsAll ») fait échouer la création de chaque session Core ML, sur tout Mac.
TEST_CASE("Core ML : unités de calcul sous les noms d'ONNX Runtime", "[vk_backend][coreml]") {
  using coreml_rt::Units;
  CHECK(coreml_rt::units_option(Units::All) == "ALL");
  CHECK(coreml_rt::units_option(Units::CpuAndGpu) == "CPUAndGPU");
  CHECK(coreml_rt::units_option(Units::CpuAndNeuralEngine) == "CPUAndNeuralEngine");
  for (const Units u : {Units::All, Units::CpuAndGpu, Units::CpuAndNeuralEngine}) {
    CHECK(coreml_rt::units_from(coreml_rt::units_name(u)) == u);
  }
  CHECK_FALSE(coreml_rt::units_from("ALL").has_value());
}
#endif
