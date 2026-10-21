// Le backend Vulkan par son interface : construction d'un artefact depuis un ONNX, chargement,
// enqueue rejoué contre l'interpréteur, aller-retour et refus des artefacts, autotuner,
// opérateur inconnu refusé à la construction, appareil de rendu qui ne convient pas.
#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "backend_rig.h"
#include "gpu/ir/interpreter.h"
#include "gpu/vulkan/backend/net_artifact.h"
#include "gpu/vulkan/backend/net_autotune.h"
#include "gpu/vulkan/backend/net_build.h"
#include "net_rig.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
using ir::test::attr_int;
using ir::test::attr_ints;
using ir::test::attr_string;
using ir::test::ModelBuilder;

namespace {

// SPAN en petit, en float16 : conv 3x3 + Relu, conv 3x3, résiduel, pixel shuffle x2.
std::vector<std::byte> span_model(bool half) {
  ModelBuilder m;
  const std::int32_t type = half ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  m.input("input", {1, 3, -1, -1}, type);
  const auto weight = [&](const std::string& name, std::vector<std::int64_t> dims, std::size_t seed,
                          float scale) {
    std::size_t count = 1;
    for (const std::int64_t d : dims) count *= static_cast<std::size_t>(d);
    const std::vector<float> v =
        test_values(ir::DataType::F32, {static_cast<std::int64_t>(count)}, seed, -scale, scale).f;
    if (half) {
      m.initializer(ir::test::tensor_f16(name, dims, v));
    } else {
      m.initializer(ir::test::tensor_f32(name, dims, v));
    }
  };
  weight("w1", {16, 3, 3, 3}, 1, 0.3F);
  weight("b1", {16}, 2, 0.1F);
  weight("w2", {16, 16, 3, 3}, 3, 0.1F);
  weight("b2", {16}, 4, 0.1F);
  weight("w3", {12, 16, 3, 3}, 5, 0.1F);
  weight("b3", {12}, 6, 0.1F);
  const auto same = attr_ints("pads", {1, 1, 1, 1});
  m.node("Conv", {"input", "w1", "b1"}, {"c1"}, {same}).node("Relu", {"c1"}, {"r1"});
  m.node("Conv", {"r1", "w2", "b2"}, {"c2"}, {same}).node("Add", {"c2", "r1"}, {"s2"});
  m.node("Conv", {"s2", "w3", "b3"}, {"c3"}, {same});
  m.node("DepthToSpace", {"c3"}, {"output"}, {attr_int("blocksize", 2), attr_string("mode", "CRD")});
  m.output("output", type);
  return m.bytes();
}

ShapeSet shapes(std::int64_t h, std::int64_t w) {
  const Shape s{{1, 3, h, w}};
  return {.min = s, .opt = s, .max = s};
}

Artifact build(vk::VulkanBackend& backend, const ModelSpec& model, const ShapeSet& set) {
  Progress progress;
  return require_value(backend.build(model, set, progress));
}

}  // namespace

TEST_CASE("Backend : construction, chargement et enqueue d'un SPAN contre l'interpréteur",
          "[vk_backend][backend]") {
  TempDir dir("span");
  auto backend = make_backend(dir.path() / "cache",
                              {.autotune = true, .tuner = {.warmup = 1, .passes = 2, .repeats = 1}});
  const bool half = backend->profile().target.f16_storage;
  const TensorType type = half ? TensorType::F16 : TensorType::F32;
  const ModelSpec model = write_model(dir.path(), "span_test", span_model(half), type);
  std::vector<float> fractions;
  Progress progress{.report = [&](float f, const std::string&) { fractions.push_back(f); }};
  const Artifact artifact = require_value(backend->build(model, shapes(20, 28), progress));
  CHECK(std::filesystem::exists(artifact.path));
  CHECK_FALSE(fractions.empty());
  CHECK(fractions.back() == 1.0F);
  CHECK(std::filesystem::exists(dir.path() / "cache" / "vulkan" / "autotune.tsv"));

  auto engine = require_value(backend->load(artifact));
  REQUIRE(engine->input_names().size() == 1);
  CHECK(engine->input_names()[0] == "input");
  CHECK(engine->output_names()[0] == "output");

  const ir::Value x = test_values(half ? ir::DataType::F16 : ir::DataType::F32, {1, 3, 20, 28}, 7);
  auto in = require_value(backend->alloc_tensor(type, Shape{{1, 3, 20, 28}}));
  auto out = require_value(backend->alloc_tensor(type, Shape{{1, 3, 40, 56}}));
  kernels::RgbImage image(28, 20);
  for (std::size_t c = 0; c < 3; ++c) {
    std::copy_n(x.f.begin() + static_cast<std::ptrdiff_t>(c * 560), 560, image.planes[c].data.begin());
  }
  write_tensor(*backend, in->tensor(), rgb_bytes(image, type));
  std::array inputs{in->tensor()};
  std::array outputs{out->tensor()};
  // Deux passages : le second rejoue l'enregistrement du premier.
  require_ok(backend->enqueue(*engine, inputs, outputs));
  require_ok(backend->enqueue(*engine, inputs, outputs));
  const Fence fence = require_value(backend->signal());
  CHECK(fence.value == 2);
  require_ok(backend->wait(fence));

  const kernels::RgbImage got =
      rgb_from(read_tensor(*backend, out->tensor(), (half ? 2U : 4U) * 3 * 40 * 56), 56, 40, type);
  // La référence : l'interpréteur sur le même graphe optimisé.
  const vk::net::BuildRequest request{
      .onnx = model.onnx_path, .model_id = model.id, .input_shape = {1, 3, 20, 28}, .fp16 = half};
  const ir::Graph graph = require_value(vk::net::prepare_graph(request, backend->profile().target));
  const std::vector<ir::NamedValue> named{{"input", x}};
  const std::vector<ir::Value> want = run_reference(graph, named);
  ir::Value got_value{.type = want[0].type, .dims = want[0].dims};
  for (const auto& p : got.planes) got_value.f.insert(got_value.f.end(), p.data.begin(), p.data.end());
  require_close(got_value, want[0], half ? kFp16 : kFp32, "SPAN");

  // Mauvais tenseur : refus nommé, rien de soumis.
  auto wrong = require_value(backend->alloc_tensor(type, Shape{{1, 3, 20, 30}}));
  std::array bad{wrong->tensor()};
  require_refusal(backend->enqueue(*engine, bad, outputs), RefusalCode::InputMismatch);
  CHECK(require_value(backend->signal()).value == 2);

  // Un second build trouve l'artefact valable en cache.
  const auto again = build(*backend, model, shapes(20, 28));
  CHECK(again.path == artifact.path);
  CHECK(again.cache_key == artifact.cache_key);
}

// Plus de huit entrées et sorties (la paire de GMFSS en a 15) : onze convolutions 1x1 d'une même
// entrée, chacune une sortie, rejouées contre l'interpréteur.
TEST_CASE("Backend : enqueue d'un graphe à onze sorties", "[vk_backend][backend]") {
  TempDir dir("wide");
  auto backend = make_backend(dir.path() / "cache");
  constexpr int kOutputs = 11;
  ModelBuilder m;
  m.input("input", {1, 3, -1, -1}, ir::onnx::kFloat);
  for (int k = 0; k < kOutputs; ++k) {
    const std::string w = std::format("w{}", k);
    const std::string b = std::format("b{}", k);
    const auto seed = static_cast<std::size_t>(k);
    m.initializer(ir::test::tensor_f32(w, {3, 3, 1, 1}, test_values(ir::DataType::F32, {9}, 10 + seed).f));
    m.initializer(ir::test::tensor_f32(b, {3}, test_values(ir::DataType::F32, {3}, 40 + seed).f));
    m.node("Conv", {"input", w, b}, {std::format("out{}", k)});
    m.output(std::format("out{}", k), ir::onnx::kFloat);
  }
  const ModelSpec model = write_model(dir.path(), "wide_test", m.bytes(), TensorType::F32);
  auto engine = require_value(backend->load(build(*backend, model, shapes(6, 10))));
  REQUIRE(engine->output_names().size() == kOutputs);

  const ir::Value x = test_values(ir::DataType::F32, {1, 3, 6, 10}, 3);
  auto in = require_value(backend->alloc_tensor(TensorType::F32, Shape{{1, 3, 6, 10}}));
  kernels::RgbImage image(10, 6);
  for (std::size_t c = 0; c < 3; ++c) {
    std::copy_n(x.f.begin() + static_cast<std::ptrdiff_t>(c * 60), 60, image.planes[c].data.begin());
  }
  write_tensor(*backend, in->tensor(), rgb_bytes(image, TensorType::F32));
  std::vector<std::unique_ptr<TensorBuffer>> buffers;
  std::vector<Tensor> outputs;
  for (int k = 0; k < kOutputs; ++k) {
    buffers.push_back(require_value(backend->alloc_tensor(TensorType::F32, Shape{{1, 3, 6, 10}})));
    outputs.push_back(buffers.back()->tensor());
  }
  std::array inputs{in->tensor()};
  require_ok(backend->enqueue(*engine, inputs, outputs));
  require_ok(backend->wait(require_value(backend->signal())));

  const vk::net::BuildRequest request{
      .onnx = model.onnx_path, .model_id = model.id, .input_shape = {1, 3, 6, 10}, .fp16 = false};
  const ir::Graph graph = require_value(vk::net::prepare_graph(request, backend->profile().target));
  const std::vector<ir::NamedValue> named{{"input", x}};
  const std::vector<ir::Value> want = run_reference(graph, named);
  REQUIRE(want.size() == kOutputs);
  for (std::size_t k = 0; k < want.size(); ++k) {
    const kernels::RgbImage got =
        rgb_from(read_tensor(*backend, outputs[k], 4U * 3 * 6 * 10), 10, 6, TensorType::F32);
    ir::Value got_value{.type = want[k].type, .dims = want[k].dims};
    for (const auto& p : got.planes) got_value.f.insert(got_value.f.end(), p.data.begin(), p.data.end());
    require_close(got_value, want[k], kFp32, std::format("sortie {}", k));
  }
}

TEST_CASE("Backend : artefact, aller-retour et refus", "[vk_backend][backend]") {
  TempDir dir("artefact");
  auto backend = make_backend(dir.path() / "cache", {.autotune = false});
  const ModelSpec model = write_model(dir.path(), "span_fp32", span_model(false), TensorType::F32);
  const Artifact artifact = build(*backend, model, shapes(12, 16));
  const std::vector<std::byte> bytes = require_value(vk::net::read_file(artifact.path));
  const vk::net::ArtifactData data = require_value(vk::net::deserialize(bytes));
  CHECK(vk::net::serialize(data) == bytes);
  CHECK(data.choices.size() == data.graph.nodes.size());
  CHECK_FALSE(data.shaders.empty());
  CHECK(data.device_key == backend->profile().key);
  (void)require_value(vk::net::verify_artifact(data, backend->profile()));

  std::vector<std::byte> corrupt = bytes;
  corrupt[corrupt.size() / 2] ^= std::byte{0x40};
  require_refusal(vk::net::deserialize(corrupt), RefusalCode::HashMismatch);
  require_refusal(vk::net::deserialize(std::span(bytes).first(bytes.size() / 3)), RefusalCode::HashMismatch);
  std::vector<std::byte> version = bytes;
  version[8] = std::byte{99};
  require_refusal(vk::net::deserialize(version), RefusalCode::FormatVersion);

  vk::DeviceProfile other = backend->profile();
  other.key = "1002:744c:radv:Mesa:0";
  require_refusal(vk::net::verify_artifact(data, other), RefusalCode::DeviceMismatch);
  vk::net::ArtifactData changed = data;
  changed.shaders.front().hash ^= 1;
  require_refusal(vk::net::verify_artifact(changed, backend->profile()), RefusalCode::FormatVersion);
  changed = data;
  changed.dispatches.front().groups[0] += 1;
  require_refusal(vk::net::verify_artifact(changed, backend->profile()), RefusalCode::FormatVersion);
  // Sans abaisser, rien n'y paraît ; le chargement refait le programme et réécrit l'artefact.
  require_ok(vk::net::check_artifact(changed, backend->profile()));
  require_ok(vk::net::write_file(artifact.path, vk::net::serialize(changed)));
  CHECK(build(*backend, model, shapes(12, 16)).path == artifact.path);
  CHECK(require_value(vk::net::read_file(artifact.path)) == vk::net::serialize(changed));
  (void)require_value(backend->load(artifact));
  CHECK(require_value(vk::net::read_file(artifact.path)) == bytes);
  changed = data;
  changed.choices.front().variant = "disparue";
  require_refusal(vk::net::check_artifact(changed, backend->profile()), RefusalCode::FormatVersion);
  changed = data;
  changed.choices.pop_back();
  require_refusal(vk::net::check_artifact(changed, backend->profile()), RefusalCode::FormatVersion);

  // Un artefact abîmé sur le disque n'est pas chargé ; build le refait.
  std::ofstream(artifact.path, std::ios::binary | std::ios::trunc).write("abc", 3);
  require_refusal(backend->load(artifact), RefusalCode::CacheIo);
  (void)build(*backend, model, shapes(12, 16));
  (void)require_value(backend->load(artifact));
}

TEST_CASE("Backend : opérateur inconnu, annulation et précision refusés à la construction",
          "[vk_backend][backend]") {
  TempDir dir("refus");
  auto backend = make_backend(dir.path() / "cache", {.autotune = false});
  ModelBuilder m;
  m.input("x", {1, 3, 8, 8}).node("Erf", {"x"}, {"y"}).output("y");
  const ModelSpec unknown = write_model(dir.path(), "erf", m.bytes(), TensorType::F32);
  Progress progress;
  const auto refused = backend->build(unknown, shapes(8, 8), progress);
  require_refusal(refused, RefusalCode::OperatorUnknown);
  CHECK(refused.error().detail.find("Erf") != std::string::npos);

  const ModelSpec span = write_model(dir.path(), "span", span_model(false), TensorType::F32);
  std::stop_source stop;
  stop.request_stop();
  Progress cancelled{.stop = stop.get_token()};
  require_refusal(backend->build(span, shapes(8, 8), cancelled), RefusalCode::Cancelled);
  ModelSpec int8 = span;
  int8.precision = TensorType::I8;
  require_refusal(backend->build(int8, shapes(8, 8), progress), RefusalCode::FormatUnsupported);
  require_refusal(backend->import(ExternalImage{}), RefusalCode::FormatUnsupported);
  require_refusal(backend->alloc_tensor(TensorType::F16, Shape{{1, 3, 0, 64}}), RefusalCode::ShapeInvalid);
}

TEST_CASE("Backend : appareil de rendu, capacités et correspondance", "[vk_backend][backend]") {
  TempDir dir("appareil");
  auto backend = make_backend(dir.path());
  const BackendCaps caps = backend->capabilities();
  CHECK(caps.name == "vulkan");
  CHECK_FALSE(caps.device_name.empty());
  CHECK_FALSE(caps.needs_host_sync);
  CHECK(caps.matrix_units == backend->profile().target.matrix_units);
  CHECK(backend->match(backend->context().uuid()) == DeviceMatch::Same);
  CHECK(backend->match(DeviceUuid{}) == DeviceMatch::Different);

  const GpuDevice cuda{.api = GpuApi::Cuda, .pci_vendor = pci::kNvidia};
  require_refusal(vk::VulkanBackend::create(BackendInit{.render = &cuda}), RefusalCode::DeviceMismatch);
}

TEST_CASE("Backend : adoption de l'appareil de rendu", "[vk_backend][backend]") {
  TempDir dir("adoption");
  auto owner = make_backend(dir.path());
  vk::VkContext& c = owner->context();
  // L'appareil d'un autre (ici celui du premier backend), décrit comme present le ferait.
  GpuDevice render{.api = GpuApi::Vulkan,
                   .pci_vendor = c.caps().vendor_id,
                   .pci_device = c.caps().device_id,
                   .name = c.caps().device_name,
                   .uuid = c.uuid()};
  VkPhysicalDeviceVulkan12Features v12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                       .shaderFloat16 = c.caps().enabled.shader_float16 ? VK_TRUE : VK_FALSE,
                                       .timelineSemaphore = VK_TRUE};
  VkPhysicalDeviceVulkan11Features v11{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = &v12,
      .storageBuffer16BitAccess = c.caps().enabled.storage_buffer_16bit ? VK_TRUE : VK_FALSE};
  const VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                           .pNext = &v11};
  render.vulkan =
      VulkanDevice{.instance = c.instance(),
                   .physical_device = c.physical_device(),
                   .device = c.device(),
                   .get_instance_proc_addr = reinterpret_cast<VoidFunction>(c.get_instance_proc_addr()),
                   .api_version = VK_API_VERSION_1_2,
                   .device_extensions = c.caps().extensions,
                   .enabled_features = &features,
                   .compute = {.family = c.queue_family(), .index = c.queue_index()}};
  auto adopted =
      require_value(vk::VulkanBackend::create(BackendInit{.render = &render, .cache_dir = dir.path()}));
  CHECK(adopted->match(render.uuid) == DeviceMatch::Same);
  CHECK(adopted->context().device() == c.device());
  CHECK_FALSE(adopted->context().owns_device());
  // Les unités matricielles ne servent que si render les a activées : ici non.
  CHECK_FALSE(adopted->profile().target.matrix_units);
  auto t = require_value(adopted->alloc_tensor(TensorType::F32, Shape{{1, 3, 8, 8}}));
  auto u = require_value(adopted->alloc_tensor(TensorType::F32, Shape{{1, 3, 16, 16}}));
  require_ok(adopted->kernels()->resample(t->tensor(), kernels::Filter::Bilinear, u->tensor()));
  require_ok(adopted->context().wait_on_host(adopted->queue().last()));
}
