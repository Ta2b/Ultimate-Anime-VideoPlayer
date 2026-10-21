// Interface des backends, version 3 : un backend complet doit pouvoir s'écrire contre
// gpu/backend.h et gpu/kernel_ops.h, leurs valeurs par défaut ne rien supposer, et le
// chargeur refuser une bibliothèque d'une autre forme avant de lui passer quoi que ce soit.
#include "gpu/backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>

#include "media/hw_api.h"

using ajp::Refusal;
using ajp::RefusalCode;
using namespace ajp::gpu;

static_assert(kBackendAbi == 9);
// Aucune exception ne traverse les points d'entrée : sous MSVC, une exception sortie d'une
// fonction extern "C" est un comportement indéfini chez l'appelant.
static_assert(std::is_nothrow_invocable_r_v<int, BackendAbiFn>);
static_assert(std::is_nothrow_invocable_r_v<std::uint64_t, BackendLayoutFn>);
static_assert(std::is_nothrow_invocable_r_v<Backend*, BackendCreateFn, int, const BackendInit*>);
static_assert(std::is_nothrow_invocable_v<BackendDestroyFn, Backend*>);
static_assert(kBackendLayout != 0);
static_assert(std::is_same_v<DeviceUuid, ajp::DeviceUuid>, "un seul type d'UUID d'appareil");
static_assert(std::is_same_v<ajp::media::DeviceUuid, ajp::DeviceUuid>, "media partage le même UUID");
static_assert(std::is_abstract_v<KernelOps> && std::has_virtual_destructor_v<KernelOps>);
static_assert(std::is_abstract_v<TensorBuffer> && std::has_virtual_destructor_v<TensorBuffer>);

namespace {

std::unexpected<Refusal> refuse(RefusalCode code) {
  return std::unexpected(Refusal{code, "factice"});
}

// Noyaux factices : chaque opération refuse, comme un port qui n'a rien d'implémenté.
class NoKernels final : public KernelOps {
 public:
  std::expected<void, Refusal> ingest(const ajp::DecodedFrame&, const IngestSpec&, const Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported);
  }
  std::expected<void, Refusal> resample(const Tensor&, ajp::kernels::Filter, const Tensor&) override {
    return refuse(RefusalCode::ShapeInvalid);
  }
  std::expected<void, Refusal> rife_pad(const ajp::DecodedFrame&, const Tensor&) override {
    return refuse(RefusalCode::ShapeInvalid);
  }
  std::expected<void, Refusal> rife_crop(const Tensor&, const Tensor&) override {
    return refuse(RefusalCode::ShapeInvalid);
  }
  std::expected<void, Refusal> warp_blend(const Tensor&, const Tensor&, const Motion&, float,
                                          const Tensor&) override {
    return refuse(RefusalCode::FieldOutOfRange);
  }
  std::expected<void, Refusal> write_ring(const Tensor&, const ajp::RingSlotDesc&,
                                          const RingWrite&) override {
    return refuse(RefusalCode::FormatUnsupported);
  }
  std::expected<void, Refusal> measure(const ajp::DecodedFrame&, const ajp::DecodedFrame&) override {
    return refuse(RefusalCode::FormatUnsupported);
  }
  std::optional<PairScores> previous_scores() override { return std::nullopt; }
};

class HostBuffer final : public TensorBuffer {
 public:
  HostBuffer(TensorType type, const Shape& shape, int& alive)
      : tensor_{.type = type, .shape = shape}, alive_(alive) {
    ++alive_;
  }
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
  ~HostBuffer() override { --alive_; }
  const Tensor& tensor() const override { return tensor_; }

 private:
  Tensor tensor_;
  int& alive_;
};

class StubBackend final : public Backend {
 public:
  explicit StubBackend(const BackendInit& init) : init_(init) {}

  BackendCaps capabilities() const override { return {.name = "factice"}; }
  DeviceMatch match(const DeviceUuid& render) const override {
    if (init_.render == nullptr) return DeviceMatch::Unknown;
    return render == init_.render->uuid ? DeviceMatch::Same : DeviceMatch::Different;
  }
  std::expected<Artifact, Refusal> build(const ModelSpec&, const ShapeSet&, Progress&) override {
    return refuse(RefusalCode::BuilderFailed);
  }
  std::expected<std::unique_ptr<Engine>, Refusal> load(const Artifact&) override {
    return refuse(RefusalCode::ModelUnavailable);
  }
  std::expected<Tensor, Refusal> import(const ExternalImage&) override {
    return refuse(RefusalCode::FormatUnsupported);
  }
  std::expected<std::unique_ptr<TensorBuffer>, Refusal> alloc_tensor(TensorType type,
                                                                     const Shape& shape) override {
    for (const std::int64_t dim : shape.dims) {
      if (dim <= 0) return refuse(RefusalCode::ShapeInvalid);
    }
    return std::make_unique<HostBuffer>(type, shape, alive_);
  }
  KernelOps* kernels() override { return &kernels_; }
  std::expected<void, Refusal> enqueue(Engine&, std::span<const Tensor>, std::span<Tensor>) override {
    return refuse(RefusalCode::VendorCallFailed);  // comme un enqueueV3 refusé
  }
  std::expected<Fence, Refusal> signal() override { return Fence{.handle = 1, .value = ++fence_}; }
  std::expected<void, Refusal> wait(const Fence& fence) override {
    if (fence.value > fence_) return refuse(RefusalCode::FieldOutOfRange);  // jamais signalée
    return {};
  }

  int alive() const { return alive_; }

 private:
  BackendInit init_;
  NoKernels kernels_;
  int alive_ = 0;
  std::uint64_t fence_ = 0;
};

class NoEngine final : public Engine {
 public:
  std::span<const std::string> input_names() const override { return {}; }
  std::span<const std::string> output_names() const override { return {}; }
};

}  // namespace

TEST_CASE("Backend : BackendInit et IngestSpec ne supposent rien", "[backend]") {
  const BackendInit init;
  CHECK(init.render == nullptr);
  CHECK(init.cache_dir.empty());
  CHECK_FALSE(init.log);
  CHECK(init.refusal == nullptr);

  const IngestSpec chain;
  CHECK(chain.chroma == ajp::kernels::kChainChromaFilter);
  CHECK_FALSE(chain.rife_conversion);
  CHECK(RingWrite{}.wait_value == 0);
  CHECK_FALSE(PairScores{}.identical);
}

TEST_CASE("Backend : un backend complet s'écrit contre l'interface", "[backend]") {
  ajp::GpuDevice render{.api = ajp::GpuApi::Vulkan, .pci_vendor = ajp::pci::kNvidia};
  render.uuid[0] = 0x42;
  StubBackend backend(BackendInit{.render = &render, .cache_dir = "cache"});

  CHECK(backend.match(render.uuid) == DeviceMatch::Same);
  CHECK(backend.match(DeviceUuid{}) == DeviceMatch::Different);

  auto bad = backend.alloc_tensor(TensorType::F16, Shape{{1, 3, 0, 64}});
  REQUIRE_FALSE(bad);
  CHECK(bad.error().code == RefusalCode::ShapeInvalid);
  {
    auto buffer = backend.alloc_tensor(TensorType::F16, Shape{{1, 3, 64, 64}});
    REQUIRE(buffer);
    CHECK((*buffer)->tensor().shape == Shape{{1, 3, 64, 64}});
    CHECK(backend.alive() == 1);
  }
  CHECK(backend.alive() == 0);  // libérée à sa destruction

  KernelOps* ops = backend.kernels();
  REQUIRE(ops != nullptr);
  CHECK(ops->resample({}, ajp::kernels::Filter::Area, {}).error().code == RefusalCode::ShapeInvalid);
  CHECK_FALSE(ops->previous_scores());
  // Un backend qui ne connaît pas les voies : une seule, et use_lane ne change rien.
  CHECK(backend.capabilities().lanes == 1);
  backend.use_lane(Lane::Interpolate);
}

TEST_CASE("Backend : la file rend ses erreurs en refus nommés", "[backend]") {
  StubBackend backend(BackendInit{});
  NoEngine engine;
  auto enqueued = backend.enqueue(engine, {}, {});
  REQUIRE_FALSE(enqueued);
  CHECK(enqueued.error().code == RefusalCode::VendorCallFailed);

  auto fence = backend.signal();
  REQUIRE(fence);
  CHECK(backend.wait(*fence));
  auto never = backend.wait(Fence{.handle = 1, .value = fence->value + 1});
  REQUIRE_FALSE(never);
  CHECK(never.error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("Backend : le chargeur refuse une autre version ou une autre empreinte", "[backend]") {
  CHECK(check_backend_abi(kBackendAbi, kBackendLayout));

  auto older = check_backend_abi(2, kBackendLayout);
  REQUIRE_FALSE(older);
  CHECK(older.error().code == RefusalCode::AbiMismatch);
  CHECK(older.error().detail.find("interface 2") != std::string::npos);

  // Même version, autre forme : en-têtes modifiés sans changer kBackendAbi, ou autre
  // bibliothèque standard (libc++ contre libstdc++, CRT de débogage).
  auto reshaped = check_backend_abi(kBackendAbi, kBackendLayout ^ 1U);
  REQUIRE_FALSE(reshaped);
  CHECK(reshaped.error().code == RefusalCode::AbiMismatch);
  CHECK(reshaped.error().detail.find("empreinte") != std::string::npos);
}

TEST_CASE("Backend : l'empreinte suit la taille, l'alignement et l'ordre des types", "[backend]") {
  struct Small {
    std::uint32_t a;
  };
  struct Grown {
    std::uint32_t a;
    std::uint32_t b;  // un champ ajouté sans changer kBackendAbi
  };
  struct Aligned {
    alignas(8) std::uint32_t a;
  };
  CHECK(layout_fingerprint({1, 2}) == layout_fingerprint({1, 2}));
  CHECK(layout_fingerprint({1, 2}) != layout_fingerprint({2, 1}));
  CHECK(shape_fingerprint<Small, Grown>() != shape_fingerprint<Grown, Small>());
  CHECK(shape_fingerprint<Small>() != shape_fingerprint<Grown>());
  CHECK(shape_fingerprint<Small>() != shape_fingerprint<Aligned>());
  CHECK(shape_fingerprint<Small>() == shape_fingerprint<std::uint32_t>());  // même forme, même empreinte
}
