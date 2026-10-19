// Backend CPU de référence : chaque noyau rend exactement ce que rend kernels/ref, l'arrondi
// fp16 en plus pour les tenseurs F16 ; les moteurs passent par l'interpréteur de gpu_ir.
#include "gpu/cpu/cpu_backend.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stop_token>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu_test_support.h"
#include "kernels/ref/color.h"
#include "kernels/ref/half.h"
#include "kernels/ref/resample.h"
#include "kernels/ref/rife.h"
#include "kernels/ref/warp.h"

using ajp::DecodedFrame;
using ajp::RefusalCode;
using namespace ajp::gpu;
namespace kernels = ajp::kernels;
namespace ref = ajp::kernels::ref;

namespace {

// Un dégradé, pour que chaque noyau ait quelque chose à calculer.
DecodedFrame gradient(std::uint32_t id, int shift = 0) {
  auto data = std::make_shared<test::HostFrame>(test::kWidth, test::kHeight, 0);
  for (int y = 0; y < test::kHeight; ++y) {
    for (int x = 0; x < test::kWidth; ++x) {
      data->plane(0)[static_cast<std::size_t>(y * test::kWidth + x)] =
          std::byte(16 + (3 * x + 5 * y + shift) % 200);
    }
  }
  for (std::size_t i = 0; i < data->plane(1).size(); ++i) data->plane(1)[i] = std::byte(100 + i % 50);
  DecodedFrame frame = test::make_frame(id, 0);
  frame.data = std::move(data);
  return frame;
}

kernels::YuvImage yuv_of(const DecodedFrame& frame) {
  auto image = cpu::read_frame(frame);
  REQUIRE(image);
  return *image;
}

std::vector<float> values_of(const TensorBuffer& buffer) {
  const Tensor& t = buffer.tensor();
  std::size_t count = 1;
  for (const auto dim : t.shape.dims) count *= static_cast<std::size_t>(dim);
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (t.type == TensorType::F32) {
      std::memcpy(&values[i], reinterpret_cast<const std::byte*>(t.handle) + 4 * i, 4);
    } else {
      std::uint16_t half = 0;
      std::memcpy(&half, reinterpret_cast<const std::byte*>(t.handle) + 2 * i, 2);
      values[i] = ref::half_to_float(half);
    }
  }
  return values;
}

std::vector<float> flat(const kernels::RgbImage& image) {
  std::vector<float> values;
  for (const auto& plane : image.planes) values.insert(values.end(), plane.data.begin(), plane.data.end());
  return values;
}

std::unique_ptr<TensorBuffer> alloc(Backend& backend, TensorType type, int c, int h, int w) {
  auto buffer = backend.alloc_tensor(type, Shape{{1, c, h, w}});
  REQUIRE(buffer);
  return std::move(*buffer);
}

}  // namespace

TEST_CASE("Backend CPU : ingestion, rééchantillonnage et RIFE égaux à la référence", "[cpu]") {
  cpu::CpuBackend backend{BackendInit{}};
  KernelOps& ops = *backend.kernels();
  const DecodedFrame frame = gradient(1);
  const kernels::YuvImage yuv = yuv_of(frame);
  const ref::ColorSpec color{.matrix = kernels::Matrix::Bt709};

  auto rgb = alloc(backend, TensorType::F32, 3, test::kHeight, test::kWidth);
  REQUIRE(ops.ingest(frame, IngestSpec{}, rgb->tensor()));
  const auto expected = ref::yuv_to_rgb(yuv, color);
  REQUIRE(expected);
  CHECK(std::ranges::equal(values_of(*rgb), flat(*expected)));

  auto half = alloc(backend, TensorType::F16, 3, test::kHeight, test::kWidth);
  REQUIRE(ops.ingest(frame, IngestSpec{}, half->tensor()));
  CHECK(std::ranges::equal(values_of(*half), flat(ref::round_to_half(*expected))));

  auto small = alloc(backend, TensorType::F32, 3, 18, 32);
  REQUIRE(ops.resample(rgb->tensor(), kernels::Filter::Area, small->tensor()));
  CHECK(std::ranges::equal(values_of(*small), flat(*ref::resize(*expected, 32, 18, kernels::Filter::Area))));
  CHECK(ops.resample(small->tensor(), kernels::Filter::Area, rgb->tensor()).error().code ==
        RefusalCode::FieldOutOfRange);  // Area n'agrandit pas

  auto padded = alloc(backend, TensorType::F32, 3, 64, 64);
  REQUIRE(ops.rife_pad(frame, padded->tensor()));
  const auto prepared = ref::rife_prepare(yuv, kernels::Range::Limited);
  REQUIRE(prepared);
  CHECK(std::ranges::equal(values_of(*padded), flat(*prepared)));
  auto cropped = alloc(backend, TensorType::F32, 3, test::kHeight, test::kWidth);
  REQUIRE(ops.rife_crop(padded->tensor(), cropped->tensor()));
  CHECK(std::ranges::equal(
      values_of(*cropped),
      flat(*ref::rife_crop(*prepared, kernels::rife_geometry(test::kWidth, test::kHeight)))));
  CHECK(ops.rife_pad(frame, rgb->tensor()).error().code == RefusalCode::ShapeInvalid);

  auto input = alloc(backend, TensorType::F32, 11, 64, 64);
  REQUIRE(backend.rife_input()->rife_input(padded->tensor(), padded->tensor(), 0.25F, input->tensor()));
  const auto planes = ref::rife_input(*prepared, *prepared, 0.25F);
  REQUIRE(planes);
  std::vector<float> expected_input;
  for (const auto& plane : *planes)
    expected_input.insert(expected_input.end(), plane.data.begin(), plane.data.end());
  CHECK(std::ranges::equal(values_of(*input), expected_input));
}

TEST_CASE("Backend CPU : warp et mélange égaux à la référence", "[cpu]") {
  cpu::CpuBackend backend{BackendInit{}};
  KernelOps& ops = *backend.kernels();
  auto a = alloc(backend, TensorType::F32, 3, test::kHeight, test::kWidth);
  auto b = alloc(backend, TensorType::F32, 3, test::kHeight, test::kWidth);
  REQUIRE(ops.ingest(gradient(1), IngestSpec{}, a->tensor()));
  REQUIRE(ops.ingest(gradient(2, 40), IngestSpec{}, b->tensor()));
  // Mouvement nul, masque à 0,5, dans la fenêtre de RIFE (64 × 64 rempli).
  auto engine = alloc(backend, TensorType::F32, 11, 64, 64);
  const kernels::RifeGeometry geometry = kernels::rife_geometry(test::kWidth, test::kHeight);
  test::TempDir dir("warp");
  auto motion_engine =
      test::load_engine(backend, test::write_model(dir.path(), "motion", test::motion_model()), "m");
  REQUIRE(motion_engine);
  auto flow = alloc(backend, TensorType::F32, 4, 64, 64);
  auto mask = alloc(backend, TensorType::F32, 1, 64, 64);
  std::array outputs{flow->tensor(), mask->tensor()};
  const std::array inputs{engine->tensor()};
  REQUIRE(backend.enqueue(*motion_engine, inputs, outputs));
  CHECK(std::ranges::equal(values_of(*mask), std::vector<float>(64 * 64, 0.5F)));

  const Motion motion{
      .flow = flow->tensor(),
      .mask = mask->tensor(),
      .area = {.x = geometry.left, .y = geometry.top, .width = test::kWidth, .height = test::kHeight}};
  auto out = alloc(backend, TensorType::F32, 3, test::kHeight, test::kWidth);
  REQUIRE(ops.warp_blend(a->tensor(), b->tensor(), motion, 0.25F, out->tensor()));
  ref::Motion reference{.mask = kernels::PlaneF(64, 64, 0.5F), .area = motion.area};
  reference.flow.fill(kernels::PlaneF(64, 64, 0.0F));
  kernels::RgbImage ia;
  kernels::RgbImage ib;
  const auto va = values_of(*a);
  const auto vb = values_of(*b);
  for (std::size_t c = 0; c < 3; ++c) {
    const std::size_t area = std::size_t{test::kWidth} * test::kHeight;
    ia.planes[c] = kernels::PlaneF(test::kWidth, test::kHeight);
    ib.planes[c] = kernels::PlaneF(test::kWidth, test::kHeight);
    std::copy_n(va.begin() + static_cast<std::ptrdiff_t>(c * area), area, ia.planes[c].data.begin());
    std::copy_n(vb.begin() + static_cast<std::ptrdiff_t>(c * area), area, ib.planes[c].data.begin());
  }
  CHECK(std::ranges::equal(values_of(*out), flat(*ref::warp_blend_at(ia, ib, reference, 0.25F))));
  CHECK(ops.warp_blend(a->tensor(), b->tensor(), motion, 1.0F, out->tensor()).error().code ==
        RefusalCode::FieldOutOfRange);
}

TEST_CASE("Backend CPU : écriture d'un slot hôte du ring", "[cpu]") {
  cpu::CpuBackend backend{BackendInit{}};
  KernelOps& ops = *backend.kernels();
  auto rgb = alloc(backend, TensorType::F16, 3, test::kHeight, test::kWidth);
  REQUIRE(ops.ingest(gradient(1), IngestSpec{}, rgb->tensor()));
  test::HostRing ring(1, test::kWidth, test::kHeight);
  const ajp::RingSlotDesc& slot = ring.slots()[0];

  REQUIRE(ops.write_ring(rgb->tensor(), slot, {.wait_value = 0, .signal_value = 1}));
  const auto expected = ref::to_rgba16(ref::round_to_half(*ref::yuv_to_rgb(yuv_of(gradient(1)), {})));
  REQUIRE(expected);
  CHECK(std::ranges::equal(ring.pixels(0), *expected));

  CHECK(ops.write_ring(rgb->tensor(), slot, {.wait_value = 2, .signal_value = 2}).error().code ==
        RefusalCode::SequenceOrder);  // valeur paire
  CHECK(ops.write_ring(rgb->tensor(), slot, {.wait_value = 0, .signal_value = 1}).error().code ==
        RefusalCode::SequenceOrder);  // la valeur reculerait
  REQUIRE(ops.write_ring(rgb->tensor(), slot, {.wait_value = 2, .signal_value = 3}));
  ajp::RingSlotDesc exported = slot;
  exported.memory = {.kind = ajp::HandleKind::OpaqueFd, .value = 3};
  CHECK(ops.write_ring(rgb->tensor(), exported, {.wait_value = 4, .signal_value = 5}).error().code ==
        RefusalCode::FormatUnsupported);
  auto other = alloc(backend, TensorType::F16, 3, 18, 32);
  CHECK(ops.write_ring(other->tensor(), slot, {.wait_value = 4, .signal_value = 5}).error().code ==
        RefusalCode::ShapeInvalid);
}

TEST_CASE("Backend CPU : mesure d'une paire, relue une seule fois", "[cpu]") {
  cpu::CpuBackend backend{BackendInit{}};
  KernelOps& ops = *backend.kernels();
  CHECK_FALSE(ops.previous_scores());
  REQUIRE(ops.measure(gradient(1), gradient(2)));
  auto same = ops.previous_scores();
  REQUIRE(same);
  CHECK(same->left == ajp::SourceId{1});
  CHECK(same->right == ajp::SourceId{2});
  CHECK(same->identical);
  CHECK(same->worst_block == 0);
  CHECK_FALSE(ops.previous_scores());  // une seule fois

  REQUIRE(ops.measure(test::make_frame(3, 16), test::make_frame(4, 235)));
  auto cut = ops.previous_scores();
  REQUIRE(cut);
  CHECK_FALSE(cut->identical);
  CHECK(kernels::is_scene_cut(cut->scene_score));
  CHECK(ops.measure(test::make_frame(5, 16), test::make_frame(6, 16, 0, 0, 32, 18)).error().code ==
        RefusalCode::ShapeInvalid);
}

TEST_CASE("Backend CPU : refus des tenseurs et images hors contrat", "[cpu]") {
  cpu::CpuBackend backend{BackendInit{}};
  KernelOps& ops = *backend.kernels();
  CHECK(backend.alloc_tensor(TensorType::F32, Shape{{1, 3, 0, 4}}).error().code == RefusalCode::ShapeInvalid);
  const Tensor stray{
      .type = TensorType::F32, .shape = Shape{{1, 3, test::kHeight, test::kWidth}}, .handle = 64};
  CHECK(ops.ingest(gradient(1), IngestSpec{}, stray).error().code == RefusalCode::InputMismatch);

  DecodedFrame device = gradient(1);
  struct DeviceFrame final : test::HostFrame {
    DeviceFrame() : HostFrame(test::kWidth, test::kHeight, 16) {}
    ajp::FrameMemory memory() const override { return ajp::FrameMemory::Cuda; }
  };
  device.data = std::make_shared<DeviceFrame>();
  auto rgb = alloc(backend, TensorType::F32, 3, test::kHeight, test::kWidth);
  CHECK(ops.ingest(device, IngestSpec{}, rgb->tensor()).error().code == RefusalCode::FormatUnsupported);
  CHECK(backend.import({}).error().code == RefusalCode::FormatUnsupported);
}

TEST_CASE("Backend CPU : un moteur ONNX s'exécute par l'interpréteur", "[cpu]") {
  cpu::CpuBackend backend{BackendInit{}};
  test::TempDir dir("engine");
  auto engine =
      test::load_engine(backend, test::write_model(dir.path(), "up2", test::upscale_x2_model()), "up2");
  REQUIRE(engine);
  CHECK(engine->input_names().size() == 1);

  auto in = alloc(backend, TensorType::F32, 3, 2, 3);
  std::vector<float> source(18);
  for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<float>(i);
  std::memcpy(reinterpret_cast<void*>(in->tensor().handle), source.data(), source.size() * sizeof(float));
  auto out = alloc(backend, TensorType::F32, 3, 4, 6);
  const std::array inputs{in->tensor()};
  std::array outputs{out->tensor()};
  REQUIRE(backend.enqueue(*engine, inputs, outputs));
  const auto values = values_of(*out);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 6; ++x) {
        CHECK(values[static_cast<std::size_t>((c * 4 + y) * 6 + x)] ==
              source[static_cast<std::size_t>((c * 2 + y / 2) * 3 + x / 2)]);
      }
    }
  }
  auto wrong = alloc(backend, TensorType::F32, 3, 3, 3);
  std::array wrong_outputs{wrong->tensor()};
  auto refused = backend.enqueue(*engine, inputs, wrong_outputs);
  REQUIRE_FALSE(refused);  // ABI 3 : la file rend son refus
  CHECK(refused.error().code == RefusalCode::ShapeInvalid);
  CHECK(backend.signal()->value == 1);
  CHECK(backend.wait({}));

  Progress cancelled;
  std::stop_source stop;
  stop.request_stop();
  cancelled.stop = stop.get_token();
  CHECK(backend.build({.id = "up2"}, {}, cancelled).error().code == RefusalCode::Cancelled);
}
