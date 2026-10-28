// RIFE fusionné (kernels/spec/rife_fused.h) dans le backend TensorRT :
//   - les trois noyaux CUDA contre la référence CPU en double (kernels/ref/rife_fused.h), sur des données
//     aléatoires, pour les deux découpages du catalogue (4.25 : 8, 8, 4, 2, 1 ; 4.22 lite : 8, 4, 2, 1) ;
//   - en test caché [.rife_fused], la chaîne complète (six moteurs construits par trtexec et noyaux) contre
//   le
//     moteur RIFE entier de TensorRT, sur une paire d'images : PSNR de l'image interpolée d'au moins 50 dB,
//     pour RIFE 4.25 et 4.22 lite, image et variante mouvement (flot et masque).
//
// Le test caché lit les modèles du lecteur en place (AJP_RIFE_MODEL_DIR, sans copie dans le dépôt) ; une
// paire d'images d'anime en PPM (AJP_RIFE_IMAGE : une image dont une fenêtre sert de première image et la
// même fenêtre décalée de 12 pixels de seconde) remplace le dessin synthétique :
//   ajp-gpu ajp_backend_tensorrt_tests "[.rife_fused]"
#include "kernels/ref/rife_fused.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "gpu/cuda/backend/rife_fused_kernels.h"
#include "kernels/ref/half.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/rife_fused.h"
#include "prepare/model_source.h"
#include "prepare/onnx_outputs.h"
#include "prepare/rife_split.h"
#include "prepare/sha256.h"

namespace ajp::test {
namespace {

namespace fs = std::filesystem;
using gpu::cuda::FusedEntryParams;
using gpu::cuda::FusedFinishParams;
using gpu::cuda::FusedGlueParams;

// Un tampon d'appareil libéré à la sortie.
class Device {
 public:
  explicit Device(std::size_t bytes) { REQUIRE(cudaMalloc(&ptr_, bytes) == cudaSuccess); }
  ~Device() { cudaFree(ptr_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

std::vector<std::uint16_t> to_half(const std::vector<double>& values) {
  std::vector<std::uint16_t> half(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    half[i] = kernels::ref::float_to_half(static_cast<float>(values[i]));
  return half;
}

// Les valeurs telles que le GPU les relira : arrondies en FP16.
std::vector<double> rounded(const std::vector<double>& values) {
  std::vector<double> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    out[i] = static_cast<double>(
        kernels::ref::half_to_float(kernels::ref::float_to_half(static_cast<float>(values[i]))));
  return out;
}

void upload_half(void* device, const std::vector<double>& values) {
  const std::vector<std::uint16_t> half = to_half(values);
  REQUIRE(cudaMemcpy(device, half.data(), half.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
}

std::vector<double> download_half(const void* device, std::size_t count) {
  std::vector<std::uint16_t> half(count);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaMemcpy(half.data(), device, count * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
  std::vector<double> out(count);
  for (std::size_t i = 0; i < count; ++i) out[i] = static_cast<double>(kernels::ref::half_to_float(half[i]));
  return out;
}

// PSNR (dB) de got contre want, le pic étant le plus grand |want| ; écart maximal en sortie.
double psnr_of(const std::vector<double>& got, const std::vector<double>& want, double& worst) {
  REQUIRE(got.size() == want.size());
  double square = 0.0;
  double peak = 0.0;
  worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = got[i] - want[i];
    square += d * d;
    worst = std::max(worst, std::abs(d));
    peak = std::max(peak, std::abs(want[i]));
  }
  const double mse = square / static_cast<double>(got.size());
  return mse == 0.0 ? 200.0 : 10.0 * std::log10(peak * peak / mse);
}

// L'état de la colle : entrée de RIFE (images au hasard, instant, maillage), encodage et sorties de blocs au
// hasard.
kernels::ref::RifeFusedState random_state(const kernels::RifeFusedPlan& plan, int h, int w,
                                          std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> signed_unit(-1.0, 1.0);
  const std::size_t area = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  kernels::ref::RifeFusedState state{.plan = plan, .height = h, .width = w};
  state.input.assign(11 * area, 0.0);
  for (std::size_t i = 0; i < 6 * area; ++i) state.input[i] = unit(rng);
  for (std::size_t i = 0; i < area; ++i) state.input[6 * area + i] = 0.5;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t i =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
      state.input[7 * area + i] = kernels::rife_mesh(x, w);
      state.input[8 * area + i] = kernels::rife_mesh(y, h);
      state.input[9 * area + i] = kernels::rife_mesh_step(w);
      state.input[10 * area + i] = kernels::rife_mesh_step(h);
    }
  }
  state.input = rounded(state.input);
  state.encode.resize(8 * area);
  for (double& v : state.encode) v = signed_unit(rng);
  state.encode = rounded(state.encode);
  for (int k = 0; k < plan.blocks; ++k) {
    const kernels::FusedDims y = kernels::rife_fused_y(plan, k, h, w);
    std::vector<double> values(static_cast<std::size_t>(y.elements()));
    for (double& v : values)
      v = 0.6 * signed_unit(rng);  // des flots de quelques pixels une fois multipliés par d
    state.y.push_back(rounded(values));
  }
  return state;
}

// Les noyaux sur l'état, comparés à la référence bloc par bloc (les Y sont les mêmes des deux côtés).
void compare_kernels(const kernels::RifeFusedPlan& plan, int h, int w, std::uint32_t seed) {
  const kernels::ref::RifeFusedState state = random_state(plan, h, w, seed);
  const kernels::RifeFusedLayout layout = kernels::rife_fused_layout(plan, h, w);
  Device input(state.input.size() * 2);
  Device encode(state.encode.size() * 2);
  upload_half(input.get(), state.input);
  upload_half(encode.get(), state.encode);
  std::vector<std::unique_ptr<Device>> y;
  std::vector<std::unique_ptr<Device>> x;
  for (int k = 0; k < plan.blocks; ++k) {
    const auto i = static_cast<std::size_t>(k);
    y.push_back(std::make_unique<Device>(state.y[i].size() * 2));
    upload_half(y[i]->get(), state.y[i]);
    x.push_back(std::make_unique<Device>(
        static_cast<std::size_t>(kernels::rife_fused_x(plan, k, h, w).elements()) * 2));
  }
  INFO("plan " << plan.reduce[0] << "…" << plan.blocks << " blocs, " << w << "x" << h);

  const kernels::FusedDims x0 = kernels::rife_fused_x(plan, 0, h, w);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(gpu::cuda::launch_fused_entry({.height = h,
                                         .width = w,
                                         .reduce = plan.reduce[0],
                                         .out_height = x0.height,
                                         .out_width = x0.width,
                                         .input = input.get(),
                                         .encode = encode.get(),
                                         .x = x[0]->get()},
                                        nullptr) == cudaSuccess);
  double worst = 0.0;
  const auto check_block = [&](int k) {
    const kernels::FusedDims dims = kernels::rife_fused_x(plan, k, h, w);
    const std::vector<double> got =
        download_half(x[static_cast<std::size_t>(k)]->get(), static_cast<std::size_t>(dims.elements()));
    const std::vector<double> want = kernels::ref::rife_fused_block_input(state, k);
    const double score = psnr_of(got, want, worst);
    INFO("X" << k << " : PSNR " << score << " dB, écart maximal " << worst);
    CHECK(score >= 55.0);
  };
  check_block(0);
  for (int b = 0; b + 1 < plan.blocks; ++b) {
    const auto i = static_cast<std::size_t>(b);
    const kernels::FusedDims next = kernels::rife_fused_x(plan, b + 1, h, w);
    FusedGlueParams glue{.height = h,
                         .width = w,
                         .reduce = plan.reduce[i + 1],
                         .out_height = next.height,
                         .out_width = next.width,
                         .flows = b + 1,
                         .input = input.get(),
                         .encode = encode.get(),
                         .x = x[i + 1]->get()};
    for (int k = 0; k <= b; ++k) {
      const auto j = static_cast<std::size_t>(k);
      const kernels::FusedDims yk = kernels::rife_fused_y(plan, k, h, w);
      glue.source[j] = {.y = y[j]->get(), .height = yk.height, .width = yk.width, .scale = plan.reduce[j]};
    }
    REQUIRE(gpu::cuda::launch_fused_glue(glue, nullptr) == cudaSuccess);
    check_block(b + 1);
  }
  // La fin, image puis flot et masque.
  const kernels::ref::RifeFusedResult want = kernels::ref::rife_fused_finish(state);
  const auto last = static_cast<std::size_t>(plan.blocks - 1);
  const kernels::FusedDims y_last = kernels::rife_fused_y(plan, plan.blocks - 1, h, w);
  Device image(3 * static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2);
  Device flow(4 * static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2);
  Device mask(static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2);
  FusedFinishParams finish{.height = h,
                           .width = w,
                           .input = input.get(),
                           .x_last = x[last]->get(),
                           .y_last = y[last]->get(),
                           .y_height = y_last.height,
                           .y_width = y_last.width};
  finish.image = image.get();
  REQUIRE(gpu::cuda::launch_fused_finish(finish, nullptr) == cudaSuccess);
  double score = psnr_of(download_half(image.get(), want.image.size()), want.image, worst);
  INFO("image : PSNR " << score << " dB, écart maximal " << worst);
  CHECK(score >= 55.0);
  finish.image = nullptr;
  finish.flow = flow.get();
  finish.mask = mask.get();
  REQUIRE(gpu::cuda::launch_fused_finish(finish, nullptr) == cudaSuccess);
  score = psnr_of(download_half(flow.get(), want.flow.size()), want.flow, worst);
  INFO("flot : PSNR " << score << " dB, écart maximal " << worst);
  CHECK(score >= 55.0);
  score = psnr_of(download_half(mask.get(), want.mask.size()), want.mask, worst);
  INFO("masque : PSNR " << score << " dB, écart maximal " << worst);
  CHECK(score >= 55.0);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(layout.bytes > 0);
}

// Une image d'anime en PPM binaire (P6, 255) en plans RGB de 0 à 1 ; vide si illisible.
std::vector<float> read_ppm(const fs::path& path, int& h, int& w) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  file >> magic >> w >> h >> maxval;
  if (!file || magic != "P6" || maxval != 255 || w <= 0 || h <= 0) return {};
  file.get();
  std::vector<unsigned char> raw(static_cast<std::size_t>(3 * h * w));
  file.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
  if (!file) return {};
  std::vector<float> planes(raw.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c)
        planes[static_cast<std::size_t>((c * h + y) * w + x)] =
            static_cast<float>(raw[static_cast<std::size_t>((y * w + x) * 3 + c)]) / 255.F;
  return planes;
}

// Un dessin d'anime synthétique : aplats, dégradé, traits nets, grain ; shift le décale vers la droite.
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
        v += 0.004F * static_cast<float>((u * 7 + y * 13 + c * 5) % 5);
        plane[static_cast<std::size_t>((c * h + y) * w + x)] = std::clamp(v, 0.F, 1.F);
      }
    }
  }
  return plane;
}

// Les deux images de la paire en plans RGB h × w : une fenêtre de l'image PPM et la même décalée de 12
// pixels, ou deux dessins décalés.
std::pair<std::vector<float>, std::vector<float>> picture_pair(int h, int w) {
  constexpr int kShift = 12;
  if (const fs::path image(AJP_RIFE_IMAGE); !image.empty() && fs::is_regular_file(image)) {
    int ih = 0, iw = 0;
    const std::vector<float> whole = read_ppm(image, ih, iw);
    if (!whole.empty() && ih >= h && iw >= w + kShift) {
      const auto window = [&](int x0) {
        std::vector<float> out(static_cast<std::size_t>(3 * h * w));
        for (int c = 0; c < 3; ++c)
          for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
              out[static_cast<std::size_t>((c * h + y) * w + x)] =
                  whole[static_cast<std::size_t>((c * ih + y) * iw + x0 + x)];
        return out;
      };
      return {window(0), window(kShift)};
    }
  }
  return {drawing(h, w, 0), drawing(h, w, kShift)};
}

gpu::ShapeSet fixed_input(int h, int w) {
  const gpu::Shape shape{.dims = {1, 11, h, w}};
  return {.min = shape, .opt = shape, .max = shape};
}

// Compare la chaîne fusionnée au RIFE entier de TensorRT sur la paire, pour un modèle du catalogue.
void compare_engines(const std::string& file, bool motion) {
  const fs::path onnx = fs::path(AJP_RIFE_MODEL_DIR) / file;
  if (!fs::is_regular_file(onnx)) SKIP("modèle absent : " << onnx.string());
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  REQUIRE(loaded.backend->capabilities().rife_fused);
  constexpr int h = 384;
  constexpr int w = 640;
  const fs::path scratch = fs::temp_directory_path() / "ajp-rife-fused-test";
  fs::create_directories(scratch);
  gpu::Progress progress;

  // Le moteur entier, ou sa variante mouvement (flot et masque en sorties) : les tenseurs connus de vs-mlrt.
  fs::path reference_onnx = onnx;
  if (motion) {
    auto bytes = prepare::read_onnx_bytes(onnx);
    REQUIRE(bytes);
    const std::vector<prepare::OnnxOutput> outputs{
        {.name = file == "rife_v4.25.onnx" ? "/Add_7_output_0" : "/Add_5_output_0"},
        {.name = "/Sigmoid_output_0"}};
    auto derived = prepare::with_outputs(*bytes, outputs);
    REQUIRE(derived);
    reference_onnx = scratch / (onnx.stem().string() + "-mouvement.onnx");
    std::ofstream out(reference_onnx, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(derived->data()), static_cast<std::streamsize>(derived->size()));
  }
  const auto sha = prepare::sha256_file(reference_onnx);
  REQUIRE(sha);
  auto single = loaded.backend->build(
      gpu::ModelSpec{.id = reference_onnx.stem().string(), .onnx_path = reference_onnx, .sha256 = *sha},
      fixed_input(h, w), progress);
  REQUIRE(single);
  auto reference = loaded.backend->load(*single);
  REQUIRE(reference);

  const auto manifest = prepare::derive_rife_fused(
      onnx, scratch, h, w, motion ? prepare::RifeFusedOutput::Motion : prepare::RifeFusedOutput::Image);
  REQUIRE(manifest);
  const auto manifest_sha = prepare::sha256_file(*manifest);
  REQUIRE(manifest_sha);
  auto fused_artifact = loaded.backend->build(
      gpu::ModelSpec{.id = manifest->stem().string(), .onnx_path = *manifest, .sha256 = *manifest_sha},
      fixed_input(h, w), progress);
  REQUIRE(fused_artifact);
  auto fused = loaded.backend->load(*fused_artifact);
  REQUIRE(fused);

  // La paire, ingérée comme le lecteur la prépare : rife_input pour l'instant 0,5.
  const auto [first, second] = picture_pair(h, w);
  const gpu::Shape rgb{.dims = {1, 3, h, w}};
  auto a = loaded.backend->alloc_tensor(gpu::TensorType::F16, rgb);
  auto b = loaded.backend->alloc_tensor(gpu::TensorType::F16, rgb);
  auto input = loaded.backend->alloc_tensor(gpu::TensorType::F16, gpu::Shape{.dims = {1, 11, h, w}});
  REQUIRE(a);
  REQUIRE(b);
  REQUIRE(input);
  upload((*a)->tensor(), first);
  upload((*b)->tensor(), second);
  REQUIRE(loaded.backend->kernels()->rife_input((*a)->tensor(), (*b)->tensor(), 0.5F, (*input)->tensor()));

  const std::array inputs{(*input)->tensor()};
  const auto run = [&](gpu::Engine& engine, const std::vector<gpu::Shape>& shapes) {
    std::vector<std::unique_ptr<gpu::TensorBuffer>> buffers;
    std::vector<gpu::Tensor> outs;
    for (const gpu::Shape& shape : shapes) {
      auto buffer = loaded.backend->alloc_tensor(gpu::TensorType::F16, shape);
      REQUIRE(buffer);
      outs.push_back((*buffer)->tensor());
      buffers.push_back(std::move(*buffer));
    }
    // Trois lancements de suite : exécution à froid, capture du graphe CUDA, rejeu ; le dernier est celui
    // qu'on lit.
    for (int i = 0; i < 3; ++i) REQUIRE(loaded.backend->enqueue(engine, inputs, outs));
    std::vector<std::vector<float>> got;
    for (const gpu::Tensor& t : outs) got.push_back(download(t));
    return got;
  };
  const std::vector<gpu::Shape> shapes =
      motion ? std::vector<gpu::Shape>{{.dims = {1, 4, h, w}}, {.dims = {1, 1, h, w}}}
             : std::vector<gpu::Shape>{rgb};
  const auto want = run(**reference, shapes);
  const auto got = run(**fused, shapes);
  REQUIRE(want.size() == got.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    const std::vector<double> x(got[i].begin(), got[i].end());
    const std::vector<double> y(want[i].begin(), want[i].end());
    double worst = 0.0;
    const double score = psnr_of(x, y, worst);
    WARN(file << (motion ? " mouvement" : " image") << " sortie " << i << " : PSNR " << score
              << " dB contre TensorRT entier, écart maximal " << worst);
    CHECK(score >= 50.0);
  }
}

}  // namespace

TEST_CASE("RIFE fusionné : la chaîne contre le moteur RIFE entier de TensorRT",
          "[.rife_fused][backend_tensorrt][cuda]") {
  SECTION("4.25, image") {
    compare_engines("rife_v4.25.onnx", false);
  }
  SECTION("4.22 lite, image") {
    compare_engines("rife_v4.22_lite.onnx", false);
  }
  SECTION("4.25, flot et masque") {
    compare_engines("rife_v4.25.onnx", true);
  }
  SECTION("4.22 lite, flot et masque") {
    compare_engines("rife_v4.22_lite.onnx", true);
  }
}

TEST_CASE("RIFE fusionné : les noyaux CUDA suivent la référence CPU en double",
          "[backend_tensorrt][cuda][rife_fused]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  // RIFE 4.25 : cinq blocs réduits 8, 8, 4, 2, 1 ; 4.22 lite : quatre blocs réduits 8, 4, 2, 1.
  const kernels::RifeFusedPlan v425{.blocks = 5, .reduce = {8, 8, 4, 2, 1}};
  const kernels::RifeFusedPlan v422{.blocks = 4, .reduce = {8, 4, 2, 1}};
  SECTION("4.25, 96x160") {
    compare_kernels(v425, 96, 160, 1);
  }
  SECTION("4.22 lite, 96x160") {
    compare_kernels(v422, 96, 160, 2);
  }
  SECTION("4.25, taille non carrée et lignes impaires de blocs") {
    compare_kernels(v425, 144, 208, 3);
  }
  SECTION("4.22 lite, 64x128") {
    compare_kernels(v422, 64, 128, 4);
  }
}

}  // namespace ajp::test
