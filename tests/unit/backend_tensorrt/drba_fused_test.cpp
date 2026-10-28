// DistilDRBA fusionné (kernels/spec/drba_fused.h) dans le backend TensorRT :
//   - les noyaux de réduction (pre et post, des moyennes 2 × 2 exactes) contre un calcul sur l'hôte, bit à
//   bit ;
//   - en test caché [.drba_fused], la chaîne complète (quatre moteurs construits par trtexec et noyaux)
//   contre le
//     moteur DistilDRBA entier de TensorRT, sur une paire d'images : PSNR de l'image interpolée d'au moins 50
//     dB, des deux côtés de l'image du milieu (tau 0,5 et 1,25).
//
// Le test caché lit le modèle du lecteur en place (AJP_DRBA_MODEL_DIR, sans copie dans le dépôt) ; une image
// d'anime en PPM (AJP_DRBA_IMAGE : une image dont une fenêtre sert de première image, la même décalée de 6
// puis de 12 pixels de deuxième et troisième) remplace le dessin synthétique :
//   ajp-gpu ajp_backend_tensorrt_tests "[.drba_fused]"
#include "kernels/spec/drba_fused.h"

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
#include "gpu/cuda/backend/drba_fused_kernels.h"
#include "kernels/ref/half.h"
#include "prepare/drba_split.h"
#include "prepare/model_source.h"
#include "prepare/onnx_dims.h"
#include "prepare/sha256.h"

namespace ajp::test {
namespace {

namespace fs = std::filesystem;
using gpu::cuda::DrbaPostParams;
using gpu::cuda::DrbaPreParams;

// Un tampon d'appareil libéré à la sortie.
class Device {
 public:
  explicit Device(std::size_t bytes) {
    REQUIRE(cudaMalloc(&ptr_, bytes) == cudaSuccess);
    REQUIRE(cudaMemset(ptr_, 0, bytes) == cudaSuccess);
  }
  ~Device() { cudaFree(ptr_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

using Halves = std::vector<std::uint16_t>;

Halves random_halves(std::size_t count, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
  Halves out(count);
  for (std::uint16_t& v : out) v = kernels::ref::float_to_half(unit(rng));
  return out;
}

void upload_halves(void* device, const Halves& values, std::size_t first = 0) {
  REQUIRE(cudaMemcpy(static_cast<std::uint16_t*>(device) + first, values.data(), values.size() * 2,
                     cudaMemcpyHostToDevice) == cudaSuccess);
}

Halves download_halves(const void* device, std::size_t first, std::size_t count) {
  Halves out(count);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaMemcpy(out.data(), static_cast<const std::uint16_t*>(device) + first, count * 2,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return out;
}

// La moyenne 2 × 2 en FP32 arrondie en FP16 du pixel (y0, x0) d'un plan de largeur w.
std::uint16_t mean2x2(const std::uint16_t* plane, int w, int y0, int x0) {
  const auto at = [&](int y, int x) {
    return kernels::ref::half_to_float(
        plane[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]);
  };
  return kernels::ref::float_to_half((at(y0, x0) + at(y0, x0 + 1) + at(y0 + 1, x0) + at(y0 + 1, x0 + 1)) *
                                     0.25F);
}

TEST_CASE("DistilDRBA fusionné : les noyaux de réduction sont exacts",
          "[backend_tensorrt][cuda][drba_fused]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int h = 128;
  constexpr int w = 192;
  for (const int tail : {4, 8}) {
    INFO("réduction " << tail);
    const kernels::DrbaFusedPlan plan{.tail = tail};
    REQUIRE(kernels::drba_fused_size_valid(plan, h, w));
    const std::size_t area = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
    const Halves pair01 = random_halves(7 * area, 1);
    const Halves pair12 = random_halves(7 * area, 2);
    const kernels::DrbaFusedLayout layout = kernels::drba_fused_layout(plan, h, w);
    Device d01(7 * area * 2), d12(7 * area * 2), scratch(layout.bytes);
    upload_halves(d01.get(), pair01);
    upload_halves(d12.get(), pair12);
    auto* base = static_cast<std::byte*>(scratch.get());
    void* x[3]{base + layout.x[0], base + layout.x[1],
               base + layout.x[2]};  // NOLINT(modernize-avoid-c-arrays)

    REQUIRE(gpu::cuda::launch_drba_pre({.height = h,
                                        .width = w,
                                        .reduce0 = kernels::drba_fused_reduce(plan, 0),
                                        .reduce1 = kernels::drba_fused_reduce(plan, 1),
                                        .reduce2 = kernels::drba_fused_reduce(plan, 2),
                                        .pair01 = d01.get(),
                                        .pair12 = d12.get(),
                                        .x0 = x[0],
                                        .x1 = x[1],
                                        .x2 = x[2]},
                                       nullptr) == cudaSuccess);
    for (int k = 0; k < 3; ++k) {
      const int s = kernels::drba_fused_reduce(plan, k);
      const int oh = h / s;
      const int ow = w / s;
      const std::size_t plane = static_cast<std::size_t>(oh) * static_cast<std::size_t>(ow);
      const Halves got = download_halves(x[k], 0, 10 * plane);
      // Neuf plans d'images : img0 = pair01[0..2], img1 = pair01[3..5], img2 = pair12[3..5].
      std::size_t wrong = 0;
      for (int c = 0; c < 9; ++c) {
        const std::uint16_t* src = c < 3 ? pair01.data() + static_cast<std::size_t>(c) * area
                                         : (c < 6 ? pair01.data() + static_cast<std::size_t>(c) * area
                                                  : pair12.data() + static_cast<std::size_t>(c - 3) * area);
        for (int y = 0; y < oh; ++y)
          for (int xx = 0; xx < ow; ++xx)
            wrong +=
                got[static_cast<std::size_t>(c) * plane +
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(ow) +
                    static_cast<std::size_t>(xx)] != mean2x2(src, w, s * y + s / 2 - 1, s * xx + s / 2 - 1);
      }
      CHECK(wrong == 0);
      // Le plan du temps (canal 95 de X1 et X2, 57 de X0) : u × 2, un pixel sur s de pair01[6].
      const int time_channel = k == 0 ? 57 : 95;
      const Halves time = download_halves(x[k], static_cast<std::size_t>(time_channel) * plane, plane);
      std::size_t wrong_time = 0;
      for (int y = 0; y < oh; ++y)
        for (int xx = 0; xx < ow; ++xx) {
          const float u = kernels::ref::half_to_float(
              pair01[6 * area + static_cast<std::size_t>(s * y) * static_cast<std::size_t>(w) +
                     static_cast<std::size_t>(s * xx)]);
          wrong_time += time[static_cast<std::size_t>(y) * static_cast<std::size_t>(ow) +
                             static_cast<std::size_t>(xx)] != kernels::ref::float_to_half(2.0F * u);
        }
      CHECK(wrong_time == 0);
    }

    // Les encodages (sortie de l'encodeur, 48 plans au plan 9 de X2) réduits au demi et au quart.
    const int eh = h / tail;
    const int ew = w / tail;
    const std::size_t enc_area = static_cast<std::size_t>(eh) * static_cast<std::size_t>(ew);
    const Halves encodings = random_halves(48 * enc_area, 3);
    upload_halves(x[2], encodings, 9 * enc_area);
    REQUIRE(gpu::cuda::launch_drba_post(
                {.enc_height = eh, .enc_width = ew, .x2 = x[2], .x1 = x[1], .x0 = x[0]}, nullptr) ==
            cudaSuccess);
    for (const int d : {2, 4}) {
      const int oh = eh / d;
      const int ow = ew / d;
      const std::size_t plane = static_cast<std::size_t>(oh) * static_cast<std::size_t>(ow);
      void* target = d == 2 ? x[1] : x[0];
      const Halves got = download_halves(target, 9 * plane, 48 * plane);
      std::size_t wrong = 0;
      for (int c = 0; c < 48; ++c)
        for (int y = 0; y < oh; ++y)
          for (int xx = 0; xx < ow; ++xx)
            wrong += got[static_cast<std::size_t>(c) * plane +
                         static_cast<std::size_t>(y) * static_cast<std::size_t>(ow) +
                         static_cast<std::size_t>(xx)] !=
                     mean2x2(encodings.data() + static_cast<std::size_t>(c) * enc_area, ew, d * y + d / 2 - 1,
                             d * xx + d / 2 - 1);
      CHECK(wrong == 0);
    }
  }
}

// PSNR (dB) de got contre want, le pic étant 1 (images de 0 à 1) ; écart maximal en sortie.
double psnr_of(const std::vector<float>& got, const std::vector<float>& want, double& worst) {
  REQUIRE(got.size() == want.size());
  double square = 0.0;
  worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
    square += d * d;
    worst = std::max(worst, std::abs(d));
  }
  const double mse = square / static_cast<double>(got.size());
  return mse == 0.0 ? 200.0 : 10.0 * std::log10(1.0 / mse);
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

// Les trois images en plans RGB h × w : une fenêtre de l'image PPM puis la même décalée de 6 et de 12 pixels,
// ou trois dessins décalés.
std::array<std::vector<float>, 3> picture_triple(int h, int w) {
  constexpr int kShift = 6;
  if (const fs::path image(AJP_DRBA_IMAGE); !image.empty() && fs::is_regular_file(image)) {
    int ih = 0, iw = 0;
    const std::vector<float> whole = read_ppm(image, ih, iw);
    if (!whole.empty() && ih >= h && iw >= w + 2 * kShift) {
      const auto window = [&](int x0) {
        std::vector<float> out(static_cast<std::size_t>(3 * h * w));
        for (int c = 0; c < 3; ++c)
          for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
              out[static_cast<std::size_t>((c * h + y) * w + x)] =
                  whole[static_cast<std::size_t>((c * ih + y) * iw + x0 + x)];
        return out;
      };
      return {window(0), window(kShift), window(2 * kShift)};
    }
  }
  return {drawing(h, w, 0), drawing(h, w, kShift), drawing(h, w, 2 * kShift)};
}

// Vrai si AJP_DRBA_IMAGE fournit les trois fenêtres h × w.
bool picture_available(int h, int w) {
  const fs::path image(AJP_DRBA_IMAGE);
  if (image.empty() || !fs::is_regular_file(image)) return false;
  int ih = 0, iw = 0;
  return !read_ppm(image, ih, iw).empty() && ih >= h && iw >= w + 12;
}

// pair01 = (image 0, image 1, tau / 2), pair12 = (image 1, image 2, tau / 2), comme rife_input en 7 plans.
std::pair<std::vector<float>, std::vector<float>> pairs_of(const std::array<std::vector<float>, 3>& images,
                                                           int h, int w, float tau) {
  const std::size_t area = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  std::vector<float> a(7 * area, tau / 2);
  std::vector<float> b(7 * area, tau / 2);
  std::copy(images[0].begin(), images[0].end(), a.begin());
  std::copy(images[1].begin(), images[1].end(), a.begin() + static_cast<std::ptrdiff_t>(3 * area));
  std::copy(images[1].begin(), images[1].end(), b.begin());
  std::copy(images[2].begin(), images[2].end(), b.begin() + static_cast<std::ptrdiff_t>(3 * area));
  return {std::move(a), std::move(b)};
}

gpu::ShapeSet fixed_input(int h, int w) {
  const gpu::Shape shape{.dims = {1, 7, h, w}};
  return {.min = shape, .opt = shape, .max = shape};
}

// Compare la chaîne fusionnée au DistilDRBA entier de TensorRT sur une paire de paires, des deux côtés.
void compare_engines(const std::string& file, int h, int w) {
  const fs::path onnx = fs::path(AJP_DRBA_MODEL_DIR) / file;
  if (!fs::is_regular_file(onnx)) SKIP("modèle absent : " << onnx.string());
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  REQUIRE(loaded.backend->capabilities().drba_fused);
  const fs::path scratch = fs::temp_directory_path() / "ajp-drba-fused-test";
  fs::create_directories(scratch);
  gpu::Progress progress;

  // Le moteur entier, sur le modèle aux dimensions fixées comme le lecteur le charge.
  const auto bytes = prepare::read_onnx_bytes(onnx);
  REQUIRE(bytes);
  const std::vector<prepare::NamedDim> dims{{"H", h}, {"W", w}};
  const auto fixed = prepare::with_fixed_dims(*bytes, dims);
  REQUIRE(fixed);
  const fs::path static_onnx =
      scratch / (onnx.stem().string() + "-" + std::to_string(w) + "x" + std::to_string(h) + ".onnx");
  {
    std::ofstream out(static_onnx, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(fixed->data()), static_cast<std::streamsize>(fixed->size()));
  }
  const auto sha = prepare::sha256_file(static_onnx);
  REQUIRE(sha);
  auto single = loaded.backend->build(
      gpu::ModelSpec{.id = static_onnx.stem().string(), .onnx_path = static_onnx, .sha256 = *sha},
      fixed_input(h, w), progress);
  REQUIRE(single);
  auto reference = loaded.backend->load(*single);
  REQUIRE(reference);

  const auto manifest = prepare::derive_drba_fused(onnx, scratch, h, w);
  REQUIRE(manifest);
  const auto manifest_sha = prepare::sha256_file(*manifest);
  REQUIRE(manifest_sha);
  auto fused_artifact = loaded.backend->build(
      gpu::ModelSpec{.id = manifest->stem().string(), .onnx_path = *manifest, .sha256 = *manifest_sha},
      fixed_input(h, w), progress);
  REQUIRE(fused_artifact);
  auto fused = loaded.backend->load(*fused_artifact);
  REQUIRE(fused);
  REQUIRE((*fused)->input_names().size() == 2);
  CHECK((*fused)->input_names()[0] == "pair01");
  CHECK((*fused)->input_names()[1] == "pair12");

  // À 4K les traits d'un pixel du dessin synthétique se déplacent d'un pixel au moindre pas de la grille FP16
  // (0,9 pixel à 3840 de large) : la comparaison de deux réalisations du même graphe y tombe vers 42 dB, sans
  // lien avec la chaîne. Une vraie image (AJP_DRBA_IMAGE, agrandie pour couvrir la taille) est exigée à cette
  // taille.
  if (w > 1920 && !picture_available(h, w))
    SKIP("4K : une image d'anime (AJP_DRBA_IMAGE) assez grande est nécessaire");
  const auto images = picture_triple(h, w);
  const gpu::Shape seven{.dims = {1, 7, h, w}};
  const gpu::Shape rgb{.dims = {1, 3, h, w}};
  auto in01 = loaded.backend->alloc_tensor(gpu::TensorType::F16, seven);
  auto in12 = loaded.backend->alloc_tensor(gpu::TensorType::F16, seven);
  REQUIRE(in01);
  REQUIRE(in12);
  const std::array inputs{(*in01)->tensor(), (*in12)->tensor()};
  const auto run = [&](gpu::Engine& engine) {
    auto buffer = loaded.backend->alloc_tensor(gpu::TensorType::F16, rgb);
    REQUIRE(buffer);
    std::vector<gpu::Tensor> outs{(*buffer)->tensor()};
    // Trois lancements de suite : exécution à froid, capture du graphe CUDA, rejeu ; le dernier est celui
    // qu'on lit.
    for (int i = 0; i < 3; ++i) REQUIRE(loaded.backend->enqueue(engine, inputs, outs));
    return download(outs[0]);
  };
  for (const float tau : {0.5F, 1.25F}) {
    const auto [a, b] = pairs_of(images, h, w, tau);
    upload(inputs[0], a);
    upload(inputs[1], b);
    const auto want = run(**reference);
    const auto got = run(**fused);
    double worst = 0.0;
    const double score = psnr_of(got, want, worst);
    WARN(file << " tau " << tau << " : PSNR " << score << " dB contre TensorRT entier, écart maximal "
              << worst);
    CHECK(score >= 50.0);
  }
}

}  // namespace

TEST_CASE("DistilDRBA fusionné : la chaîne contre le moteur DistilDRBA entier de TensorRT",
          "[.drba_fused][backend_tensorrt][cuda]") {
  SECTION("v2 lite, flot à l'échelle 1") {
    compare_engines("distil_drba_v2_lite.onnx", 384, 640);
  }
  SECTION("v2 lite, flot à l'échelle 0,5") {
    // Le modèle d'échelle 0,5 du lecteur est dérivé à 2176 × 3840 : la chaîne se compare à cette taille.
    compare_engines("distil_drba_v2_lite_s05.onnx", 2176, 3840);
  }
}

}  // namespace ajp::test
