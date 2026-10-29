// GMFSS fusionné (prepare/gmfss_split.h) : la paire coupée en deux moteurs TensorRT, les caractéristiques
// d'une image et le flot, contre la paire entière, sur de vraies images d'anime. Test caché :
// `ajp_backend_tensorrt_tests "[.gmfss_fused]"`, sous `ajp-gpu`. Le modèle du lecteur est lu en place
// (AJP_GMFSS_MODEL_DIR : gmfss_fortuna_base_s05_pair.onnx et _s05_fusion.onnx) ; les images, des PPM P6 de
// 640×384 séparés par des virgules dans AJP_GMFSS_IMAGES (dix : cinq images consécutives de la vidéo à deux
// instants), sinon SKIP : un dessin synthétique ne prouve rien ici.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "kernels/spec/gmfss.h"
#include "prepare/gmfss_split.h"
#include "prepare/model_source.h"
#include "prepare/onnx_dims.h"
#include "prepare/sha256.h"

#ifndef AJP_GMFSS_MODEL_DIR
#define AJP_GMFSS_MODEL_DIR ""
#endif
#ifndef AJP_GMFSS_IMAGES
#define AJP_GMFSS_IMAGES ""
#endif

namespace ajp::test {
namespace {

namespace fs = std::filesystem;
namespace k = ajp::kernels;

constexpr int kWidth = 640;
constexpr int kHeight = 384;

// Une image en PPM binaire (P6, 255) en plans RGB de 0 à 1 ; vide si illisible ou d'une autre taille.
std::vector<float> read_ppm(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int w = 0;
  int h = 0;
  int maxval = 0;
  file >> magic >> w >> h >> maxval;
  if (!file || magic != "P6" || maxval != 255 || w != kWidth || h != kHeight) return {};
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

std::vector<fs::path> image_paths() {
  std::vector<fs::path> out;
  std::stringstream list{std::string(AJP_GMFSS_IMAGES)};
  for (std::string item; std::getline(list, item, ',');) {
    if (!item.empty()) out.emplace_back(item);
  }
  return out;
}

// PSNR (pics à 1) et SSIM (blocs de 8 × 8 de la luma, sans fenêtre gaussienne) de deux images 3 × h × w.
std::pair<double, double> compare(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double squares = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    squares += d * d;
  }
  const double mse = squares / static_cast<double>(a.size());
  const std::size_t area = static_cast<std::size_t>(kWidth) * kHeight;
  const auto luma = [&](const std::vector<float>& v, std::size_t i) {
    return 0.299 * static_cast<double>(v[i]) + 0.587 * static_cast<double>(v[area + i]) +
           0.114 * static_cast<double>(v[2 * area + i]);
  };
  double ssim = 0.0;
  int blocks = 0;
  for (int by = 0; by + 8 <= kHeight; by += 8) {
    for (int bx = 0; bx + 8 <= kWidth; bx += 8) {
      double ma = 0.0, mb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0;
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
          const std::size_t i = static_cast<std::size_t>((by + y) * kWidth + bx + x);
          const double la = luma(a, i);
          const double lb = luma(b, i);
          ma += la;
          mb += lb;
          saa += la * la;
          sbb += lb * lb;
          sab += la * lb;
        }
      ma /= 64.0;
      mb /= 64.0;
      const double va = saa / 64.0 - ma * ma;
      const double vb = sbb / 64.0 - mb * mb;
      const double cov = sab / 64.0 - ma * mb;
      constexpr double c1 = 0.01 * 0.01;
      constexpr double c2 = 0.03 * 0.03;
      ssim += (2.0 * ma * mb + c1) * (2.0 * cov + c2) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
      ++blocks;
    }
  }
  return {mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 200.0, ssim / blocks};
}

}  // namespace

TEST_CASE("GMFSS fusionné : la chaîne (caractéristiques gardées + flot) contre la paire entière de TensorRT",
          "[.gmfss_fused][backend_tensorrt][cuda][gmfss]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const fs::path dir = AJP_GMFSS_MODEL_DIR;
  const fs::path pair_onnx = dir / "gmfss_fortuna_base_s05_pair.onnx";
  const fs::path fusion_onnx = dir / "gmfss_fortuna_base_s05_fusion.onnx";
  if (!fs::is_regular_file(pair_onnx) || !fs::is_regular_file(fusion_onnx))
    SKIP("modèle GMFSS du lecteur absent (AJP_GMFSS_MODEL_DIR)");
  const std::vector<fs::path> paths = image_paths();
  if (paths.size() != 10) SKIP("dix images PPM 640x384 attendues (AJP_GMFSS_IMAGES)");
  std::vector<std::vector<float>> images;
  for (const fs::path& path : paths) {
    images.push_back(read_ppm(path));
    if (images.back().empty())
      SKIP("image PPM illisible ou d'une autre taille que 640x384 : " << path.string());
  }
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  REQUIRE(loaded.backend->capabilities().gmfss_fused);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  const fs::path scratch = fs::temp_directory_path() / "ajp-gmfss-fused-test";
  fs::create_directories(scratch);

  // Les graphes d'origine aux dimensions fixées, comme le lecteur les charge (gmfss_dims).
  const std::vector<prepare::NamedDim> dims{{"H", kHeight},       {"W", kWidth},        {"H/2", kHeight / 2},
                                            {"W/2", kWidth / 2},  {"H/4", kHeight / 4}, {"W/4", kWidth / 4},
                                            {"H/8", kHeight / 8}, {"W/8", kWidth / 8}};
  const auto fixed = [&](const fs::path& onnx) {
    const auto bytes = prepare::read_onnx_bytes(onnx);
    REQUIRE(bytes);
    const auto cut = prepare::with_fixed_dims(*bytes, dims);
    REQUIRE(cut);
    const fs::path target = scratch / std::format("{}-{}x{}.onnx", onnx.stem().string(), kWidth, kHeight);
    std::ofstream out(target, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(cut->data()), static_cast<std::streamsize>(cut->size()));
    return target;
  };
  const auto engine = [&](const fs::path& file, const gpu::Shape& shape) {
    gpu::Progress progress;
    const auto sha = prepare::sha256_file(file);
    REQUIRE(sha);
    auto artifact =
        loaded.backend->build(gpu::ModelSpec{.id = file.stem().string(), .onnx_path = file, .sha256 = *sha},
                              {.min = shape, .opt = shape, .max = shape}, progress);
    REQUIRE(artifact);
    auto made = loaded.backend->load(*artifact);
    REQUIRE(made);
    return std::move(*made);
  };
  const auto whole = engine(fixed(pair_onnx), {.dims = {1, k::kGmfssFramePlanes, kHeight, kWidth}});
  const auto fusion = engine(fixed(fusion_onnx), {.dims = {1, 3, kHeight / 2, kWidth / 2}});
  const auto derived = prepare::derive_gmfss_fused(pair_onnx, scratch, kHeight, kWidth);
  REQUIRE(derived);
  const auto features = engine(derived->features, {.dims = {1, 3, kHeight, kWidth}});
  const auto flow = engine(derived->flow, {.dims = {1, k::kGmfssFramePlanes, kHeight, kWidth}});

  // Les tenseurs par nom et par chaîne (« entier/ », « fusionne/ »).
  std::vector<std::pair<std::string, std::unique_ptr<gpu::TensorBuffer>>> tensors;
  const auto shape_of = [&](const std::string& name) {
    int level = 1;
    int channels = 3;
    if (!name.starts_with("half")) {
      level = name.back() - '0';
      channels = name.starts_with("motion")    ? k::kGmfssMotionChannels
                 : name.starts_with("feature") ? 64
                                               : k::gmfss_source_channels(level);
    }
    return gpu::Shape{.dims = {1, channels, kHeight >> level, kWidth >> level}};
  };
  const auto tensor = [&](const std::string& name, const gpu::Shape& shape) -> const gpu::Tensor& {
    for (const auto& [known, buffer] : tensors) {
      if (known == name) return buffer->tensor();
    }
    auto buffer = loaded.backend->alloc_tensor(gpu::TensorType::F16, shape);
    REQUIRE(buffer);
    tensors.emplace_back(name, std::move(*buffer));
    return tensors.back().second->tensor();
  };
  const auto slice = [](gpu::Tensor part, std::int64_t first, std::int64_t count) {
    const std::int64_t plane = part.shape.dims[2] * part.shape.dims[3];
    part.shape.dims[1] = count;
    part.offset += static_cast<std::uint64_t>(first * plane) * sizeof(std::uint16_t);
    return part;
  };
  auto frames = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 6, kHeight, kWidth}});
  REQUIRE(frames);
  const auto load_frames = [&](int a, int b) {
    std::vector<float> both = images[static_cast<std::size_t>(a)];
    both.insert(both.end(), images[static_cast<std::size_t>(b)].begin(),
                images[static_cast<std::size_t>(b)].end());
    upload((*frames)->tensor(), both);
  };

  // Les caractéristiques de chaque tranche : de quelle image (indice dans images), -1 : aucune.
  std::array<int, 2> held{-1, -1};
  int features_run = 0;
  // Une paire (a, b) à l'instant t par la chaîne : fused, les caractéristiques de a et b dans les tranches
  // a_slot et b_slot (celles de a restent si la tranche les porte déjà), le flot sans elles ; sinon la paire
  // entière. Rend l'image de sortie.
  const auto render = [&](const std::string& tag, bool fused, int a, int b, std::size_t a_slot, float t) {
    load_frames(a, b);
    const std::size_t b_slot = a_slot ^ 1U;
    gpu::Engine& pair = fused ? *flow : *whole;
    if (fused) {
      for (const auto& [image, slot, plane] : {std::tuple{a, a_slot, 0}, std::tuple{b, b_slot, 3}}) {
        if (held[slot] == image) continue;
        std::vector<gpu::Tensor> outputs;
        for (const std::string& name : features->output_names()) {
          const std::string wanted = name.starts_with("feature")
                                         ? std::format("feature{}_1", slot)
                                         : std::format("source{}_{}", slot, name.back() - '0');
          outputs.push_back(tensor(tag + wanted, shape_of(wanted)));
        }
        const std::array input{slice((*frames)->tensor(), plane, 3)};
        REQUIRE(loaded.backend->enqueue(*features, input, outputs));
        held[slot] = image;
        ++features_run;
      }
    }
    std::vector<gpu::Tensor> outputs;
    for (const std::string& name : pair.output_names()) outputs.push_back(tensor(tag + name, shape_of(name)));
    const std::array inputs{(*frames)->tensor()};
    REQUIRE(loaded.backend->enqueue(pair, inputs, outputs));
    std::vector<gpu::KernelOps::SplatJob> jobs;
    for (std::size_t side = 0; side < 2; ++side) {
      const std::size_t holder = fused ? (side == 0 ? a_slot : b_slot) : side;
      for (int level = 1; level <= k::kGmfssLevels; ++level) {
        const gpu::Tensor& warped = tensor(
            std::format("{}warped{}_{}", tag, side, level),
            gpu::Shape{.dims = {1, k::gmfss_source_channels(level), kHeight >> level, kWidth >> level}});
        const gpu::Tensor& motion = tensor(std::format("{}motion{}_{}", tag, side, level),
                                           shape_of(std::format("motion{}_{}", side, level)));
        const float s = k::gmfss_side_time(static_cast<int>(side), t);
        if (level == 1) {
          jobs.push_back({.source = tensor(std::format("{}half{}", tag, side), shape_of("half0")),
                          .motion = motion,
                          .s = s,
                          .out = slice(warped, 0, 3)});
          jobs.push_back({.source = tensor(std::format("{}feature{}_1", tag, holder), shape_of("feature0_1")),
                          .motion = motion,
                          .s = s,
                          .out = slice(warped, 3, 64)});
        } else {
          jobs.push_back(
              {.source = tensor(std::format("{}source{}_{}", tag, holder, level), shape_of("source0_2")),
               .motion = motion,
               .s = s,
               .out = warped});
        }
      }
    }
    REQUIRE(ops.soft_splat_all(jobs));
    std::vector<gpu::Tensor> fusion_inputs;
    for (const std::string& name : fusion->input_names()) {
      fusion_inputs.push_back(
          name.starts_with("half")
              ? tensor(tag + name, shape_of(name))
              : tensor(tag + name,
                       gpu::Shape{.dims = {1, k::gmfss_source_channels(name.back() - '0'),
                                           kHeight >> (name.back() - '0'), kWidth >> (name.back() - '0')}}));
    }
    auto image = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, kHeight, kWidth}});
    REQUIRE(image);
    std::array results{(*image)->tensor()};
    REQUIRE(loaded.backend->enqueue(*fusion, fusion_inputs, results));
    return download((*image)->tensor());
  };

  // Deux instants de la vidéo, cinq images consécutives chacun ; des paires à une image d'écart (0, 1, 2) et
  // à deux (0, 2, 4), chaque suite jouée comme la lecture : (i0, i1) puis (i1, i2), dont la seconde reprend
  // les caractéristiques de i1.
  struct Sequence {
    std::array<int, 3> images;
    const char* name;
  };
  const std::array sequences{
      Sequence{{0, 1, 2}, "960 s, 1 image d'écart"}, Sequence{{0, 2, 4}, "960 s, 2 images"},
      Sequence{{5, 6, 7}, "1010 s, 1 image d'écart"}, Sequence{{5, 7, 9}, "1010 s, 2 images"}};
  double worst_psnr = 1000.0;
  double worst_ssim = 1.0;
  int cases = 0;
  for (const Sequence& seq : sequences) {
    held = {-1, -1};
    for (std::size_t n = 0; n < 2; ++n) {
      const int a = seq.images[n];
      const int b = seq.images[n + 1];
      for (const float t : {0.25F, 0.5F, 0.75F}) {
        const std::size_t a_slot = n % 2;
        const std::vector<float> reference = render("entier/", false, a, b, a_slot, t);
        const int before = features_run;
        const std::vector<float> chained = render("fusionne/", true, a, b, a_slot, t);
        // La paire (i1, i2) reprend i1 dans sa tranche : seule i2 est calculée (la première fois du tour de
        // t).
        if (n == 1 && t == 0.25F) CHECK(features_run - before == 1);
        if (t != 0.25F) CHECK(features_run == before);  // les autres instants de la paire ne refont rien
        const auto [psnr, ssim] = compare(reference, chained);
        WARN(std::format("{} paire {} t {} : PSNR {:.2f} dB, SSIM {:.6f}", seq.name, n, t, psnr, ssim));
        worst_psnr = std::min(worst_psnr, psnr);
        worst_ssim = std::min(worst_ssim, ssim);
        CHECK(psnr >= 50.0);
        ++cases;
      }
    }
  }
  WARN(std::format("{} comparaisons : PSNR minimal {:.2f} dB, SSIM minimal {:.6f}", cases, worst_psnr,
                   worst_ssim));
  // Le bruit propre de la chaîne entière (l'ordre des additions atomiques) pour situer les écarts.
  const std::vector<float> first = render("entier/", false, 0, 1, 0, 0.5F);
  const std::vector<float> second = render("bruit/", false, 0, 1, 0, 0.5F);
  WARN(std::format("paire entière contre elle-même : PSNR {:.2f} dB", compare(first, second).first));
}

}  // namespace ajp::test
