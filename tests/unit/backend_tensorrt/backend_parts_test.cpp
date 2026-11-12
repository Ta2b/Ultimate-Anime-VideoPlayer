// Les pièces du backend TensorRT testées seules (B01-09) : la zone d'activation commune avec un pool simulé
// (sans carte), la disposition des plans envoyés depuis l'hôte, et les paires de minutage (avec carte).
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "gpu/cuda/backend/gpu_timing.h"
#include "gpu/cuda/backend/host_upload.h"
#include "gpu/cuda/backend/shared_activations.h"
#include "gpu/cuda/backend/tensorrt_backend.h"

namespace {

using namespace ajp;
using namespace ajp::gpu::cuda;

// Un pool simulé : des blocs de l'hôte, un plafond d'octets tenus en même temps, chaque libération notée.
struct FakePool {
  std::size_t ceiling = 0;
  std::size_t held = 0;
  std::vector<std::unique_ptr<std::byte[]>> blocks;
  std::vector<std::size_t> sizes;
  std::vector<void*> released;

  void attach(SharedActivations& zone) {
    zone.set_pool(
        [this](std::size_t bytes) -> void* {
          if (held + bytes > ceiling) return nullptr;
          held += bytes;
          blocks.push_back(std::make_unique<std::byte[]>(bytes));
          sizes.push_back(bytes);
          return blocks.back().get();
        },
        [this](void* memory) {
          for (std::size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i].get() == memory) held -= sizes[i];
          }
          released.push_back(memory);
        });
  }
};

constexpr std::int64_t kMiB = std::int64_t{1} << 20;

}  // namespace

TEST_CASE("Activations partagées : la zone grandit, l'ancienne est rendue et la génération change (B01-09)",
          "[backend_tensorrt]") {
  FakePool pool{.ceiling = 8 << 20};
  {
    SharedActivations zone;
    pool.attach(zone);
    REQUIRE(zone.reserve(kMiB));
    void* first = zone.data();
    const std::uint64_t generation = zone.generation();
    REQUIRE(zone.reserve(kMiB / 2));  // plus petit : rien ne bouge
    CHECK(zone.data() == first);
    CHECK(zone.generation() == generation);
    REQUIRE(zone.reserve(2 * kMiB));
    CHECK(zone.bytes() == 2 * kMiB);
    CHECK(zone.generation() != generation);
    REQUIRE(pool.released.size() == 1);
    CHECK(pool.released.front() == first);
  }
  CHECK(pool.released.size() == 2);  // la dernière zone rendue à la destruction
  CHECK(pool.held == 0);
}

TEST_CASE("Activations partagées : un refus d'agrandir garde la zone des moteurs déjà posés (B01-01)",
          "[backend_tensorrt]") {
  FakePool pool{.ceiling = 5 << 19};  // 2,5 Mo : assez pour 2 Mo, pas pour 1 + 2 ni pour 4
  SharedActivations zone;
  pool.attach(zone);
  REQUIRE(zone.reserve(kMiB));
  // 1 + 2 Mo refusés ensemble : l'ancienne part, la nouvelle passe seule.
  REQUIRE(zone.reserve(2 * kMiB));
  CHECK(zone.bytes() == 2 * kMiB);
  // 4 Mo refusés même seuls : refus, et la zone revient à sa taille d'avant pour SPAN déjà posé.
  const std::uint64_t generation = zone.generation();
  const auto refused = zone.reserve(4 * kMiB);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::GpuMemoryExhausted);
  CHECK(zone.data() != nullptr);
  CHECK(zone.bytes() == 2 * kMiB);
  CHECK(zone.generation() != generation);  // nouvelle adresse : les moteurs se reposent dessus
}

TEST_CASE("Activations partagées : sans mémoire du tout, refus et zone vide", "[backend_tensorrt]") {
  FakePool pool{.ceiling = 0};
  SharedActivations zone;
  pool.attach(zone);
  const auto refused = zone.reserve(kMiB);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::GpuMemoryExhausted);
  CHECK(zone.data() == nullptr);
  CHECK(zone.bytes() == 0);
}

TEST_CASE(
    "Activations partagées : l'encodage gardé tient tant qu'aucun moteur n'écrit au-dessus de lui (C03-01)",
    "[backend_tensorrt]") {
  SharedActivations zone;
  int rife = 0, span = 0, drba = 0;  // des adresses qui nomment les moteurs
  zone.used(&span, 40 * kMiB);       // SPAN sur la même zone : ses activations
  CHECK(zone.activation_peak() == 40 * kMiB);
  // RIFE fusionné pose ses tampons au-dessus du plus haut moteur entier, et les garde.
  const std::int64_t from = zone.activation_peak();
  zone.wrote(&rife, from + 8 * kMiB);
  zone.keep(&rife, from);
  CHECK(zone.kept_by(&rife));
  CHECK_FALSE(zone.kept_by(&drba));
  // SPAN repasse entre deux appels de RIFE (une seule voie) : il n'écrit que sous les tampons gardés.
  zone.used(&span, 40 * kMiB);
  CHECK(zone.kept_by(&rife));
  // RIFE lui-même réécrit ses tampons : toujours à lui.
  zone.wrote(&rife, from + 8 * kMiB);
  CHECK(zone.kept_by(&rife));
  // Un moteur entier plus gros écrit par-dessus : l'encodage est perdu, RIFE le refait.
  zone.used(&span, 50 * kMiB);
  CHECK_FALSE(zone.kept_by(&rife));
  CHECK(zone.activation_peak() == 50 * kMiB);
  // DistilDRBA fusionné qui écrit au-dessus de la limite fait de même.
  zone.keep(&rife, 50 * kMiB);
  zone.wrote(&drba, 50 * kMiB + 1);
  CHECK_FALSE(zone.kept_by(&rife));
  CHECK_FALSE(zone.kept_by(nullptr));
}

TEST_CASE("Activations partagées : deux moteurs fusionnés, deux régions, deux encodages gardés",
          "[backend_tensorrt]") {
  SharedActivations zone;
  int full = 0, half = 0, span = 0;
  zone.used(&span, 40 * kMiB);
  // RIFE plein puis RIFE demi-taille (échelle par paire) : la seconde région au-dessus de la première.
  const std::int64_t a = zone.region_offset(&full, 10 * kMiB, 100 * kMiB + 3);
  const std::int64_t b = zone.region_offset(&half, 5 * kMiB, 25 * kMiB);
  CHECK(a == 40 * kMiB);
  CHECK(b == 140 * kMiB + 256);  // la fin de la première, alignée sur 256 octets
  CHECK(zone.region_offset(&full, 10 * kMiB, 100 * kMiB + 3) == a);  // stable d'un appel à l'autre
  zone.wrote(&full, a, a + 100 * kMiB + 3);
  zone.keep(&full, a, a + 100 * kMiB + 3);
  zone.wrote(&half, b, b + 25 * kMiB);
  zone.keep(&half, b, b + 25 * kMiB);
  CHECK(zone.kept_by(&full));  // le demi-taille n'a pas écrit sur l'encodage du plein
  CHECK(zone.kept_by(&half));
  zone.used(&span, 40 * kMiB);  // SPAN sous les deux régions
  CHECK(zone.kept_by(&full));
  CHECK(zone.kept_by(&half));
  // Des activations plus hautes : les deux régions remontent, et l'écriture efface ce qu'elle recouvre.
  zone.used(&span, 60 * kMiB);
  CHECK_FALSE(zone.kept_by(&full));
  CHECK(zone.kept_by(&half));
  CHECK(zone.region_offset(&full, 10 * kMiB, 100 * kMiB + 3) == 60 * kMiB);
  // Le plein parti, le demi-taille descend à sa place.
  zone.drop_region(&full);
  CHECK(zone.region_offset(&half, 5 * kMiB, 25 * kMiB) == 60 * kMiB);
  CHECK_FALSE(zone.kept_by(&full));
}

TEST_CASE("Voies de calcul : deux voies à partir de 7,5 Gio de mémoire et 5 Gio libres (C01-09)",
          "[backend_tensorrt]") {
  constexpr std::size_t kGiB = std::size_t{1} << 30;
  constexpr int lanes = static_cast<int>(gpu::kLanes);
  REQUIRE(lanes > 1);
  CHECK(TensorRtBackend::lane_count(8 * kGiB, 7 * kGiB) == lanes);
  CHECK(TensorRtBackend::lane_count(15 * kGiB / 2, 5 * kGiB) == lanes);  // les deux bornes comprises
  CHECK(TensorRtBackend::lane_count(15 * kGiB / 2 - 1, 7 * kGiB) == 1);  // une 6 Go, ou une 8 Go qui ment
  CHECK(TensorRtBackend::lane_count(6 * kGiB, 6 * kGiB) == 1);
  CHECK(TensorRtBackend::lane_count(16 * kGiB, 5 * kGiB - 1) == 1);  // un jeu tient déjà la carte
  CHECK(TensorRtBackend::lane_count(0, 0) == 1);
}

TEST_CASE("Envois depuis l'hôte : disposition des plans, tailles impaires comprises (B01-09)",
          "[backend_tensorrt]") {
  constexpr int w = 1919;
  constexpr int h = 1079;
  constexpr std::size_t cw = 960;
  constexpr std::size_t ch = 540;
  using Layout = HostUploads::Layout;
  const auto check = [](PixelFormat format, Layout want) {
    CAPTURE(static_cast<int>(format));
    const auto got = HostUploads::layout(format, w, h);
    REQUIRE(got);
    CHECK(got->planes == want.planes);
    CHECK(got->row_bytes == want.row_bytes);
    CHECK(got->rows == want.rows);
  };
  check(PixelFormat::Nv12, {.planes = 2, .row_bytes = {w, 2 * cw, 0}, .rows = {h, ch, 0}});
  check(PixelFormat::P010, {.planes = 2, .row_bytes = {2 * w, 4 * cw, 0}, .rows = {h, ch, 0}});
  check(PixelFormat::P016, {.planes = 2, .row_bytes = {2 * w, 4 * cw, 0}, .rows = {h, ch, 0}});
  check(PixelFormat::Yuv420p, {.planes = 3, .row_bytes = {w, cw, cw}, .rows = {h, ch, ch}});
  check(PixelFormat::Yuv420p10, {.planes = 3, .row_bytes = {2 * w, 2 * cw, 2 * cw}, .rows = {h, ch, ch}});
  // 4:2:2 : la moitié en largeur seulement ; 4:4:4 : la chroma entière.
  check(PixelFormat::Yuv422p16, {.planes = 3, .row_bytes = {2 * w, 2 * cw, 2 * cw}, .rows = {h, h, h}});
  check(PixelFormat::Yuv444p16, {.planes = 3, .row_bytes = {2 * w, 2 * w, 2 * w}, .rows = {h, h, h}});
  CHECK_FALSE(HostUploads::layout(PixelFormat::Rgba16Unorm, w, h));
  CHECK_FALSE(HostUploads::layout(PixelFormat::Nv12, 0, h));
  CHECK_FALSE(HostUploads::layout(PixelFormat::Nv12, w, -1));
}

TEST_CASE("Minutage GPU : une paire restée ouverte ou finie sur une autre file ne compte pas (B01-06)",
          "[backend_tensorrt][cuda]") {
  if (!test::has_cuda()) SKIP("pas de GPU NVIDIA");
  cudaStream_t one = nullptr;
  cudaStream_t other = nullptr;
  REQUIRE(cudaStreamCreateWithFlags(&one, cudaStreamNonBlocking) == cudaSuccess);
  REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
  {
    GpuTiming timing("essai", {});
    timing.begin(one);  // étape sortie en erreur : jamais refermée
    timing.begin(one);
    timing.end(one);
    timing.begin(one);
    timing.end(other);  // fin sur une autre voie : abandonnée
    timing.end(one);    // déjà abandonnée : rien
    timing.begin(one);
    timing.end(one);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(timing.profile().count == 2);
  }
  CHECK(cudaStreamDestroy(one) == cudaSuccess);
  CHECK(cudaStreamDestroy(other) == cudaSuccess);
}
