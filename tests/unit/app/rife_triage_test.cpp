// Tri des paires de RIFE par la carte de la scène (rife_triage.h) : un seul test pour le module, qui fait
// passer les mêmes paires sans carte, sans le réglage, puis avec la carte (tenue, impossible, douteuse), et
// pour GMFSS seul (fondu et panoramique pur par la caméra, jamais par endroits).
#include "app/play/rife_triage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/spec/motion_camera.h"
#include "kernels/spec/motion_doubt.h"

namespace kirari::app::play {
namespace {

// 8 × 6 tuiles de dessin sûres et immobiles : une scène calme.
kernels::SceneMotion calm_scene() {
  kernels::SceneMotion s;
  s.camera_fit = true;
  s.tile_px = 32;
  s.cols = 8;
  s.rows = 6;
  s.tiles.assign(48, kernels::SceneTile{.kind = kernels::SceneClass::Drawing, .confidence = 0.9F});
  return s;
}

// Un tenseur à une adresse à lui, sans mémoire.
class NoMemory final : public gpu::TensorBuffer {
 public:
  explicit NoMemory(gpu::Tensor tensor) : tensor_(tensor) {}
  const gpu::Tensor& tensor() const override { return tensor_; }

 private:
  gpu::Tensor tensor_;
};

// Un backend sans GPU : il compte ses allocations et note le flot écrit par chaque fondu du mouvement.
class DoubtBackend final : public gpu::Backend, public gpu::KernelOps {
 public:
  int allocated = 0;
  std::vector<std::uintptr_t> written;

  static std::unexpected<Refusal> refused() {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "factice"});
  }

  gpu::BackendCaps capabilities() const override { return {}; }
  gpu::DeviceMatch match(const DeviceUuid& /*render_device*/) const override {
    return gpu::DeviceMatch::Same;
  }
  std::expected<gpu::Artifact, Refusal> build(const gpu::ModelSpec& /*model*/,
                                              const gpu::ShapeSet& /*shapes*/,
                                              gpu::Progress& /*progress*/) override {
    return refused();
  }
  std::expected<std::unique_ptr<gpu::Engine>, Refusal> load(const gpu::Artifact& /*artifact*/) override {
    return refused();
  }
  std::expected<std::unique_ptr<gpu::TensorBuffer>, Refusal> alloc_tensor(gpu::TensorType type,
                                                                          const gpu::Shape& shape) override {
    ++allocated;
    return std::make_unique<NoMemory>(
        gpu::Tensor{.type = type, .shape = shape, .handle = static_cast<std::uintptr_t>(allocated)});
  }
  gpu::KernelOps* kernels() override { return this; }
  std::expected<void, Refusal> enqueue(gpu::Engine& /*engine*/, std::span<const gpu::Tensor> /*inputs*/,
                                       std::span<gpu::Tensor> /*outputs*/) override {
    return refused();
  }
  std::expected<gpu::Fence, Refusal> signal() override { return gpu::Fence{}; }
  std::expected<void, Refusal> wait(const gpu::Fence& /*fence*/) override { return {}; }
  std::expected<void, Refusal> ingest(const DecodedFrame& /*frame*/, const gpu::IngestSpec& /*spec*/,
                                      const gpu::Tensor& /*rgb*/) override {
    return refused();
  }
  std::expected<void, Refusal> resample(const gpu::Tensor& /*in*/, kernels::Filter /*filter*/,
                                        const gpu::Tensor& /*out*/) override {
    return refused();
  }
  std::expected<void, Refusal> rife_crop(const gpu::Tensor& /*padded*/, const gpu::Tensor& /*out*/) override {
    return refused();
  }
  std::expected<void, Refusal> warp_blend(const gpu::Tensor& /*a*/, const gpu::Tensor& /*b*/,
                                          const gpu::Motion& /*motion*/, float /*t*/,
                                          const gpu::Tensor& /*out*/) override {
    return refused();
  }
  std::expected<void, Refusal> write_ring(const gpu::Tensor& /*rgb*/, const RingSlotDesc& /*slot*/,
                                          const gpu::RingWrite& /*sync*/) override {
    return refused();
  }
  std::expected<void, Refusal> motion_doubt(const gpu::Motion& /*motion*/,
                                            std::span<const std::uint8_t> /*weights*/, int /*cols*/,
                                            int /*rows*/, float /*fade*/, const gpu::Motion& out) override {
    written.push_back(out.flow.handle);
    return {};
  }
};

// Le mouvement de RIFE à une hauteur de plans : flot 1 × 4 et masque 1 × 1 FP16.
gpu::Motion motion_of(std::int64_t height) {
  const std::int64_t width = height * 16 / 9;
  return {.flow = {.shape = gpu::Shape{.dims = {1, 4, height, width}}, .handle = 1000},
          .mask = {.shape = gpu::Shape{.dims = {1, 1, height, width}}, .handle = 2000},
          .area = {.x = 0, .y = 0, .width = static_cast<int>(width), .height = static_cast<int>(height)}};
}

// Une paire qui bouge (la mesure des blocs laisse tourner RIFE), avec sa carte.
PairFacts moving(const kernels::SceneMotion& scene, bool triage) {
  return PairFacts{.factor = 4,
                   .same_epoch = true,
                   .gap_us = 41'708,
                   .scores = gpu::PairScores{.scene_score = 0.02, .worst_block = 2000},
                   .scene = scene,
                   .flow_triage = triage};
}

}  // namespace

TEST_CASE("tri des paires : sans carte rien ne change, avec elle tenue, fondu et fondu par endroits",
          "[app][rife][triage]") {
  // La carte d'un dessin tenu : rien ne bouge, rien ne change.
  kernels::SceneMotion held = calm_scene();
  held.drawing_held = true;
  held.background_held = true;

  // Sans le réglage, ou sans carte : la décision d'aujourd'hui, et aucun fondu.
  RifeTriage off;
  for (PairFacts facts : {moving(held, false), PairFacts{moving(held, true)}}) {
    if (facts.flow_triage) facts.scene.reset();
    CHECK(rife_runs(facts));
    CHECK_FALSE(pair_held(facts));
    CHECK(off.sort(facts, rife_runs(facts)) == PairSort::Today);
  }
  CHECK(off.counts().read == 0);

  // Dessin tenu par le flux (le grain trompait la mesure des blocs) : RIFE ne tourne pas, la paire est
  // comptée.
  RifeTriage triage;
  const PairFacts by_flow = moving(held, true);
  CHECK(pair_held(by_flow));
  CHECK_FALSE(rife_runs(by_flow));
  CHECK(triage.sort(by_flow, rife_runs(by_flow)) == PairSort::Held);
  CHECK(triage.counts().held == 1);
  // Un effet qui change sur le dessin tenu : plus tenu.
  kernels::SceneMotion rain = held;
  for (std::size_t i = 0; i < 4; ++i) rain.tiles[i].changed = true;
  CHECK_FALSE(flow_held(rain));
  CHECK(rife_runs(moving(rain, true)));

  // Scène calme : RIFE tel quel.
  const kernels::SceneMotion calm = calm_scene();
  CHECK(triage.sort(moving(calm, true), true) == PairSort::Today);
  CHECK(triage.counts().read == 1);

  // Presque rien qui corresponde : toute la paire fondue.
  kernels::SceneMotion lost = calm_scene();
  lost.unmatched = 0.7;
  CHECK(triage.sort(moving(lost, true), true) == PairSort::Fade);

  // Un effet dans un coin : fondu par endroits, le bord adouci, le coin opposé intact.
  triage.reset();
  kernels::SceneMotion smoke = calm_scene();
  smoke.tiles[0] = {.kind = kernels::SceneClass::Effect, .confidence = 0.2F, .changed = true};
  CHECK(triage.sort(moving(smoke, true), true) == PairSort::Masked);
  const std::vector<float> doubt = tile_doubt(smoke);
  CHECK(doubt[0] == 1.0F);
  CHECK(doubt[47] == 0.0F);
  const DoubtGrid grid = doubt_grid(doubt, smoke.cols, smoke.rows);
  REQUIRE(grid.cols == 8);
  REQUIRE(grid.rows == 6);
  CHECK(grid.weights[0] == kernels::kDoubtFull);  // l'effet et la tuile qui le borde : fondu entier
  CHECK(grid.weights[2] > 0);                     // deux tuiles plus loin : le bord adouci
  CHECK(grid.weights[2] < kernels::kDoubtFull);
  CHECK(grid.weights[47] == 0);
  // La paire suivante, sans l'effet, garde la moitié du doute : la zone ne clignote pas.
  CHECK(triage.sort(moving(calm, true), true) == PairSort::Masked);
  // Une coupe l'oublie.
  PairFacts cut = moving(calm, true);
  cut.scores->scene_score = 0.5;
  CHECK(triage.sort(cut, rife_runs(cut)) == PairSort::Today);
  CHECK(triage.sort(moving(calm, true), true) == PairSort::Today);
  CHECK(triage.counts().masked == 2);
  CHECK(triage.counts().faded == 1);

  // Le doute gardé passe une tenue de dessin en quatre (trois paires sans RIFE), pas une plus longue.
  triage.reset();
  const PairFacts still = moving(held, true);
  for (const int pairs : {1, 3, 4}) {
    REQUIRE(triage.sort(moving(smoke, true), true) == PairSort::Masked);
    for (int i = 0; i < pairs; ++i) CHECK(triage.sort(still, false) == PairSort::Held);
    CHECK(triage.sort(moving(calm, true), true) ==
          (pairs <= triage::kKeepAcross ? PairSort::Masked : PairSort::Today));
  }

  // Une grande carte (1080p : 60 × 34 tuiles) se regroupe sous kDoubtMaxCells en gardant le plus fort doute.
  std::vector<float> wide(60 * 34, 0.0F);
  wide[59] = 1.0F;
  const DoubtGrid pooled = doubt_grid(wide, 60, 34);
  CHECK(pooled.cols <= kernels::kDoubtMaxCells);
  CHECK(pooled.rows <= kernels::kDoubtMaxCells);
  CHECK(pooled.weights[static_cast<std::size_t>(pooled.cols - 1)] > 0);
  CHECK(std::ranges::count_if(pooled.weights, [](std::uint8_t w) { return w > 0; }) < 8);

  // GMFSS seul : le dessin tenu copié comme pour RIFE ; l'effet dans un coin laisse la fusion ; la paire
  // impossible fondue par la caméra identité ; le panoramique pur par la caméra de la carte.
  RifeTriage gmfss;
  gmfss.for_gmfss(true);
  CHECK(gmfss.sort(by_flow, rife_runs(by_flow)) == PairSort::Held);
  CHECK(gmfss.sort(moving(smoke, true), true) == PairSort::Today);
  CHECK(gmfss.sort(moving(lost, true), true) == PairSort::Fade);
  CHECK(gmfss.camera() == kernels::kCameraStill);
  kernels::SceneMotion pan = calm_scene();
  for (kernels::SceneTile& t : pan.tiles) t = {.kind = kernels::SceneClass::Background, .confidence = 0.9F};
  pan.drawing_held = true;
  pan.camera = {12.0, 1.0, 0.0, -3.0, 0.0, 1.0};
  pan.residual = 0.4;
  pan.unmatched = 0.05;
  CHECK(flow_pan(pan));
  gmfss.reset();
  CHECK(gmfss.sort(moving(pan, true), true) == PairSort::Pan);
  CHECK(gmfss.camera()[0] == 12.0F);
  CHECK(gmfss.camera()[3] == -3.0F);
  CHECK(gmfss.counts().panned == 1);
  CHECK(gmfss.counts().faded == 1);
  // Un dessin qui bouge, un décor qui s'écarte de la caméra, ou un effet : pas un panoramique pur.
  kernels::SceneMotion acting = pan;
  acting.drawing_held = false;
  CHECK_FALSE(flow_pan(acting));
  kernels::SceneMotion loose = pan;
  loose.residual = 3.0;
  CHECK_FALSE(flow_pan(loose));
  kernels::SceneMotion sparks = pan;
  sparks.tiles[5] = {.kind = kernels::SceneClass::Effect, .confidence = 0.3F};
  CHECK_FALSE(flow_pan(sparks));
  // Une seule tuile de personnage qui marche pendant le panoramique (sous la part de drawing_held) : la
  // fusion, pas la caméra, qui le ferait saccader.
  kernels::SceneMotion walker = pan;
  walker.tiles[20] = {.kind = kernels::SceneClass::Drawing, .confidence = 0.6F, .dx = 15.0F, .changed = true};
  CHECK(walker.drawing_held);
  CHECK_FALSE(flow_pan(walker));
  CHECK(gmfss.sort(moving(walker, true), true) == PairSort::Today);
  CHECK(gmfss.counts().panned == 1);
  // RIFE ne prend jamais le panoramique : la même carte y reste à RIFE tel quel.
  CHECK(triage.sort(moving(pan, true), true) == PairSort::Today);
}

TEST_CASE("tri des paires : un jeu de tampons par forme du mouvement, sans réallocation à chaque instant",
          "[app][rife][triage]") {
  DoubtBackend backend;
  RifeTriage triage;
  // Sans fondu, le mouvement passe tel quel et rien n'est alloué.
  const gpu::Motion middle = motion_of(720);
  CHECK(triage.doubted(backend, middle, 0.5F).flow.handle == middle.flow.handle);
  CHECK(backend.allocated == 0);
  kernels::SceneMotion smoke = calm_scene();
  smoke.tiles[0] = {.kind = kernels::SceneClass::Effect, .confidence = 0.2F, .changed = true};
  REQUIRE(triage.sort(moving(smoke, true), true) == PairSort::Masked);
  // Une paire forte avec l'échelle par paire : milieu à pleine taille, autres instants à demi-taille.
  const gpu::Motion half = motion_of(360);
  for (const gpu::Motion* motion : {&half, &middle, &half, &middle}) {
    const gpu::Motion out = triage.doubted(backend, *motion, 0.5F);
    CHECK(out.flow.shape == motion->flow.shape);
    CHECK(out.mask.shape == motion->mask.shape);
    CHECK(out.flow.handle != motion->flow.handle);  // l'entrée reste intacte pour les autres instants
  }
  CHECK(backend.allocated == 4);  // flot et masque, deux formes
  REQUIRE(backend.written.size() == 4);
  CHECK(backend.written[0] == backend.written[2]);
  CHECK(backend.written[1] == backend.written[3]);
  CHECK(backend.written[0] != backend.written[1]);
  CHECK(triage.counts().refused == 0);
}

}  // namespace kirari::app::play
