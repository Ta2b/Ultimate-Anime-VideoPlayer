// La carte de la scène dans l'étage (app/play/scene_map.h) contre un backend doublé : le flux et l'affinage
// finissent quand le test le dit, l'affinage est celui de la référence (ref::scene_camera_refine) sur les
// lumas de la paire. Dans un plan, l'affinage part derrière le flux depuis la caméra de la paire d'avant et
// finit avec les tuiles ; un affinage pas fini quand l'étage décide de sa paire est relu ensuite
// (last_refined) ; un départ dont la vérification sur l'image passe de loin celle du plan repart depuis la
// caméra de NVOF, un départ juste reste même quand NVOF biaise la caméra de plusieurs pixels.
// Un seul cas, comme le veut la consigne des tests groupés.
#include "app/play/scene_map.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/ref/scene_camera.h"

namespace kirari::app::play {
namespace {

using Catch::Matchers::WithinAbs;
using kernels::Plane;
using kernels::SceneCameraRefined;
using kernels::SceneCameraRequest;
using kernels::SceneFlowTiles;
using kernels::SceneMotion;
using kernels::SceneRoundtripTiles;
namespace ref = kernels::ref;

constexpr int kWidth = 256;  // image du flux : 16 × 9 tuiles
constexpr int kHeight = 144;
constexpr int kDivisor = 2;
constexpr double kNvofBias = 0.6;  // NVOF voit la translation 0,6 pixel du flux trop loin

double scene(double x, double y) {
  return 128.0 + 40.0 * std::sin(0.21 * x + 0.05 * y) + 30.0 * std::sin(0.13 * y - 0.07 * x) +
         20.0 * std::sin(0.31 * x + 0.27 * y);
}

std::uint8_t code(double v) {
  return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 255.0)));
}

// Une paire en translation (tx, ty) pixels du flux, et les tuiles que NVOF en tirerait (biaisées de bias).
struct Flow {
  Plane<std::uint8_t> a{kWidth, kHeight};
  Plane<std::uint8_t> b{kWidth, kHeight};
  SceneFlowTiles tiles;
};

Flow translated(double tx, double ty, double bias = kNvofBias) {
  Flow out;
  for (int j = 0; j < kHeight; ++j) {
    for (int i = 0; i < kWidth; ++i) {
      out.a.at(i, j) = code(scene(i + 0.5, j + 0.5));
      out.b.at(i, j) = code(scene(i + 0.5 - tx, j + 0.5 - ty));
    }
  }
  Plane<ref::FlowVector> flow(kernels::scene_flow_cells(kWidth), kernels::scene_flow_cells(kHeight));
  for (ref::FlowVector& v : flow.data) {
    v = {.x = static_cast<std::int16_t>(std::lround((tx + bias) * 32.0)),
         .y = static_cast<std::int16_t>(std::lround(ty * 32.0))};
  }
  const Plane<std::uint8_t> cost(flow.width, flow.height, 10);
  auto tiles = ref::scene_tile_sums(out.a, out.b, flow, cost, kDivisor);
  REQUIRE(tiles);
  out.tiles = std::move(*tiles);
  return out;
}

// Le backend : chaque paire lancée prend la prochaine de queue ; rien ne finit sans done_flow ou
// done_refine.
class Ops : public gpu::KernelOps {
 public:
  std::vector<Flow> queue;
  struct Asked {
    std::uint64_t flow = 0;
    SceneCameraRequest request;
  };
  std::vector<Asked> asked;  // les affinages demandés, dans l'ordre
  // Affinage rendu faux de 4 pixels (un départ qui a glissé ailleurs) : la vérification de chaque tuile
  // voit ces 4 pixels.
  std::map<std::uint64_t, bool> off;
  std::map<std::uint64_t, bool> photo_edges;
  std::map<std::uint64_t, std::optional<SceneRoundtripTiles>> roundtrips;
  bool fail_flow = false;

  void done_flow(std::uint64_t ticket) { flows_.at(ticket).done = true; }
  void done_refine(std::uint64_t ticket) { refines_.at(ticket).done = true; }

  std::expected<void, Refusal> ingest(const DecodedFrame&, const gpu::IngestSpec&,
                                      const gpu::Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> resample(const gpu::Tensor&, kernels::Filter, const gpu::Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> rife_crop(const gpu::Tensor&, const gpu::Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> warp_blend(const gpu::Tensor&, const gpu::Tensor&, const gpu::Motion&, float,
                                          const gpu::Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> write_ring(const gpu::Tensor&, const RingSlotDesc&,
                                          const gpu::RingWrite&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }

  std::expected<std::uint64_t, Refusal> scene_flow_start(const gpu::Tensor& /*a*/, const gpu::Tensor& /*b*/,
                                                         bool /*restart*/) override {
    REQUIRE(!queue.empty());
    flows_[++flow_tickets_] = {.flow = queue.front(), .done = false};
    queue.erase(queue.begin());
    return flow_tickets_;
  }
  std::expected<std::optional<SceneFlowTiles>, Refusal> scene_flow_take(std::uint64_t ticket) override {
    if (fail_flow) return refuse(RefusalCode::DeviceLost, "retour du flot factice refusé");
    const Started& s = flows_.at(ticket);
    if (!s.done) return std::optional<SceneFlowTiles>{};
    return std::optional(s.flow.tiles);
  }
  std::expected<std::optional<SceneRoundtripTiles>, Refusal> scene_roundtrip_take(
      std::uint64_t ticket) override {
    auto it = roundtrips.find(ticket);
    if (it == roundtrips.end()) return refuse(RefusalCode::FormatUnsupported, "sans retour");
    auto out = std::move(it->second);
    if (out) roundtrips.erase(it);
    return out;
  }
  std::expected<std::uint64_t, Refusal> scene_camera_start(std::uint64_t flow_ticket,
                                                           const SceneCameraRequest& request) override {
    asked.push_back({.flow = flow_ticket, .request = request});
    refines_[++refine_tickets_] = {.flow = flow_ticket, .request = request, .done = false};
    return refine_tickets_;
  }
  std::expected<std::optional<SceneCameraRefined>, Refusal> scene_camera_take(std::uint64_t ticket) override {
    const Refining& r = refines_.at(ticket);
    if (!r.done) return std::optional<SceneCameraRefined>{};
    const Flow& f = flows_.at(r.flow).flow;
    auto refined = ref::scene_camera_refine(f.a, f.b, r.request);
    REQUIRE(refined);
    if (off.contains(r.flow)) {
      refined->model[0] += 4.0;
      for (kernels::SceneTileCheck& c : refined->tiles) c.sx -= 4.0F;
    }
    if (photo_edges.contains(r.flow)) {
      refined->tiles[0].pixels = 64.0F;
      refined->tiles[0].sharp = 0.025F;
      refined->tiles[1].pixels = 63.0F;
      refined->tiles[1].sharp = 0.5F;
      refined->tiles[2].pixels = 64.0F;
      refined->tiles[2].sharp = 0.02F;
    }
    return std::optional(std::move(*refined));
  }

 private:
  struct Started {
    Flow flow;
    bool done = false;
  };
  struct Refining {
    std::uint64_t flow = 0;
    SceneCameraRequest request;
    bool done = false;
  };
  std::map<std::uint64_t, Started> flows_;
  std::map<std::uint64_t, Refining> refines_;
  std::uint64_t flow_tickets_ = 0;
  std::uint64_t refine_tickets_ = 0;
};

SourceId id(std::uint32_t v) {
  return SourceId{.value = v};
}

// La translation de la caméra de la carte, pixels de travail.
double camera_x(const SceneMotion& m) {
  return m.camera[0];
}

}  // namespace

TEST_CASE("Carte de scène : une ancienne époque et un flot refusé ne laissent aucun état réutilisable",
          "[app][scene]") {
  Ops ops;
  ops.queue = {translated(1, 0), translated(1, 0), translated(1, 0)};
  SceneMap map(true);
  const gpu::Tensor tensor;
  map.start(ops, tensor, tensor, SourceId{1}, SourceId{2}, 1);
  ops.done_flow(1);
  REQUIRE(map.for_pair(ops, SourceId{1}, SourceId{2}, 1));
  map.start(ops, tensor, tensor, SourceId{2}, SourceId{3}, 2);
  CHECK_FALSE(map.latest());
  CHECK_FALSE(map.for_pair(ops, SourceId{1}, SourceId{2}, 1));
  ops.fail_flow = true;
  map.start(ops, tensor, tensor, SourceId{3}, SourceId{4}, 2);
  CHECK_FALSE(map.on());
  CHECK_FALSE(map.latest());
  CHECK(ops.queue.size() == 1);  // la panne relue au lancement a déjà arrêté la carte
}

TEST_CASE("carte de la scène : l'affinage derrière le flux, et relu après sa paire", "[scene_map]") {
  Ops ops;
  SceneMap map(true);
  const gpu::Tensor t;
  // Un panoramique : 3 pixels du flux par image, puis 3,2 ; sur la troisième paire, NVOF voit la caméra
  // 3 pixels trop loin.
  ops.queue.push_back(translated(3.0, 0.5));
  ops.queue.push_back(translated(3.0, 0.5));
  ops.queue.push_back(translated(3.2, 0.5, 3.0));
  ops.queue.push_back(translated(3.2, 0.5));
  ops.queue.push_back(translated(3.2, 0.5));

  // La première paire du plan : pas de caméra d'où partir, l'affinage attend les tuiles relues.
  map.start(ops, t, t, id(1), id(2), 0);
  CHECK(ops.asked.empty());
  ops.done_flow(1);
  map.advance(ops);
  REQUIRE(ops.asked.size() == 1);
  CHECK_THAT(ops.asked[0].request.model[0], WithinAbs(3.0 + kNvofBias, 0.05));  // celle de NVOF
  ops.done_refine(1);
  const auto first = map.for_pair(ops, id(1), id(2));
  REQUIRE(first);
  CHECK(first->camera_refined);
  CHECK_THAT(camera_x(*first), WithinAbs(3.0 * kDivisor, 0.1));  // le biais de NVOF retiré
  REQUIRE(map.last_refined());
  CHECK(map.last_refined()->right == id(2));

  // La suivante : l'affinage part au lancement, depuis la caméra affinée, et finit avec les tuiles.
  map.start(ops, t, t, id(2), id(3), 0);
  REQUIRE(ops.asked.size() == 2);
  CHECK(ops.asked[1].flow == 2);
  CHECK_THAT(ops.asked[1].request.model[0], WithinAbs(3.0, 0.05));
  ops.done_flow(2);
  ops.done_refine(2);
  const auto second = map.for_pair(ops, id(2), id(3));
  REQUIRE(second);
  CHECK(second->camera_refined);
  CHECK_THAT(camera_x(*second), WithinAbs(3.0 * kDivisor, 0.1));
  CHECK(ops.asked.size() == 2);  // aucun second affinage

  // La caméra accélère (3,2) et NVOF la voit 3 pixels trop loin : le départ de 3 y converge, sa
  // vérification est celle du plan, il reste.
  map.start(ops, t, t, id(3), id(4), 0);
  REQUIRE(ops.asked.size() == 3);
  ops.done_flow(3);
  ops.done_refine(3);
  const auto third = map.for_pair(ops, id(3), id(4));
  REQUIRE(third);
  CHECK(third->camera_refined);
  CHECK_THAT(camera_x(*third), WithinAbs(3.2 * kDivisor, 0.1));
  CHECK(ops.asked.size() == 3);

  // L'étage décide avant la fin de l'affinage : la carte part avec la caméra de NVOF, l'affinage est relu
  // ensuite pour last_refined().
  map.start(ops, t, t, id(4), id(5), 0);
  REQUIRE(ops.asked.size() == 4);
  ops.done_flow(4);
  const auto late = map.for_pair(ops, id(4), id(5));
  REQUIRE(late);
  CHECK_FALSE(late->camera_refined);
  CHECK_THAT(camera_x(*late), WithinAbs((3.2 + kNvofBias) * kDivisor, 0.1));
  CHECK(map.last_refined()->right == id(4));
  ops.done_refine(4);
  map.advance(ops);
  REQUIRE(map.last_refined());
  CHECK(map.last_refined()->left == id(4));
  CHECK(map.last_refined()->right == id(5));
  CHECK_THAT(map.last_refined()->camera[0], WithinAbs(3.2 * kDivisor, 0.1));
  CHECK(map.last_refined()->residual < 0.5);
  CHECK(map.last_refined()->residual_tiles > 0);

  // Un départ qui glisse ailleurs (4 pixels, que la vérification voit) : écarté, l'affinage repart depuis la
  // caméra de NVOF.
  ops.off[5] = true;
  map.start(ops, t, t, id(5), id(6), 0);
  REQUIRE(ops.asked.size() == 5);
  ops.done_flow(5);
  ops.done_refine(5);
  map.advance(ops);
  REQUIRE(ops.asked.size() == 6);
  CHECK(ops.asked[5].flow == 5);
  CHECK_THAT(ops.asked[5].request.model[0], WithinAbs(3.2 + kNvofBias, 0.05));
  ops.off.clear();
  ops.done_refine(6);
  const auto again = map.for_pair(ops, id(5), id(6));
  REQUIRE(again);
  CHECK(again->camera_refined);
  CHECK_THAT(camera_x(*again), WithinAbs(3.2 * kDivisor, 0.1));

  // Une coupe sur (5, 6) : sa carte n'a plus que des tuiles incertaines, sans caméra, et la paire suivante
  // repart sans caméra d'où partir.
  map.cut(id(5), id(6));
  const auto cut = map.for_pair(ops, id(5), id(6));
  REQUIRE(cut);
  CHECK(cut->cut_likely);
  CHECK_FALSE(cut->camera_fit);
  CHECK(std::ranges::all_of(cut->tiles, [](const kernels::SceneTile& tile) {
    return tile.kind == kernels::SceneClass::Uncertain && tile.changed;
  }));
  ops.queue.push_back(translated(-2.0, 0.0));
  map.start(ops, t, t, id(6), id(7), 0);
  CHECK(ops.asked.size() == 6);
}

TEST_CASE("carte de la scène : le retour ne sert qu'à sa paire et peut manquer", "[scene_map]") {
  Ops ops;
  SceneMap map(true, true);
  const gpu::Tensor t;
  const SceneRoundtripTiles first{.ticket = 1,
                                  .flow_width = kWidth,
                                  .flow_height = kHeight,
                                  .divisor = kDivisor,
                                  .tiles = std::vector<kernels::SceneRoundtripTile>(16 * 9)};
  ops.queue.push_back(translated(3.0, 0.5));
  map.start(ops, t, t, id(1), id(2), 7);
  ops.done_flow(1);
  const auto no_return = map.for_pair(ops, id(1), id(2), 7);
  REQUIRE(no_return);
  CHECK_FALSE(no_return->roundtrip);
  CHECK(no_return->roundtrip_photo_edge_a.empty());
  ops.roundtrips[1] = std::nullopt;
  CHECK_FALSE(map.for_pair(ops, id(1), id(2), 7)->roundtrip);
  auto bad_shape = first;
  bad_shape.tiles.pop_back();
  ops.roundtrips[1] = bad_shape;
  CHECK_FALSE(map.for_pair(ops, id(1), id(2), 7)->roundtrip);
  ops.roundtrips[1] = first;
  const auto old_pair = map.for_pair(ops, id(1), id(2), 7);
  REQUIRE(old_pair);
  REQUIRE(old_pair->roundtrip);
  CHECK(old_pair->roundtrip->ticket == 1);
  CHECK_FALSE(old_pair->camera_refined);
  CHECK(old_pair->roundtrip_photo_edge_a.empty());

  // Une nouvelle époque peut reprendre les mêmes identifiants source, avec un nouveau ticket.
  CHECK(map.started(id(1), id(2), 7));
  CHECK_FALSE(map.started(id(1), id(2), 8));
  CHECK_FALSE(map.for_pair(ops, id(1), id(2), 8));
  ops.queue.push_back(translated(3.0, 0.5));
  map.start(ops, t, t, id(1), id(2), 8);
  ops.done_flow(2);
  map.advance(ops);
  ops.photo_edges[2] = true;
  ops.done_refine(2);
  ops.roundtrips[2] = first;  // le backend rendrait le résultat d'une autre paire
  const auto stale = map.for_pair(ops, id(1), id(2), 8);
  REQUIRE(stale);
  CHECK_FALSE(stale->roundtrip);
  CHECK(stale->camera_refined);
  CHECK(stale->roundtrip_photo_edge_a.empty());
  auto second = first;
  second.ticket = 2;
  ops.roundtrips[2] = second;
  const auto new_pair = map.for_pair(ops, id(1), id(2), 8);
  REQUIRE(new_pair);
  REQUIRE(new_pair->roundtrip);
  CHECK(new_pair->roundtrip->ticket == 2);
  REQUIRE(new_pair->roundtrip_photo_edge_a.size() == 16 * 9);
  CHECK(new_pair->roundtrip_photo_edge_a[0] == 1);
  CHECK(new_pair->roundtrip_photo_edge_a[1] == 0);
  CHECK(new_pair->roundtrip_photo_edge_a[2] == 0);
  map.cut(id(1), id(2), 8);
  const auto cut = map.for_pair(ops, id(1), id(2), 8);
  REQUIRE(cut);
  CHECK_FALSE(cut->roundtrip);
  CHECK(cut->roundtrip_photo_edge_a.empty());
}

}  // namespace kirari::app::play
