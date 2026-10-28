// Propriété des passes sur des graphes aléatoires à graine fixe : chaînes de Conv,
// opérations par canal, activations, résiduels, Identity, pixel (un)shuffle écrits
// en Reshape/Transpose/Reshape, sorties intermédiaires et lecteurs multiples.
// Après optimize, chaque sortie de l'interpréteur reste à 1e-5 près (relatif à
// l'amplitude, poids bornés) et le plan mémoire ne superpose aucun tampon vivant.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/memory_plan.h"
#include "gpu/ir/passes.h"
#include "test_util.h"

using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

constexpr std::uint32_t kSeed = 20261019;
constexpr int kGraphs = 250;

struct Activation4 {
  std::string name;
  std::int64_t c = 0;
  std::int64_t h = 0;
  std::int64_t w = 0;
};

struct RandomModel {
  ModelBuilder model;
  Dims input;
};

class RandomNet {
 public:
  explicit RandomNet(std::uint32_t seed) : rng_(seed) {}

  RandomModel make() {
    const Dims input{1, pick(1, 4), 2 * pick(1, 3), 2 * pick(1, 3)};
    m_.input("x", input);
    pool_.push_back({.name = "x", .c = input[1], .h = input[2], .w = input[3]});
    const std::int64_t steps = pick(4, 14);
    for (std::int64_t s = 0; s < steps; ++s) {
      switch (pick(0, 9)) {
        case 0:
        case 1:
        case 2:
          conv();
          break;
        case 3:
        case 4:
          channel_op();
          break;
        case 5:
          activation();
          break;
        case 6:
          residual();
          break;
        case 7:
          identity();
          break;
        default:
          rearrange();
          break;
      }
      // Une sortie intermédiaire de temps en temps : elle bloque les fusions qui la traversent.
      if (pool_.size() > 1 && pick(0, 7) == 0) mark_output(pool_.back().name);
    }
    if (pool_.size() == 1) conv();
    mark_output(pool_.back().name);
    return {.model = std::move(m_), .input = input};
  }

 private:
  std::int64_t pick(std::int64_t lo, std::int64_t hi) {
    return std::uniform_int_distribution<std::int64_t>(lo, hi)(rng_);
  }
  float real(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng_); }
  std::vector<float> reals(std::int64_t count, float lo, float hi) {
    std::vector<float> out(static_cast<std::size_t>(count));
    for (float& v : out) v = real(lo, hi);
    return out;
  }
  std::string fresh(std::string_view base) { return std::format("{}{}", base, next_++); }

  // Le dernier tenseur le plus souvent (chaînes), un plus ancien sinon (lecteurs multiples).
  Activation4 source() {
    if (pick(0, 2) != 0) return pool_.back();
    return pool_[static_cast<std::size_t>(pick(0, static_cast<std::int64_t>(pool_.size()) - 1))];
  }

  void push(const std::string& name, const Activation4& like) {
    pool_.push_back({.name = name, .c = like.c, .h = like.h, .w = like.w});
  }

  void mark_output(const std::string& name) {
    if (name == "x" || std::ranges::find(outputs_, name) != outputs_.end()) return;
    outputs_.push_back(name);
    m_.output(name);
  }

  void conv() {
    const Activation4 in = source();
    const std::int64_t out_c = pick(1, 8);
    const std::int64_t k = pick(0, 1) == 1 ? 3 : 1;
    const std::int64_t p = k / 2;
    const float bound = 1.0F / std::sqrt(static_cast<float>(in.c * k * k));
    const std::string w = fresh("w");
    const std::string y = fresh("conv");
    m_.weight(w, {out_c, in.c, k, k}, reals(out_c * in.c * k * k, -bound, bound));
    const ProtoWriter pads = attr_ints("pads", {p, p, p, p});
    if (pick(0, 1) == 1) {
      const std::string b = fresh("b");
      m_.weight(b, {out_c}, reals(out_c, -0.2F, 0.2F));
      m_.node("Conv", {in.name, w, b}, {y}, {pads});
    } else {
      m_.node("Conv", {in.name, w}, {y}, {pads});
    }
    pool_.push_back({.name = y, .c = out_c, .h = in.h, .w = in.w});
  }

  // Add, Sub, Mul ou Div par une constante par canal, scalaire, ou pleine (non fusionnable).
  void channel_op() {
    const Activation4 in = source();
    static constexpr std::array<std::string_view, 4> kOps{"Add", "Sub", "Mul", "Div"};
    const auto op = static_cast<std::size_t>(pick(0, 3));
    const bool scaling = op >= 2;
    Dims dims;
    switch (pick(0, 3)) {
      case 0:
        dims = {1, in.c, 1, 1};
        break;
      case 1:
        dims = {in.c, 1, 1};
        break;
      case 2:
        dims = {1};
        break;
      default:
        dims = {1, in.c, in.h, in.w};
        break;
    }
    const std::string k = fresh("k");
    const std::string y = fresh("e");
    m_.weight(
        k, dims,
        reals(static_cast<std::int64_t>(count_of(dims)), scaling ? 0.5F : -0.5F, scaling ? 1.5F : 0.5F));
    if ((op == 0 || op == 2) && pick(0, 1) == 1) {
      m_.node(kOps[op], {k, in.name}, {y});  // constante à gauche : Add et Mul commutent
    } else {
      m_.node(kOps[op], {in.name, k}, {y});
    }
    push(y, in);
  }

  void activation() {
    const Activation4 in = source();
    const std::string y = fresh("a");
    switch (pick(0, 5)) {
      case 0:
        m_.node("Relu", {in.name}, {y});
        break;
      case 1:
        m_.node("LeakyRelu", {in.name}, {y}, {attr_float("alpha", real(0.01F, 0.3F))});
        break;
      case 2: {
        const std::string slope = fresh("slope");
        m_.weight(slope, {in.c, 1, 1}, reals(in.c, 0.0F, 0.4F));
        m_.node("PRelu", {in.name, slope}, {y});
        break;
      }
      case 3:
        m_.node("Sigmoid", {in.name}, {y});
        break;
      case 4:
        m_.node("Tanh", {in.name}, {y});
        break;
      default: {
        const std::string lo = fresh("lo");
        const std::string hi = fresh("hi");
        m_.weight(lo, {}, {real(-1.0F, -0.1F)}).weight(hi, {}, {real(0.1F, 1.0F)});
        m_.node("Clip", {in.name, lo, hi}, {y});
        break;
      }
    }
    push(y, in);
  }

  // Somme de deux activations de même forme, éventuellement la même deux fois.
  void residual() {
    const Activation4 a = source();
    std::vector<Activation4> same;
    for (const Activation4& t : pool_) {
      if (t.c == a.c && t.h == a.h && t.w == a.w) same.push_back(t);
    }
    const Activation4& b =
        same[static_cast<std::size_t>(pick(0, static_cast<std::int64_t>(same.size()) - 1))];
    const std::string y = fresh("r");
    if (pick(0, 1) == 1) {
      m_.node("Add", {a.name, b.name}, {y});
    } else {
      m_.node("Add", {b.name, a.name}, {y});
    }
    push(y, a);
  }

  void identity() {
    const Activation4 in = source();
    const std::string y = fresh("id");
    m_.node("Identity", {in.name}, {y});
    push(y, in);
  }

  // pixel_unshuffle / pixel_shuffle de PyTorch (CRD) ou SpaceToDepth / DepthToSpace (DCR).
  void rearrange() {
    const Activation4 in = source();
    const bool crd = pick(0, 1) == 1;
    const bool can_unshuffle = in.h % 2 == 0 && in.w % 2 == 0 && in.c * 4 <= 16;
    const bool can_shuffle = in.c % 4 == 0 && in.h * 2 <= 12 && in.w * 2 <= 12;
    if (!can_unshuffle && !can_shuffle) {
      conv();
      return;
    }
    const bool unshuffle = can_unshuffle && (!can_shuffle || pick(0, 1) == 1);
    const std::string s6 = fresh("s6");
    const std::string s4 = fresh("s4");
    const std::string t6 = fresh("t6");
    const std::string tt = fresh("tt");
    const std::string y = fresh("px");
    Activation4 out = in;
    Dims perm;
    if (unshuffle) {
      m_.ints(s6, {1, in.c, in.h / 2, 2, in.w / 2, 2});
      perm = crd ? Dims{0, 1, 3, 5, 2, 4} : Dims{0, 3, 5, 1, 2, 4};
      out = {.name = y, .c = in.c * 4, .h = in.h / 2, .w = in.w / 2};
    } else {
      m_.ints(s6, crd ? Dims{1, in.c / 4, 2, 2, in.h, in.w} : Dims{1, 2, 2, in.c / 4, in.h, in.w});
      perm = crd ? Dims{0, 1, 4, 2, 5, 3} : Dims{0, 3, 4, 1, 5, 2};
      out = {.name = y, .c = in.c / 4, .h = in.h * 2, .w = in.w * 2};
    }
    m_.ints(s4, {1, out.c, out.h, out.w});
    m_.node("Reshape", {in.name, s6}, {t6})
        .node("Transpose", {t6}, {tt}, {attr_ints("perm", perm)})
        .node("Reshape", {tt, s4}, {y});
    pool_.push_back(out);
  }

  std::mt19937_64 rng_;
  ModelBuilder m_;
  std::vector<Activation4> pool_;
  std::vector<std::string> outputs_;
  int next_ = 0;
};

float magnitude(const std::vector<float>& values) {
  float top = 1.0F;
  for (const float v : values) top = std::max(top, std::abs(v));
  return top;
}

void check_plan(const MemoryPlan& plan) {
  for (std::size_t a = 0; a < plan.allocations.size(); ++a) {
    const Allocation& p = plan.allocations[a];
    CHECK(p.offset + p.bytes <= plan.arena_bytes);
    for (std::size_t b = a + 1; b < plan.allocations.size(); ++b) {
      const Allocation& q = plan.allocations[b];
      const bool alive_together = p.first <= q.last && q.first <= p.last;
      const bool apart = p.offset + p.bytes <= q.offset || q.offset + q.bytes <= p.offset;
      CHECK((!alive_together || apart));
    }
  }
}

}  // namespace

TEST_CASE("propriété : graphes aléatoires préservés par les passes", "[gpu_ir][passes][property]") {
  PassStats total;
  std::size_t nodes_before = 0;
  std::size_t nodes_after = 0;
  for (int g = 0; g < kGraphs; ++g) {
    const std::uint32_t seed = kSeed + static_cast<std::uint32_t>(g);
    INFO("graine " << seed);
    RandomNet net(seed);
    const RandomModel generated = net.make();
    const Graph before = build(generated.model);
    const std::vector<NamedValue> inputs{
        {.name = "x", .value = random_floats(generated.input, static_cast<std::size_t>(g))}};
    const std::vector<NamedValue> expected = execute(before, inputs);

    Graph after = before;
    auto stats = optimize(after);
    if (!stats) FAIL("refus " << ajp::name(stats.error().code) << " : " << stats.error().detail);
    total.folded += stats->folded;
    total.identities += stats->identities;
    total.pixel_shuffles += stats->pixel_shuffles;
    total.bias += stats->bias;
    total.residuals += stats->residuals;
    total.activations += stats->activations;
    total.nhwc += stats->nhwc;
    nodes_before += before.nodes.size();
    nodes_after += after.nodes.size();
    // Seule reste une Identity d'une interface vers une sortie : aucun nœud ne peut écrire à sa place.
    for (const Node& node : after.nodes) {
      if (node.op != OpKind::Identity) continue;
      const TensorId in = node.inputs[0];
      CHECK(is_graph_output(after, node.outputs[0]));
      CHECK((is_graph_input(after, in) || is_graph_output(after, in)));
    }

    const std::vector<NamedValue> got = execute(after, inputs);
    REQUIRE(got.size() == expected.size());
    for (std::size_t k = 0; k < got.size(); ++k) {
      CHECK(got[k].name == expected[k].name);
      REQUIRE(got[k].value.dims == expected[k].value.dims);
      CHECK(max_abs_diff(got[k].value.f, expected[k].value.f) <= 1e-5F * magnitude(expected[k].value.f));
    }
    auto plan = plan_memory(after);
    REQUIRE(plan);
    check_plan(*plan);
  }
  // Le générateur atteint chaque passe : la propriété ne se vérifie pas sur des graphes intacts.
  INFO("nœuds " << nodes_before << " -> " << nodes_after);
  CHECK(total.identities > 0);
  CHECK(total.pixel_shuffles > 0);
  CHECK(total.bias > 0);
  CHECK(total.residuals > 0);
  CHECK(total.activations > 0);
  CHECK(total.nhwc > 0);
  CHECK(nodes_after < nodes_before);
}
