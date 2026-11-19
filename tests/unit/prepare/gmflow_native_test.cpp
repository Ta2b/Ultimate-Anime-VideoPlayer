// Le squelette CNN de GMFlow lu pour les noyaux natifs (prepare/gmflow_native.h) : un faux squelette aux
// noms, opérateurs et attributs de la paire exportée (canaux réduits), lu et rangé, ses fautes refusées, le
// plan et ses fautes, puis la vraie paire quand elle est là (KIRARI_RIFE_MODEL_DIR).
#include "prepare/gmflow_native.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <set>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/ref/half.h"
#include "prepare/model_source.h"
#include "rife_fake.h"

#ifndef KIRARI_RIFE_MODEL_DIR
#define KIRARI_RIFE_MODEL_DIR ""
#endif

namespace kirari::prepare {
namespace {

Bytes ints_attribute(std::string_view name, std::initializer_list<std::int64_t> values) {
  Bytes out;
  delimited(out, 1, text(name));
  for (const std::int64_t v : values) integer(out, 8, static_cast<std::uint64_t>(v));
  return out;
}
Bytes float_attribute(std::string_view name, float value) {
  Bytes out;
  delimited(out, 1, text(name));
  varint(out, (2U << 3U) | 5U);
  const auto bits = std::bit_cast<std::uint32_t>(value);
  for (unsigned i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>((bits >> (8U * i)) & 0xffU));
  return out;
}
Bytes with_attributes(Bytes n, const std::vector<Bytes>& attrs) {
  for (const Bytes& a : attrs) delimited(n, 5, a);
  return n;
}
Bytes raw(const std::vector<float>& values) {
  Bytes out(values.size() * 4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(values[i]);
    std::memcpy(out.data() + 4 * i, &bits, 4);
  }
  return out;
}

// Le poids ONNX (o, c, ky, kx) du faux squelette : des valeurs distinctes, exactes en FP32.
float weight(int o, int c, int ky, int kx) {
  return static_cast<float>((o * 7 + c * 3 + ky * 5 + kx) % 17 - 8) * 0.0625F;
}

// Le faux squelette : 3 → c1 (7 × 7), layer1 c1, layer2 c1 → c2 (pas 2), layer3 c2 → c2, conv2 c2 → 128,
// trident 128. Chaque faute casse un maillon que la lecture doit voir.
struct FakeBackbone {
  int c1 = 32;
  int c2 = 64;
  std::int64_t trident_stride = 2;
  std::int64_t layer1_pads = 1;
  std::int64_t group = 1;
  std::int64_t dilation = 1;
  float norm_scale = 1.0F;
  float last_epsilon = 1e-5F;
  bool ds_bias = true;
  bool add_swapped = false;
  bool second_mean_differs = false;
  bool conv1_bias = false;
};

class Builder {
 public:
  explicit Builder(const FakeBackbone& f) : f_(f) {}

  std::string conv(const std::string& name, const std::string& module, const std::string& in, int cin,
                   int cout, std::int64_t k, std::int64_t stride, bool bias, std::int64_t pads = -1) {
    const std::string w = std::format("gmfss.flownet.backbone.{}.weight", module);
    std::vector<float> values;
    for (int o = 0; o < cout; ++o)
      for (int c = 0; c < cin; ++c)
        for (int y = 0; y < k; ++y)
          for (int x = 0; x < k; ++x) values.push_back(weight(o, c, y, x));
    if (!weights_.contains(w)) {
      init(w, {cout, cin, k, k}, values);
      weights_.insert(w);
    }
    std::vector<std::string> inputs{in, w};
    if (bias) {
      const std::string b = std::format("gmfss.flownet.backbone.{}.bias", module);
      init(b, {cout}, std::vector<float>(static_cast<std::size_t>(cout), 0.5F));
      inputs.push_back(b);
    }
    const std::int64_t p = pads >= 0 ? pads : k / 2;
    add(with_attributes(
        node(name, "Conv", inputs, {name + "_output_0"}),
        {ints_attribute("kernel_shape", {k, k}), ints_attribute("strides", {stride, stride}),
         ints_attribute("pads", {p, p, p, p}), ints_attribute("dilations", {f_.dilation, f_.dilation}),
         attribute({.name = "group", .num = f_.group})}));
    return name + "_output_0";
  }

  std::string norm(const std::string& name, const std::string& in, int channels, float epsilon = 1e-5F) {
    const std::string scale = std::format("ones{}", channels), shift = std::format("zeros{}", channels);
    if (!weights_.contains(scale)) {
      init(scale, {channels}, std::vector<float>(static_cast<std::size_t>(channels), f_.norm_scale));
      init(shift, {channels}, std::vector<float>(static_cast<std::size_t>(channels), 0.0F));
      weights_.insert(scale);
    }
    add(with_attributes(node(name, "InstanceNormalization", {in, scale, shift}, {name + "_output_0"}),
                        {float_attribute("epsilon", epsilon)}));
    return name + "_output_0";
  }

  std::string simple(const std::string& name, std::string_view op, std::vector<std::string> in) {
    add(node(name, op, std::move(in), {name + "_output_0"}));
    return name + "_output_0";
  }

  void init(const std::string& name, const std::vector<std::int64_t>& dims,
            const std::vector<float>& values) {
    delimited(graph_, 5, tensor_dims(name, dims, raw(values)));
  }
  void add(const Bytes& n) { delimited(graph_, 1, n); }
  Bytes model() const {
    Bytes out;
    integer(out, 1, 8);
    delimited(out, 7, graph_);
    return out;
  }

 private:
  const FakeBackbone& f_;
  Bytes graph_;
  std::set<std::string> weights_;
};

std::string block(Builder& b, const FakeBackbone& f, int layer, int index, const std::string& x, int cin,
                  int cout, int stride, bool ds) {
  const std::string module = std::format("layer{}.{}", layer, index);
  const std::string p = std::format("/flownet/backbone/layer{}/{}/", layer, module);
  const bool first = layer == 1 && index == 0;
  const std::string a = b.conv(p + "conv1/Conv", module + ".conv1", x, cin, cout, 3, stride,
                               first && f.conv1_bias, layer == 1 ? f.layer1_pads : -1);
  const std::string h1 =
      b.simple(p + "relu/Relu", "Relu", {b.norm(p + "norm1/InstanceNormalization", a, cout)});
  const std::string c = b.conv(p + "conv2/Conv", module + ".conv2", h1, cout, cout, 3, 1, false);
  const bool last = layer == 3 && index == 1;
  const std::string h2 =
      b.simple(p + "relu_1/Relu", "Relu",
               {b.norm(p + "norm2/InstanceNormalization", c, cout, last ? f.last_epsilon : 1e-5F)});
  std::string skip = x;
  if (ds) {
    const std::string d = b.conv(p + "downsample/downsample.0/Conv", module + ".downsample.0", x, cin, cout,
                                 1, stride, f.ds_bias);
    skip = b.norm(p + "downsample/norm3/InstanceNormalization", d, cout);
  }
  const std::string sum =
      b.simple(p + "Add", "Add", f.add_swapped ? std::vector{h2, skip} : std::vector{skip, h2});
  return b.simple(p + "relu_2/Relu", "Relu", {sum});
}

constexpr float kMean[3] = {0.5F, 0.25F, 0.125F};
constexpr float kDeviation[3] = {0.25F, 0.5F, 2.0F};

Bytes fake_backbone(const FakeBackbone& f) {
  Builder b(f);
  b.init("mean", {1, 3, 1, 1}, {kMean[0], kMean[1], kMean[2]});
  b.init("mean_1", {1, 3, 1, 1}, {kMean[0], kMean[1], f.second_mean_differs ? 0.5F : kMean[2]});
  b.init("deviation", {1, 3, 1, 1}, {kDeviation[0], kDeviation[1], kDeviation[2]});
  const std::string d0 =
      b.simple("/flownet/Div", "Div", {b.simple("/flownet/Sub", "Sub", {"half0", "mean"}), "deviation"});
  const std::string d1 = b.simple("/flownet/Div_1", "Div",
                                  {b.simple("/flownet/Sub_1", "Sub", {"half1", "mean_1"}), "deviation"});
  b.add(node("/flownet/Concat", "Concat", {d0, d1}, {"/flownet/Concat_output_0"},
             {{.name = "axis", .num = 0}}));
  const std::string n = "/flownet/backbone/";
  const std::string stem =
      b.conv(n + "conv1/Conv", "conv1", "/flownet/Concat_output_0", 3, f.c1, 7, 2, false);
  std::string x = b.simple(n + "relu1/Relu", "Relu", {b.norm(n + "norm1/InstanceNormalization", stem, f.c1)});
  x = block(b, f, 1, 0, x, f.c1, f.c1, 1, false);
  x = block(b, f, 1, 1, x, f.c1, f.c1, 1, false);
  x = block(b, f, 2, 0, x, f.c1, f.c2, 2, true);
  x = block(b, f, 2, 1, x, f.c2, f.c2, 1, false);
  x = block(b, f, 3, 0, x, f.c2, f.c2, 1, true);
  x = block(b, f, 3, 1, x, f.c2, f.c2, 1, false);
  const std::string e = b.conv(n + "conv2/Conv", "conv2", x, f.c2, 128, 1, 1, true);
  b.conv(n + "trident_conv/Conv", "trident_conv", e, 128, 128, 3, 1, false);
  b.conv(n + "trident_conv/Conv_1", "trident_conv", e, 128, 128, 3, f.trident_stride, false);
  return b.model();
}

float as_half(float v) {
  return kernels::ref::half_to_float(kernels::ref::float_to_half(v));
}

}  // namespace

TEST_CASE("GMFlow natif : le faux squelette se lit, rangé pour les noyaux", "[prepare][gmflow]") {
  const auto backbone = gmflow_backbone(fake_backbone(FakeBackbone{}));
  INFO((backbone ? std::string() : backbone.error().detail));
  REQUIRE(backbone);
  const GmflowBackbone& b = *backbone;
  CHECK(b.convs.size() == 17);  // 18 convolutions, le trident deux fois aux mêmes poids
  CHECK(b.ops.size() == 18 + 6);
  CHECK(b.stats == 15);
  CHECK(b.epsilon == 1e-5F);
  CHECK(b.tensors[static_cast<std::size_t>(b.quarter)].level == 2);
  CHECK(b.tensors[static_cast<std::size_t>(b.eighth)].level == 3);
  // [cout][prise][cin] depuis [cout][cin][ky][kx] : layer1.0.conv1, sortie 5, prise (2, 1), entrée 7.
  const GmflowConv& l1 = b.convs[1];
  CHECK(l1.weights[(5 * 9 + 7) * 32 + 7] == weight(5, 7, 2, 1));
  // La normalisation de l'image dans la 7 × 7 : poids × 1/écart, biais -Σ poids · moyenne, bord à la moyenne.
  const GmflowConv& stem = b.convs[0];
  CHECK(b.pad[1] == as_half(kMean[1]));
  CHECK(stem.weights[(3 * 49 + 10) * 3 + 2] == weight(3, 2, 1, 3) / kDeviation[2]);
  double bias = 0.0;
  for (int c = 0; c < 3; ++c)
    for (int t = 0; t < 49; ++t) bias -= double{weight(4, c, t / 7, t % 7)} / kDeviation[c] * kMean[c];
  CHECK(std::abs(stem.bias[4] - bias) <= 1e-6 * (1.0 + std::abs(bias)));
  // Le premier bloc ajoute la sortie normalisée de la 7 × 7 (Relu comprise) ; layer2.0, son raccourci 1 × 1
  // normalisé, sans Relu ; le trident : mêmes poids, pas 1 puis 2.
  const GmflowOp& end1 = b.ops[3];
  CHECK(end1.kind == GmflowOpKind::kBlockEnd);
  CHECK((end1.in.relu && end1.skip.stats == 0 && end1.skip.relu));
  const GmflowOp& end3 = b.ops[10];
  CHECK(end3.kind == GmflowOpKind::kBlockEnd);
  CHECK((end3.skip.stats >= 0 && !end3.skip.relu));
  CHECK(b.convs[7].bias == std::vector<float>(64, 0.5F));
  const GmflowOp& quarter = b.ops[22];
  const GmflowOp& eighth = b.ops[23];
  CHECK((quarter.conv == eighth.conv && quarter.stride == 1 && eighth.stride == 2));
  CHECK((quarter.out == b.quarter && eighth.out == b.eighth));
}

TEST_CASE("GMFlow natif : les fautes du faux squelette sont refusées", "[prepare][gmflow]") {
  const auto refused = [](const FakeBackbone& f) {
    const auto backbone = gmflow_backbone(fake_backbone(f));
    return !backbone && backbone.error().code == RefusalCode::FieldMissing;
  };
  CHECK(refused({.trident_stride = 1}));  // le trident ne sort plus le huitième
  CHECK(refused({.layer1_pads = 0}));
  CHECK(refused({.group = 2}));
  CHECK(refused({.dilation = 2}));
  CHECK(refused({.norm_scale = 2.0F}));  // une InstanceNorm apprise
  CHECK(refused({.last_epsilon = 1e-3F}));
  CHECK(refused({.ds_bias = false}));
  CHECK(refused({.add_swapped = true}));
  CHECK(refused({.second_mean_differs = true}));
  CHECK(refused({.conv1_bias = true}));
  CHECK(refused({.c1 = 48}));  // des canaux d'entrée qui ne se prennent pas par 32
  const auto garbage = gmflow_backbone(std::vector<std::byte>(16, std::byte{0x7f}));
  REQUIRE_FALSE(garbage);
  CHECK(garbage.error().code == RefusalCode::ModelUnreadable);
}

TEST_CASE("GMFlow natif : le plan se tient, ses fautes sont refusées", "[prepare][gmflow]") {
  const auto read = gmflow_backbone(fake_backbone(FakeBackbone{}));
  REQUIRE(read);
  CHECK(gmflow_backbone_check(*read));
  const auto broken = [&](auto&& change) {
    GmflowBackbone b = *read;
    change(b);
    const auto ok = gmflow_backbone_check(b);
    return !ok && ok.error().code == RefusalCode::FieldMissing;
  };
  CHECK(broken([](GmflowBackbone& b) { b.ops[1].in.tensor = b.ops[2].out; }));  // pas encore écrit
  CHECK(broken([](GmflowBackbone& b) { b.ops[2].in.stats = 0; }));        // statistiques d'un autre tenseur
  CHECK(broken([](GmflowBackbone& b) { b.ops[4].in.relu = true; }));      // Relu sans InstanceNorm
  CHECK(broken([](GmflowBackbone& b) { b.ops[2].out = b.ops[1].out; }));  // écrit deux fois
  CHECK(broken([](GmflowBackbone& b) { b.ops[9].in = b.ops[3].in; }));    // normalisation sur une 1 × 1
  CHECK(broken([](GmflowBackbone& b) { b.ops[7].stride = 1; }));          // la taille ne suit plus le pas
  CHECK(broken([](GmflowBackbone& b) { b.ops[1].conv = 0; }));            // la 7 × 7 hors de l'image
  CHECK(broken([](GmflowBackbone& b) { b.ops.erase(b.ops.begin() + 2); }));  // une opération manquante
  CHECK(broken([](GmflowBackbone& b) { ++b.stats; }));                       // une InstanceNorm jamais faite
  CHECK(broken([](GmflowBackbone& b) { b.eighth = b.quarter; }));
  CHECK(broken([](GmflowBackbone& b) { b.convs[3].weights.pop_back(); }));
}

TEST_CASE("GMFlow natif : le squelette de la paire exportée se lit", "[prepare][gmflow]") {
  const std::filesystem::path model =
      std::filesystem::path(KIRARI_RIFE_MODEL_DIR) / "gmfss_fortuna_union_pair.onnx";
  if (!std::filesystem::exists(model)) SKIP("gmfss_fortuna_union_pair.onnx absent");
  auto bytes = read_onnx_bytes(model);
  REQUIRE(bytes);
  const auto backbone = gmflow_backbone(*bytes);
  INFO((backbone ? std::string() : backbone.error().detail));
  REQUIRE(backbone);
  const GmflowBackbone& b = *backbone;
  CHECK(b.convs.size() == 17);
  CHECK(b.ops.size() == 24);
  CHECK(b.stats == 15);
  CHECK(std::abs(b.epsilon - 1e-5F) < 1e-9F);
  CHECK(b.pad[0] == as_half(0.485F));  // la moyenne d'ImageNet, déjà en FP16 dans la paire du lecteur
  const std::vector<int> kernels{7, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 1, 3};
  const std::vector<int> outs{64, 64, 64, 64, 64, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128};
  for (std::size_t i = 0; i < b.convs.size(); ++i) {
    CHECK(b.convs[i].kernel == kernels[i]);
    CHECK(b.convs[i].cout == outs[i]);
  }
  CHECK(b.tensors[static_cast<std::size_t>(b.quarter)].channels == 128);
}

}  // namespace kirari::prepare
