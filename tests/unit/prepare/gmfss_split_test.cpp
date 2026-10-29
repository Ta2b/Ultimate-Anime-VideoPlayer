// Découpe du graphe « paire » de GMFSS Fortuna en deux morceaux (prepare/gmfss_split.h) : la reconnaissance
// de la structure sur un graphe qui en copie les noms, les opérateurs et les constantes, ses refus, la
// dérivation et son cache et, quand le modèle du lecteur est sur la machine, la découpe du vrai graphe
// demi-flot.
#include "prepare/gmfss_split.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/model_source.h"
#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"

#ifndef AJP_GMFSS_MODEL_DIR
#define AJP_GMFSS_MODEL_DIR ""
#endif

namespace ajp::prepare {
namespace {

using Bytes = std::vector<std::byte>;

void varint(Bytes& out, std::uint64_t v) {
  while (v >= 0x80U) {
    out.push_back(static_cast<std::byte>((v & 0x7fU) | 0x80U));
    v >>= 7U;
  }
  out.push_back(static_cast<std::byte>(v));
}
void delimited(Bytes& out, std::uint32_t number, const Bytes& payload) {
  varint(out, (std::uint64_t{number} << 3U) | 2U);
  varint(out, payload.size());
  out.insert(out.end(), payload.begin(), payload.end());
}
void integer(Bytes& out, std::uint32_t number, std::uint64_t value) {
  varint(out, (std::uint64_t{number} << 3U) | 0U);
  varint(out, value);
}
Bytes text(std::string_view s) {
  return {reinterpret_cast<const std::byte*>(s.data()),
          reinterpret_cast<const std::byte*>(s.data()) + s.size()};
}

struct Attr {
  std::string name;
  std::string str;
  std::int64_t num = 0;
  bool is_text = false;
  Bytes tensor;  // TensorProto d'un attribut « value » (Constant)
};

Bytes attribute(const Attr& a) {
  Bytes out;
  delimited(out, 1, text(a.name));
  if (!a.tensor.empty()) {
    delimited(out, 5, a.tensor);
  } else if (a.is_text) {
    delimited(out, 4, text(a.str));
  } else {
    integer(out, 3, static_cast<std::uint64_t>(a.num));
  }
  return out;
}

Bytes node(std::string_view name, std::string_view op, const std::vector<std::string>& in,
           const std::vector<std::string>& out, const std::vector<Attr>& attrs = {}) {
  Bytes n;
  for (const std::string& i : in) delimited(n, 1, text(i));
  for (const std::string& o : out) delimited(n, 2, text(o));
  delimited(n, 3, text(name));
  delimited(n, 4, text(op));
  for (const Attr& a : attrs) delimited(n, 5, attribute(a));
  return n;
}

// TensorProto dims = {n}, type, raw_data.
Bytes tensor(std::string_view name, std::int32_t type, std::int64_t n, const Bytes& raw) {
  Bytes t;
  integer(t, 1, static_cast<std::uint64_t>(n));
  integer(t, 2, static_cast<std::uint64_t>(type));
  if (!name.empty()) delimited(t, 8, text(name));
  delimited(t, 9, raw);
  return t;
}
Bytes int64s(std::initializer_list<std::int64_t> values) {
  Bytes raw(values.size() * 8);
  std::size_t i = 0;
  for (const std::int64_t v : values) std::memcpy(raw.data() + 8 * i++, &v, 8);
  return raw;
}
Bytes floats(std::initializer_list<float> values) {
  Bytes raw(values.size() * 4);
  std::size_t i = 0;
  for (const float v : values) {
    const auto bits = std::bit_cast<std::uint32_t>(v);
    std::memcpy(raw.data() + 4 * i++, &bits, 4);
  }
  return raw;
}

Bytes value_info(std::string_view name, std::int32_t type, std::vector<std::int64_t> dims) {
  Bytes shape;
  for (const std::int64_t d : dims) {
    Bytes one;
    integer(one, 1, static_cast<std::uint64_t>(d));
    delimited(shape, 1, one);
  }
  Bytes inner;
  integer(inner, 1, static_cast<std::uint64_t>(type));
  delimited(inner, 2, shape);
  Bytes t;
  delimited(t, 1, inner);
  Bytes info;
  delimited(info, 1, text(name));
  delimited(info, 2, t);
  return info;
}

// Une paire en miniature : les noms, opérateurs et constantes de la vraie structure, sans ses poids. Chaque
// côté de FeatureNet est une chaîne de trois convolutions (niveaux 1 à 3) de l'image de son côté ; le flot et
// la métrique se réduisent à un Add des images réduites de moitié.
struct Fake {
  std::string input = "frames";
  std::int64_t first_end = 3;     // fin du Slice de l'image a
  bool v1_export = false;         // sortie source0_1 au lieu de feature0_1
  bool other_weights = false;     // le côté b utilise d'autres poids de FeatureNet
  bool extra_feature_op = false;  // un nœud de plus au côté b
  std::int64_t planes = 6;        // plans de l'entrée
  std::int32_t input_type = 10;   // FLOAT16
};

Bytes fake_pair(const Fake& f) {
  Bytes graph;
  const auto add = [&](const Bytes& n) { delimited(graph, 1, n); };
  const auto constant = [&](const std::string& name, std::int64_t value) {
    add(node(name, "Constant", {}, {name},
             {Attr{.name = "value", .tensor = tensor("", 7, 1, int64s({value}))}}));
  };
  // Les deux Slice de « frames » : starts, ends, axes, steps comme l'export les écrit.
  const auto slice = [&](int k, std::int64_t first, std::int64_t last) {
    const std::string p = std::format("/c{}_", k);
    constant(p + "start", first);
    constant(p + "end", last);
    constant(p + "axes", 1);
    constant(p + "step", 1);
    add(node(k == 0 ? "/Slice" : "/Slice_1", "Slice",
             {"frames", p + "start", p + "end", p + "axes", p + "step"},
             {k == 0 ? "/Slice_output_0" : "/Slice_1_output_0"}));
  };
  slice(0, 0, f.first_end);
  slice(1, 3, 6);
  const std::string names[2] = {"/Slice_output_0", "/Slice_1_output_0"};
  for (int side = 0; side < 2; ++side) {
    const std::string suffix = side == 0 ? "" : "_1";
    const auto weight = [&](int level) {
      return std::format("gmfss.feat_ext.block{}.weight{}", level, side == 1 && f.other_weights ? "_b" : "");
    };
    const std::string level1 = side == 0 && f.v1_export ? "source0_1" : std::format("feature{}_1", side);
    add(node(std::format("/feat_ext/block1/Conv{}", suffix), "Conv", {names[side], weight(1)}, {level1}));
    add(node(std::format("/feat_ext/block2/Conv{}", suffix), "Conv", {level1, weight(2)},
             {std::format("source{}_2", side)}));
    const std::string last = std::format("source{}_2", side);
    if (side == 1 && f.extra_feature_op) {
      add(node("/feat_ext/block3/Relu_1", "Relu", {last}, {"/feat_ext/relu_output"}));
      add(node("/feat_ext/block3/Conv_1", "Conv", {"/feat_ext/relu_output", weight(3)}, {"source1_3"}));
    } else {
      add(node(std::format("/feat_ext/block3/Conv{}", suffix), "Conv", {last, weight(3)},
               {std::format("source{}_3", side)}));
    }
    add(node(std::format("/Resize{}", side == 0 ? "" : "_1"), "Resize", {names[side], "", "w"},
             {std::format("half{}", side)}));
    for (int level = 1; level <= 3; ++level) {
      add(node(std::format("/Concat{}_{}", side, level), "Add", {std::format("half{}", side), "w"},
               {std::format("motion{}_{}", side, level)}));
    }
  }
  delimited(graph, 2, text("fake"));
  delimited(graph, 5, tensor("w", 1, 1, floats({0.0F})));
  for (const char* weight :
       {"gmfss.feat_ext.block1.weight", "gmfss.feat_ext.block2.weight", "gmfss.feat_ext.block3.weight",
        "gmfss.feat_ext.block1.weight_b", "gmfss.feat_ext.block2.weight_b", "gmfss.feat_ext.block3.weight_b"})
    delimited(graph, 5, tensor(weight, 1, 1, floats({1.0F})));
  delimited(graph, 11, value_info(f.input, f.input_type, {1, f.planes, -1, -1}));
  Bytes out;
  integer(out, 1, 8);
  delimited(out, 7, graph);
  Bytes opset;
  integer(opset, 2, 17);
  delimited(out, 8, opset);
  return out;
}

TEST_CASE("Une paire à la structure de GMFSS se reconnaît et se coupe en deux",
          "[prepare][onnx][gmfss_fused]") {
  const Bytes model = fake_pair(Fake{});
  REQUIRE(analyze_gmfss(model));
  const auto pieces = split_gmfss(model, 128, 256);
  REQUIRE(pieces);
  const auto features = scan_onnx(pieces->features);
  REQUIRE(features);
  REQUIRE(features->inputs.size() == 1);
  CHECK(features->inputs[0].name == "/Slice_output_0");
  CHECK(features->inputs[0].dims == std::vector<std::int64_t>{1, 3, 128, 256});
  const auto feature_graph = read_onnx_graph(pieces->features);
  REQUIRE(feature_graph);
  CHECK(feature_graph->nodes.size() == 3);  // les trois convolutions du côté a, rien d'autre
  const auto flow = scan_onnx(pieces->flow);
  REQUIRE(flow);
  REQUIRE(flow->inputs.size() == 1);
  CHECK(flow->inputs[0].name == "frames");
  CHECK(flow->inputs[0].dims == std::vector<std::int64_t>{1, 6, 128, 256});
  const auto flow_graph = read_onnx_graph(pieces->flow);
  REQUIRE(flow_graph);
  // Les deux Slice, les deux Resize, les six mouvements : aucune convolution de FeatureNet.
  for (const OnnxFullNode& n : flow_graph->nodes) CHECK(n.op_type != "Conv");
  CHECK(std::ranges::count_if(flow_graph->nodes,
                              [](const OnnxFullNode& n) { return n.op_type == "Slice"; }) == 2);
  // Une taille qui n'est pas un multiple de 8 : refusée avant toute coupe.
  CHECK(split_gmfss(model, 100, 256).error().code == RefusalCode::ShapeInvalid);
  CHECK(split_gmfss(model, 128, 0).error().code == RefusalCode::ShapeInvalid);
}

TEST_CASE("Une paire qui s'écarte de GMFSS n'est pas coupée", "[prepare][onnx][gmfss_fused]") {
  Fake wrong_slice;
  wrong_slice.first_end = 4;
  CHECK(analyze_gmfss(fake_pair(wrong_slice)).error().code == RefusalCode::FieldMissing);
  Fake old_export;
  old_export.v1_export = true;  // sortie source0_1 de 67 canaux : pas d'export version 2
  CHECK(analyze_gmfss(fake_pair(old_export)).error().code == RefusalCode::FieldMissing);
  Fake other_weights;
  other_weights.other_weights = true;  // les deux côtés ne sont plus le même réseau
  CHECK(analyze_gmfss(fake_pair(other_weights)).error().code == RefusalCode::FieldMissing);
  Fake extra;
  extra.extra_feature_op = true;
  CHECK(analyze_gmfss(fake_pair(extra)).error().code == RefusalCode::FieldMissing);
  Fake renamed;
  renamed.input = "images";
  CHECK(analyze_gmfss(fake_pair(renamed)).error().code == RefusalCode::InputMismatch);
  Fake planes;
  planes.planes = 7;
  CHECK(analyze_gmfss(fake_pair(planes)).error().code == RefusalCode::InputMismatch);
  Fake single;
  single.input_type = 1;
  CHECK(analyze_gmfss(fake_pair(single)).error().code == RefusalCode::InputMismatch);
  const Bytes broken{std::byte{0x3a}, std::byte{0xff}};
  CHECK_FALSE(analyze_gmfss(broken).has_value());
}

TEST_CASE("La dérivation écrit les deux morceaux une fois, puis les relit", "[prepare][gmfss_fused]") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "ajp-gmfss-split-test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const Bytes model = fake_pair(Fake{});
  const fs::path source = dir / "fake_pair.onnx";
  {
    std::ofstream out(source, std::ios::binary);
    out.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
  }
  const fs::path folder = dir / "modeles";
  const auto files = derive_gmfss_fused(source, folder, 128, 256);
  REQUIRE(files);
  CHECK(files->features.filename().string().find("fusion1-128x256") != std::string::npos);
  CHECK(fs::file_size(files->features) > 0);
  CHECK(fs::file_size(files->flow) > 0);
  // Une seconde dérivation relit : les morceaux ne sont pas réécrits.
  const auto stamp = fs::last_write_time(files->flow);
  const auto again = derive_gmfss_fused(source, folder, 128, 256);
  REQUIRE(again);
  CHECK(again->flow == files->flow);
  CHECK(fs::last_write_time(files->flow) == stamp);
  // Un morceau perdu : la dérivation le refait.
  fs::remove(files->features);
  REQUIRE(derive_gmfss_fused(source, folder, 128, 256));
  CHECK(fs::file_size(files->features) > 0);
  // Une autre taille a d'autres fichiers ; une taille refusée n'en écrit pas.
  const auto other = derive_gmfss_fused(source, folder, 256, 256);
  REQUIRE(other);
  CHECK(other->flow != files->flow);
  const auto odd = derive_gmfss_fused(source, folder, 100, 256);
  REQUIRE_FALSE(odd);
  CHECK(odd.error().code == RefusalCode::ShapeInvalid);
  // Un modèle qui n'est pas la paire reconnue : refusé, pour que la lecture garde la paire entière.
  Fake strange;
  strange.other_weights = true;
  const Bytes bytes = fake_pair(strange);
  const fs::path strange_file = dir / "etrange.onnx";
  {
    std::ofstream out(strange_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  const auto refused = derive_gmfss_fused(strange_file, folder, 128, 256);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldMissing);
  CHECK_FALSE(derive_gmfss_fused(dir / "absent.onnx", folder, 128, 256));
  std::error_code ignored;
  fs::remove_all(dir, ignored);
}

// Le vrai graphe du lecteur quand il est sur la machine (AJP_GMFSS_MODEL_DIR) : la paire demi-flot du modèle
// base.
TEST_CASE("La paire GMFSS du catalogue se coupe en deux morceaux", "[prepare][onnx][gmfss_fused]") {
  const std::filesystem::path dir = AJP_GMFSS_MODEL_DIR;
  const std::filesystem::path path = dir / "gmfss_fortuna_base_s05_pair.onnx";
  if (dir.empty() || !std::filesystem::is_regular_file(path))
    SKIP("paire GMFSS du lecteur absente (AJP_GMFSS_MODEL_DIR)");
  const auto bytes = read_onnx_bytes(path);
  REQUIRE(bytes);
  if (const auto ok = analyze_gmfss(*bytes); !ok) FAIL(ok.error().detail);
  const auto pieces = split_gmfss(*bytes, 640, 1024);
  REQUIRE(pieces);
  const auto features = read_onnx_graph(pieces->features);
  const auto flow = read_onnx_graph(pieces->flow);
  REQUIRE(features);
  REQUIRE(flow);
  // FeatureNet : 36 nœuds de convolutions et d'activations ; le flot garde presque tout le graphe.
  CHECK(features->nodes.size() == 36);
  CHECK(flow->nodes.size() > 12000);
  for (const OnnxFullNode& n : features->nodes) {
    CHECK((n.op_type == "Conv" || n.op_type == "Relu" || n.op_type == "Neg" || n.op_type == "Mul" ||
           n.op_type == "Sub"));
  }
  for (const OnnxFullNode& n : flow->nodes) CHECK(n.name.find("/feat_ext/") == std::string::npos);
  const auto flow_summary = scan_onnx(pieces->flow);
  REQUIRE(flow_summary);
  CHECK(flow_summary->inputs[0].dims == std::vector<std::int64_t>{1, 6, 640, 1024});
}

}  // namespace
}  // namespace ajp::prepare
