// Découpe de RIFE en six morceaux (prepare/rife_split.h) : l'extraction d'un morceau de graphe sur un modèle
// écrit à la main, la reconnaissance de la structure de RIFE sur un graphe qui en copie les noms, les
// opérateurs et les constantes, ses refus, et, quand les modèles du lecteur sont sur la machine, la découpe
// de RIFE 4.25 et 4.22 lite.
#include "prepare/rife_split.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/model_source.h"
#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"

#ifndef AJP_RIFE_MODEL_DIR
#define AJP_RIFE_MODEL_DIR ""
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

Bytes node(std::string_view name, std::string_view op, std::vector<std::string> in,
           std::vector<std::string> out, const std::vector<Attr>& attrs = {}) {
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
Bytes floats(std::initializer_list<float> values) {
  Bytes raw(values.size() * 4);
  std::size_t i = 0;
  for (const float v : values) {
    const auto bits = std::bit_cast<std::uint32_t>(v);
    std::memcpy(raw.data() + 4 * i++, &bits, 4);
  }
  return raw;
}
Bytes int64s(std::initializer_list<std::int64_t> values) {
  Bytes raw(values.size() * 8);
  std::size_t i = 0;
  for (const std::int64_t v : values) std::memcpy(raw.data() + 8 * i++, &v, 8);
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

// Les puissances de deux de 1/8 à 8 en binary16, deux octets little-endian.
Bytes half(double v) {
  struct Entry {
    double value;
    std::uint16_t bits;
  };
  static constexpr Entry table[]{{8.0, 0x4800}, {4.0, 0x4400},  {2.0, 0x4000},  {1.0, 0x3C00},
                                 {0.5, 0x3800}, {0.25, 0x3400}, {0.125, 0x3000}};
  for (const Entry& e : table) {
    if (e.value == v) return {static_cast<std::byte>(e.bits & 0xFFU), static_cast<std::byte>(e.bits >> 8U)};
  }
  return {std::byte{0}, std::byte{0}};
}

// Un RIFE en miniature : les noms, opérateurs, constantes et attributs de la vraie structure, sans ses poids.
// reduce : la réduction d'entrée de chaque bloc.
struct Fake {
  std::vector<int> reduce{8, 4, 2, 1};
  double resize_in_scale_override = 0.0;  // remplace l'échelle du Resize d'entrée du bloc 1
  double flow_scale_override = 0.0;       // remplace le 1/d du flot devant le bloc 1
  std::int64_t align_corners = 1;
  bool wrong_split_sizes = false;
};

Bytes fake_rife(const Fake& f) {
  Bytes graph;
  std::vector<Bytes> inits;
  const auto add = [&](const Bytes& n) { delimited(graph, 1, n); };
  const auto constant = [&](const std::string& out, std::int32_t type, const Bytes& raw, std::int64_t n) {
    add(node(out, "Constant", {}, {out}, {Attr{.name = "value", .tensor = tensor("", type, n, raw)}}));
  };
  const auto init = [&](const std::string& name, std::int32_t type, const Bytes& raw, std::int64_t n) {
    inits.push_back(tensor(name, type, n, raw));
  };
  const std::vector<Attr> linear{
      {.name = "mode", .str = "linear", .is_text = true},
      {.name = "coordinate_transformation_mode", .str = "half_pixel", .is_text = true}};
  const int blocks = static_cast<int>(f.reduce.size());

  init("onnx::Split_a", 7, int64s({6, 1, 2, 2}), 4);
  init("w", 1, floats({0.0F}), 1);
  add(node("/Split", "Split", {"input", "onnx::Split_a"},
           {"/Split_output_0", "/Split_output_1", "/Split_output_2", "/Split_output_3"}));
  add(node("/Reshape", "Reshape", {"/Split_output_0", "w"}, {"/Reshape_output_0"}));
  add(node("/encode/cnn3/ConvTranspose", "ConvTranspose", {"/Reshape_output_0", "w"},
           {"/encode/cnn3/ConvTranspose_output_0"}));
  add(node("/Reshape_1", "Reshape", {"/encode/cnn3/ConvTranspose_output_0", "w"}, {"/Reshape_1_output_0"}));
  add(node("/Concat_1", "Concat", {"/Split_output_0", "/Reshape_1_output_0", "/Split_output_1"},
           {"/Concat_1_output_0"}));
  init("onnx::Split_b", 7,
       int64s(f.wrong_split_sizes ? std::initializer_list<std::int64_t>{4, 2, 7}
                                  : std::initializer_list<std::int64_t>{4, 1, 8}),
       3);
  for (int b = 0; b < blocks; ++b) {
    const int d = f.reduce[static_cast<std::size_t>(b)];
    const std::string p = std::format("/block{}/", b);
    std::string entry;
    if (b == 0) {
      entry = "/Concat_1_output_0";
    } else {
      const std::string previous = std::format("/block{}/", b - 1);
      std::string flow = std::format("/Add_{}_output_0", b);
      if (d > 1) {
        const std::string scale = std::format("/Constant_flow{}_output_0", b);
        constant(scale, 10, half(b == 1 && f.flow_scale_override != 0.0 ? f.flow_scale_override : 1.0 / d),
                 1);
        flow = std::format("/Mul_{}_output_0", b);
        add(node(std::format("/Mul_{}", b), "Mul", {previous + "Mul_output_0", scale}, {flow}));
      } else {
        add(node(std::format("/Add_{}", b), "Add", {previous + "Mul_output_0", "w"}, {flow}));
      }
      add(node(std::format("/Concat_{}", b + 1), "Concat",
               {"/Reshape_a", "/Reshape_b", "/Split_output_1", previous + "Split_output_1",
                previous + "Split_output_2", flow},
               {std::format("/Concat_{}_output_0", b + 1)}));
      entry = std::format("/Concat_{}_output_0", b + 1);
    }
    std::string conv_in = entry;
    if (d > 1) {
      double scale = 1.0 / d;
      if (b == 1 && f.resize_in_scale_override != 0.0) scale = f.resize_in_scale_override;
      constant(p + "Constant_output_0", 1,
               floats({1.0F, 1.0F, static_cast<float>(scale), static_cast<float>(scale)}), 4);
      add(node(p + "Resize", "Resize", {entry, "", p + "Constant_output_0"}, {p + "Resize_output_0"},
               linear));
      conv_in = p + "Resize_output_0";
    }
    add(node(p + "conv0/conv0.0/conv0.0.0/Conv", "Conv", {conv_in, "w", "w"}, {p + "conv_out"}));
    add(node(p + "lastconv/lastconv.0/ConvTranspose", "ConvTranspose", {p + "conv_out", "w"},
             {p + "lastconv/lastconv.0/ConvTranspose_output_0"}));
    add(node(p + "lastconv/lastconv.1/DepthToSpace", "DepthToSpace",
             {p + "lastconv/lastconv.0/ConvTranspose_output_0"},
             {p + "lastconv/lastconv.1/DepthToSpace_output_0"},
             {{.name = "blocksize", .num = 2}, {.name = "mode", .str = "CRD", .is_text = true}}));
    std::string split_in = p + "lastconv/lastconv.1/DepthToSpace_output_0";
    if (d > 1) {
      constant(p + "Constant_1_output_0", 1,
               floats({1.0F, 1.0F, static_cast<float>(d), static_cast<float>(d)}), 4);
      add(node(p + "Resize_1", "Resize", {split_in, "", p + "Constant_1_output_0"}, {p + "Resize_1_output_0"},
               linear));
      split_in = p + "Resize_1_output_0";
    }
    add(node(p + "Split", "Split", {split_in, "onnx::Split_b"},
             {p + "Split_output_0", p + "Split_output_1", p + "Split_output_2"}));
    if (d > 1) {
      constant(p + "Constant_2_output_0", 10, half(d), 1);
      add(node(p + "Mul", "Mul", {p + "Split_output_0", p + "Constant_2_output_0"}, {p + "Mul_output_0"}));
    }
    add(node(std::format("/GridSample_{}", b), "GridSample", {"x", "g"},
             {std::format("/GridSample_{}_output_0", b)},
             {{.name = "align_corners", .num = f.align_corners},
              {.name = "mode", .str = "bilinear", .is_text = true},
              {.name = "padding_mode", .str = "border", .is_text = true}}));
  }
  add(node("/Sigmoid", "Sigmoid", {std::format("/block{}/Split_output_1", blocks - 1)},
           {"/Sigmoid_output_0"}));
  delimited(graph, 2, text("fake"));
  for (const Bytes& t : inits) delimited(graph, 5, t);
  delimited(graph, 11, value_info("input", 10, {1, 11, -1, -1}));
  delimited(graph, 12, value_info("output", 10, {1, 3, -1, -1}));
  Bytes out;
  integer(out, 1, 8);
  delimited(out, 7, graph);
  Bytes opset;
  integer(opset, 2, 16);
  delimited(out, 8, opset);
  return out;
}

TEST_CASE("Un morceau de graphe est extrait avec ses constantes et ses poids",
          "[prepare][onnx][rife_fused]") {
  Bytes graph;
  const auto add = [&](const Bytes& n) { delimited(graph, 1, n); };
  add(node("n0", "Relu", {"x"}, {"a"}));
  add(node("n1", "Add", {"a", "w"}, {"b"}));
  add(node("n2", "Mul", {"b", "w2"}, {"c"}));
  add(node("n3", "Sigmoid", {"c"}, {"y"}));
  add(node("n4", "Relu", {"x"}, {"unused"}));
  delimited(graph, 5, tensor("w", 1, 1, floats({2.0F})));
  delimited(graph, 5, tensor("w2", 1, 1, floats({3.0F})));
  delimited(graph, 5, tensor("w_unused", 1, 1, floats({4.0F})));
  delimited(graph, 11, value_info("x", 10, {1, 3}));
  delimited(graph, 12, value_info("y", 10, {1, 3}));
  Bytes model;
  integer(model, 1, 8);
  delimited(model, 7, graph);

  const SplitTensor in{.name = "b", .elem_type = 10, .dims = {1, 4, 8, 8}};
  const SplitTensor out{.name = "y", .elem_type = 10};
  const auto cut = extract_subgraph(model, std::span(&in, 1), std::span(&out, 1), "piece");
  REQUIRE(cut);
  const auto parsed = read_onnx_graph(*cut);
  REQUIRE(parsed);
  REQUIRE(parsed->nodes.size() == 2);  // n2 et n3 : tout ce qui précède l'entrée b est coupé
  CHECK(parsed->nodes[0].name == "n2");
  CHECK(parsed->nodes[1].name == "n3");
  CHECK(parsed->initializers == std::vector<std::string>{"w2"});
  const auto summary = scan_onnx(*cut);
  REQUIRE(summary);
  REQUIRE(summary->inputs.size() == 1);
  CHECK(summary->inputs[0].name == "b");
  CHECK(summary->inputs[0].dims == std::vector<std::int64_t>{1, 4, 8, 8});

  // Une sortie que personne ne produit, un tenseur qu'aucune entrée ni constante ne fournit.
  const SplitTensor nothing{.name = "nothing", .elem_type = 10};
  CHECK(extract_subgraph(model, std::span(&in, 1), std::span(&nothing, 1), "p").error().code ==
        RefusalCode::FieldMissing);
  const SplitTensor other{.name = "x", .elem_type = 10, .dims = {1}};
  CHECK(extract_subgraph(model, std::span(&other, 1), std::span(&out, 1), "p").has_value());
  const SplitTensor lost{.name = "zzz", .elem_type = 10, .dims = {1}};
  CHECK(extract_subgraph(model, std::span(&lost, 1), std::span(&out, 1), "p").error().code ==
        RefusalCode::FieldMissing);
}

TEST_CASE("Un graphe à la structure de RIFE se reconnaît et se coupe", "[prepare][onnx][rife_fused]") {
  const Fake fake;
  const Bytes model = fake_rife(fake);
  const auto split = analyze_rife(model);
  REQUIRE(split);
  CHECK(split->plan.blocks == 4);
  CHECK(split->plan.reduce[0] == 8);
  CHECK(split->plan.reduce[1] == 4);
  CHECK(split->plan.reduce[2] == 2);
  CHECK(split->plan.reduce[3] == 1);
  CHECK(split->encode_input == "/Reshape_output_0");
  CHECK(split->encode_output == "/encode/cnn3/ConvTranspose_output_0");
  CHECK(split->block_input[0] == "/block0/Resize_output_0");
  CHECK(split->block_input[3] == "/Concat_4_output_0");  // d = 1 : pas de Resize d'entrée
  CHECK(split->block_output[2] == "/block2/lastconv/lastconv.0/ConvTranspose_output_0");

  const auto pieces = split_rife(model, *split, 128, 256);
  REQUIRE(pieces);
  REQUIRE(pieces->size() == 5);
  CHECK((*pieces)[0].name == "encode");
  CHECK((*pieces)[4].name == "bloc3");
  const auto summary = scan_onnx((*pieces)[2].model);  // le bloc 1 : 28 canaux à 1/4
  REQUIRE(summary);
  REQUIRE(summary->inputs.size() == 1);
  CHECK(summary->inputs[0].name == "/block1/Resize_output_0");
  CHECK(summary->inputs[0].dims == std::vector<std::int64_t>{1, 28, 32, 64});
  const auto encode = scan_onnx((*pieces)[0].model);
  REQUIRE(encode);
  CHECK(encode->inputs[0].dims == std::vector<std::int64_t>{2, 3, 128, 256});
  // Une taille que les noyaux ne prennent pas : refusée avant toute coupe.
  CHECK(split_rife(model, *split, 100, 256).error().code == RefusalCode::ShapeInvalid);
}

TEST_CASE("Un graphe qui s'écarte de RIFE n'est pas coupé", "[prepare][onnx][rife_fused]") {
  Fake wrong_scale;
  wrong_scale.resize_in_scale_override = 0.3;
  CHECK(analyze_rife(fake_rife(wrong_scale)).error().code == RefusalCode::FieldMissing);
  Fake wrong_grid;
  wrong_grid.align_corners = 0;
  CHECK(analyze_rife(fake_rife(wrong_grid)).error().code == RefusalCode::FieldMissing);
  Fake wrong_sizes;
  wrong_sizes.wrong_split_sizes = true;
  CHECK(analyze_rife(fake_rife(wrong_sizes)).error().code == RefusalCode::FieldMissing);
  Fake last_reduces;
  last_reduces.reduce = {8, 4, 2};  // le dernier bloc doit retrouver la pleine résolution
  CHECK_FALSE(analyze_rife(fake_rife(last_reduces)).has_value());
  // Une constante FP16 fausse sur le flot (1/8 au lieu de 1/4 devant le bloc 1) est vue.
  Fake wrong_flow;
  wrong_flow.flow_scale_override = 0.125;
  CHECK(analyze_rife(fake_rife(wrong_flow)).error().code == RefusalCode::FieldMissing);
  const Bytes broken{std::byte{0x3a}, std::byte{0xff}};
  CHECK_FALSE(analyze_rife(broken).has_value());
}

TEST_CASE("Le manifeste d'un RIFE fusionné s'écrit, se relit et refuse l'inattendu",
          "[prepare][rife_fused]") {
  RifeFusedManifest manifest{
      .output = RifeFusedOutput::Motion,
      .height = 192,
      .width = 320,
      .plan = {.blocks = 4, .reduce = {8, 4, 2, 1}},
      .pieces = {"a-encode.onnx", "a-bloc0.onnx", "a-bloc1.onnx", "a-bloc2.onnx", "a-bloc3.onnx"}};
  const auto parsed = parse_manifest(format_manifest(manifest));
  REQUIRE(parsed);
  CHECK(parsed->output == RifeFusedOutput::Motion);
  CHECK(parsed->height == 192);
  CHECK(parsed->width == 320);
  CHECK(parsed->plan == manifest.plan);
  CHECK(parsed->pieces == manifest.pieces);
  CHECK(is_rife_fused_manifest("x/y-image.rifefused"));
  CHECK_FALSE(is_rife_fused_manifest("x/y.onnx"));

  // Un morceau hors du dossier, un nombre de morceaux faux, une taille ou un découpage que les noyaux
  // refusent.
  RifeFusedManifest escaping = manifest;
  escaping.pieces[2] = "../autre.onnx";
  CHECK_FALSE(parse_manifest(format_manifest(escaping)));
  RifeFusedManifest short_list = manifest;
  short_list.pieces.pop_back();
  CHECK_FALSE(parse_manifest(format_manifest(short_list)));
  RifeFusedManifest odd_size = manifest;
  odd_size.height = 100;
  CHECK_FALSE(parse_manifest(format_manifest(odd_size)));
  RifeFusedManifest long_tail = manifest;
  long_tail.plan.reduce[3] = 2;
  CHECK_FALSE(parse_manifest(format_manifest(long_tail)));
  CHECK_FALSE(parse_manifest("ajp-rife-fused 2\noutput image\nsize 192 320\n"));
  CHECK_FALSE(parse_manifest(""));
  CHECK_FALSE(parse_manifest("ajp-rife-fused 1\ninconnu 1\n"));
}

TEST_CASE("La dérivation écrit six morceaux et leur manifeste une fois, puis les relit",
          "[prepare][rife_fused]") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "ajp-rife-split-test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const Bytes model = fake_rife(Fake{});
  const fs::path source = dir / "fake_rife.onnx";
  {
    std::ofstream out(source, std::ios::binary);
    out.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
  }
  const fs::path folder = dir / "modeles";
  const auto manifest = derive_rife_fused(source, folder, 128, 256, RifeFusedOutput::Image);
  REQUIRE(manifest);
  CHECK(manifest->extension() == ".rifefused");
  CHECK(manifest->filename().string().find("fusion1-128x256") != std::string::npos);
  // Lu puis refermé : sous Windows, un fichier resté ouvert ne se remplace pas (la dérivation réécrit le
  // manifeste).
  std::string content;
  {
    std::ifstream text(*manifest);
    content.assign(std::istreambuf_iterator<char>(text), std::istreambuf_iterator<char>());
  }
  const auto parsed = parse_manifest(content);
  REQUIRE(parsed);
  REQUIRE(parsed->pieces.size() == 5);
  for (const std::string& piece : parsed->pieces) CHECK(fs::file_size(folder / piece) > 0);
  // Les morceaux portent la taille : une autre taille ou l'autre sortie ne réécrit pas les morceaux d'avant.
  const auto motion = derive_rife_fused(source, folder, 128, 256, RifeFusedOutput::Motion);
  REQUIRE(motion);
  CHECK(*motion != *manifest);
  const auto stamp = fs::last_write_time(folder / parsed->pieces[1]);
  const auto again = derive_rife_fused(source, folder, 128, 256, RifeFusedOutput::Image);
  REQUIRE(again);
  CHECK(*again == *manifest);
  // Entre parenthèses : Catch ne cherche pas à écrire les durées (libc++ : __int128, ambigu).
  CHECK((fs::last_write_time(folder / parsed->pieces[1]) == stamp));
  // Un morceau perdu : la dérivation le refait au lieu de rendre un manifeste qui ment.
  fs::remove(folder / parsed->pieces[2]);
  REQUIRE(derive_rife_fused(source, folder, 128, 256, RifeFusedOutput::Image));
  CHECK(fs::file_size(folder / parsed->pieces[2]) > 0);
  // Une taille que les noyaux ne prennent pas : refusée, rien d'écrit pour elle.
  const auto odd = derive_rife_fused(source, folder, 100, 256, RifeFusedOutput::Image);
  REQUIRE_FALSE(odd);
  CHECK(odd.error().code == RefusalCode::ShapeInvalid);
  // Un modèle qui n'est pas un RIFE reconnu : refusé, pour que la lecture garde le moteur entier.
  Fake other;
  other.align_corners = 0;
  const Bytes strange = fake_rife(other);
  const fs::path strange_file = dir / "etrange.onnx";
  {
    std::ofstream out(strange_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(strange.data()), static_cast<std::streamsize>(strange.size()));
  }
  const auto refused = derive_rife_fused(strange_file, folder, 128, 256, RifeFusedOutput::Image);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldMissing);
  CHECK_FALSE(derive_rife_fused(dir / "absent.onnx", folder, 128, 256, RifeFusedOutput::Image));
  std::error_code ignored;
  fs::remove_all(dir, ignored);
}

// Les vrais modèles du lecteur, quand ils sont sur la machine (AJP_RIFE_MODEL_DIR) : RIFE 4.25 coupé en 5
// blocs réduits 8, 8, 4, 2, 1 et 4.22 lite en 4 blocs réduits 8, 4, 2, 1.
TEST_CASE("Les RIFE du catalogue se coupent en six morceaux", "[prepare][onnx][rife_fused]") {
  const std::filesystem::path dir = AJP_RIFE_MODEL_DIR;
  struct Known {
    std::string_view file;
    std::vector<int> reduce;
  };
  const std::vector<Known> known{{"rife_v4.25.onnx", {8, 8, 4, 2, 1}},
                                 {"rife_v4.22_lite.onnx", {8, 4, 2, 1}}};
  bool any = false;
  for (const Known& each : known) {
    const std::filesystem::path path = dir / each.file;
    if (dir.empty() || !std::filesystem::is_regular_file(path)) continue;
    any = true;
    INFO(path.string());
    const auto bytes = read_onnx_bytes(path);
    REQUIRE(bytes);
    const auto split = analyze_rife(*bytes);
    REQUIRE(split);
    REQUIRE(split->plan.blocks == static_cast<int>(each.reduce.size()));
    for (std::size_t i = 0; i < each.reduce.size(); ++i) CHECK(split->plan.reduce[i] == each.reduce[i]);
    const auto pieces = split_rife(*bytes, *split, 192, 320);
    REQUIRE(pieces);
    CHECK(pieces->size() == each.reduce.size() + 1);
    std::size_t nodes = 0;
    for (const RifePiece& piece : *pieces) {
      const auto graph = read_onnx_graph(piece.model);
      REQUIRE(graph);
      nodes += graph->nodes.size();
      for (const OnnxFullNode& n : graph->nodes) {
        // Aucune opération pleine résolution dans les morceaux : tout cela est refait par les noyaux.
        CHECK(n.op_type != "GridSample");
        CHECK(n.op_type != "Resize");
        CHECK(n.op_type != "DepthToSpace");
      }
    }
    CHECK(nodes > 100);
  }
  if (!any) SKIP("modèles RIFE du lecteur absents (AJP_RIFE_MODEL_DIR)");
}

}  // namespace
}  // namespace ajp::prepare
