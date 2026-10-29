// Découpe de DistilDRBA en quatre morceaux (prepare/drba_split.h) : la reconnaissance de la structure sur un
// graphe fabriqué depuis le câblage attendu (drba_expected_wiring), ses refus, le manifeste, la dérivation
// et, quand le modèle du lecteur est sur la machine, la découpe du vrai DistilDRBA v2 lite.
#include "prepare/drba_split.h"

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

#ifndef AJP_DRBA_MODEL_DIR
#define AJP_DRBA_MODEL_DIR ""
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
Bytes floats(const std::vector<double>& values) {
  Bytes raw(values.size() * 4);
  std::size_t i = 0;
  for (const double v : values) {
    const auto bits = std::bit_cast<std::uint32_t>(static_cast<float>(v));
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

// Un DistilDRBA en miniature : tous les nœuds du câblage attendu, avec leurs entrées, leurs constantes et
// leurs attributs, sans poids. edit : change le câblage avant la fabrication (pour fabriquer un graphe qui
// s'écarte).
struct Edit {
  int tail = 4;
  std::int64_t align_corners = 1;
  bool extra_resize = false;
  bool pair_name_wrong = false;
  std::string rename_node;  // un nœud du câblage renommé (absent du graphe)
  std::string scale_node;   // un nœud dont la première constante est changée en 3,0
  bool drop_fourth_block = true;
};

Bytes fake_drba(const Edit& e) {
  DrbaExpectedWiring wiring = drba_expected_wiring(e.tail);
  Bytes graph;
  const auto add = [&](const Bytes& n) { delimited(graph, 1, n); };
  for (DrbaExpectedConstant& c : wiring.constants) {
    if (c.node == e.scale_node) c.values.assign(c.values.size(), 3.0);
  }
  // Les constantes : un nœud Constant par entrée exigée, nommée d'après le nœud et le rang.
  const auto constant_name = [](const std::string& node_name, std::size_t input) {
    return std::format("{}/c{}", node_name, input);
  };
  for (const DrbaExpectedConstant& c : wiring.constants) {
    const std::string name = constant_name(c.node, c.input);
    add(node(name, "Constant", {}, {name},
             {Attr{.name = "value",
                   .tensor = tensor("", 1, static_cast<std::int64_t>(c.values.size()), floats(c.values))}}));
  }
  for (const DrbaExpectedNode& n : wiring.nodes) {
    std::vector<std::string> inputs = n.inputs;
    for (const DrbaExpectedConstant& c : wiring.constants) {
      if (c.node != n.name) continue;
      if (inputs.size() <= c.input) inputs.resize(c.input + 1);
      inputs[c.input] = constant_name(c.node, c.input);
    }
    std::vector<Attr> attrs;
    if (n.op == "Resize") {
      attrs = {{.name = "mode", .str = "linear", .is_text = true},
               {.name = "coordinate_transformation_mode", .str = "half_pixel", .is_text = true}};
      inputs.resize(3);
      inputs[1].clear();
    } else if (n.op == "GridSample") {
      attrs = {{.name = "align_corners", .num = e.align_corners},
               {.name = "mode", .str = "bilinear", .is_text = true},
               {.name = "padding_mode", .str = "border", .is_text = true}};
      inputs.resize(2);
    } else if (n.op == "DepthToSpace") {
      attrs = {{.name = "blocksize", .num = 2}, {.name = "mode", .str = "CRD", .is_text = true}};
    }
    for (std::string& i : inputs) {
      if (i.empty() && n.op != "Resize") i = "w";
    }
    const std::string name = n.name == e.rename_node ? n.name + "_renomme" : n.name;
    add(node(name, n.op, inputs, {std::format("{}_output_0", n.name)}, attrs));
  }
  if (e.extra_resize) {
    add(node("/Resize_extra", "Resize", {"w", "", "w"}, {"/Resize_extra_output_0"},
             {{.name = "mode", .str = "linear", .is_text = true},
              {.name = "coordinate_transformation_mode", .str = "half_pixel", .is_text = true}}));
  }
  delimited(graph, 2, text("fake"));
  delimited(graph, 5, tensor("w", 1, 1, floats({0.0})));
  delimited(graph, 11, value_info(e.pair_name_wrong ? "autre" : "pair01", 10, {1, 7, -1, -1}));
  delimited(graph, 11, value_info("pair12", 10, {1, 7, -1, -1}));
  delimited(graph, 12, value_info("out", 10, {1, 3, -1, -1}));
  Bytes out;
  integer(out, 1, 8);
  delimited(out, 7, graph);
  Bytes opset;
  integer(opset, 2, 17);
  delimited(out, 8, opset);
  return out;
}

TEST_CASE("Un graphe à la structure de DistilDRBA se reconnaît et se coupe", "[prepare][onnx][drba_fused]") {
  for (const int tail : {4, 8}) {
    INFO("réduction " << tail);
    Edit edit;
    edit.tail = tail;
    const Bytes model = fake_drba(edit);
    const auto split = analyze_drba(model);
    REQUIRE(split);
    CHECK(split->plan.tail == tail);
    CHECK(split->encode_input == "/Resize_output_0/cast16");
    CHECK(split->encode_output == "/encode/cnn3/ConvTranspose_output_0");
    CHECK(split->block_input[0] == "/Concat_output_0");
    CHECK(split->block_input[1] == "/block1/Concat_output_0");
    CHECK(split->block_input[2] == "/block2/Concat_output_0");
    CHECK(split->block_output[2] == "/block2/lastconv/lastconv.0/ConvTranspose_output_0");

    // Une taille multiple de 2 s_0 (32 ou 64) se coupe ; l'encodeur prend le lot de trois images réduites de
    // t.
    const int h = tail == 4 ? 128 : 256;
    const int w = tail == 4 ? 192 : 256;
    const auto pieces = split_drba(model, *split, h, w);
    REQUIRE(pieces);
    REQUIRE(pieces->size() == 4);
    CHECK((*pieces)[0].name == "encode");
    CHECK((*pieces)[3].name == "bloc2");
    const auto encode = scan_onnx((*pieces)[0].model);
    REQUIRE(encode);
    REQUIRE(encode->inputs.size() == 1);
    CHECK(encode->inputs[0].dims == std::vector<std::int64_t>{3, 3, h / tail, w / tail});
    const auto block1 = scan_onnx((*pieces)[2].model);  // 109 canaux à 1/(2 t)
    REQUIRE(block1);
    CHECK(block1->inputs[0].name == "/block1/Concat_output_0");
    CHECK(block1->inputs[0].dims == std::vector<std::int64_t>{1, 109, h / (2 * tail), w / (2 * tail)});
    const auto block0 = scan_onnx((*pieces)[1].model);  // 58 canaux à 1/(4 t)
    REQUIRE(block0);
    CHECK(block0->inputs[0].dims == std::vector<std::int64_t>{1, 58, h / (4 * tail), w / (4 * tail)});
    // Une taille que les noyaux ne prennent pas : refusée avant toute coupe.
    CHECK(split_drba(model, *split, h + 8, w).error().code == RefusalCode::ShapeInvalid);
  }
}

TEST_CASE("Un graphe qui s'écarte de DistilDRBA n'est pas coupé", "[prepare][onnx][drba_fused]") {
  Edit grid;
  grid.align_corners = 0;
  CHECK(analyze_drba(fake_drba(grid)).error().code == RefusalCode::FieldMissing);
  Edit extra;
  extra.extra_resize = true;
  CHECK(analyze_drba(fake_drba(extra)).error().code == RefusalCode::FieldMissing);
  Edit renamed;
  renamed.rename_node = "/GridSample_5";
  CHECK(analyze_drba(fake_drba(renamed)).error().code == RefusalCode::FieldMissing);
  // Une échelle fausse sur le flot du bloc 1, un Resize d'entrée d'une autre réduction, un temps mal doublé.
  for (const char* wrong : {"/Mul_7", "/Resize_14", "/Mul_59", "/Slice_16", "/Clip"}) {
    INFO(wrong);
    Edit edit;
    edit.scale_node = wrong;
    CHECK(analyze_drba(fake_drba(edit)).error().code == RefusalCode::FieldMissing);
  }
  Edit pair;
  pair.pair_name_wrong = true;
  CHECK(analyze_drba(fake_drba(pair)).error().code == RefusalCode::InputMismatch);
  Edit tail;
  tail.tail = 6;  // une réduction que les noyaux ne font pas
  CHECK_FALSE(analyze_drba(fake_drba(tail)).has_value());
  const Bytes broken{std::byte{0x3a}, std::byte{0xff}};
  CHECK_FALSE(analyze_drba(broken).has_value());
}

TEST_CASE("Le manifeste d'un DistilDRBA fusionné s'écrit, se relit et refuse l'inattendu",
          "[prepare][drba_fused]") {
  DrbaFusedManifest manifest{.height = 192,
                             .width = 320,
                             .plan = {.tail = 4},
                             .pieces = {"a-encode.onnx", "a-bloc0.onnx", "a-bloc1.onnx", "a-bloc2.onnx"}};
  const auto parsed = parse_drba_manifest(format_drba_manifest(manifest));
  REQUIRE(parsed);
  CHECK(parsed->height == 192);
  CHECK(parsed->width == 320);
  CHECK(parsed->plan == manifest.plan);
  CHECK(parsed->pieces == manifest.pieces);
  CHECK(is_drba_fused_manifest("x/y.drbafused"));
  CHECK_FALSE(is_drba_fused_manifest("x/y.rifefused"));
  CHECK_FALSE(is_drba_fused_manifest("x/y.onnx"));

  // Un morceau hors du dossier, un nombre de morceaux faux, une taille ou une réduction que les noyaux
  // refusent.
  DrbaFusedManifest escaping = manifest;
  escaping.pieces[2] = "../autre.onnx";
  CHECK_FALSE(parse_drba_manifest(format_drba_manifest(escaping)));
  DrbaFusedManifest short_list = manifest;
  short_list.pieces.pop_back();
  CHECK_FALSE(parse_drba_manifest(format_drba_manifest(short_list)));
  DrbaFusedManifest odd_size = manifest;
  odd_size.height = 100;
  CHECK_FALSE(parse_drba_manifest(format_drba_manifest(odd_size)));
  DrbaFusedManifest odd_tail = manifest;
  odd_tail.plan.tail = 5;
  CHECK_FALSE(parse_drba_manifest(format_drba_manifest(odd_tail)));
  // La réduction 8 (flot à l'échelle 0,5) veut un multiple de 64.
  DrbaFusedManifest wide = manifest;
  wide.plan.tail = 8;
  wide.height = 160;  // multiple de 32, pas de 64
  CHECK_FALSE(parse_drba_manifest(format_drba_manifest(wide)));
  wide.height = 128;  // 128 = 2 × 64
  wide.width = 320;
  CHECK(parse_drba_manifest(format_drba_manifest(wide)));
  CHECK_FALSE(parse_drba_manifest("ajp-drba-fused 2\nsize 192 320\ntail 4\n"));
  CHECK_FALSE(parse_drba_manifest(""));
  CHECK_FALSE(parse_drba_manifest("ajp-drba-fused 1\ninconnu 1\n"));
}

TEST_CASE("La dérivation écrit quatre morceaux et leur manifeste une fois, puis les relit",
          "[prepare][drba_fused]") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "ajp-drba-split-test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const Bytes model = fake_drba(Edit{});
  const fs::path source = dir / "fake_drba.onnx";
  {
    std::ofstream out(source, std::ios::binary);
    out.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
  }
  const fs::path folder = dir / "modeles";
  const auto manifest = derive_drba_fused(source, folder, 128, 192);
  REQUIRE(manifest);
  CHECK(manifest->extension() == ".drbafused");
  CHECK(manifest->filename().string().find("fusion1-128x192") != std::string::npos);
  std::ifstream text(*manifest);
  const std::string content{std::istreambuf_iterator<char>(text), std::istreambuf_iterator<char>()};
  const auto parsed = parse_drba_manifest(content);
  REQUIRE(parsed);
  REQUIRE(parsed->pieces.size() == 4);
  for (const std::string& piece : parsed->pieces) CHECK(fs::file_size(folder / piece) > 0);
  // Une seconde dérivation relit : les morceaux ne sont pas réécrits.
  const auto stamp = fs::last_write_time(folder / parsed->pieces[1]);
  const auto again = derive_drba_fused(source, folder, 128, 192);
  REQUIRE(again);
  CHECK(*again == *manifest);
  // Entre parenthèses : Catch ne cherche pas à écrire les durées (libc++ : __int128, ambigu).
  CHECK((fs::last_write_time(folder / parsed->pieces[1]) == stamp));
  // Un morceau perdu : la dérivation le refait au lieu de rendre un manifeste qui ment.
  fs::remove(folder / parsed->pieces[2]);
  REQUIRE(derive_drba_fused(source, folder, 128, 192));
  CHECK(fs::file_size(folder / parsed->pieces[2]) > 0);
  // Une taille que les noyaux ne prennent pas : refusée, rien d'écrit pour elle.
  const auto odd = derive_drba_fused(source, folder, 100, 192);
  REQUIRE_FALSE(odd);
  CHECK(odd.error().code == RefusalCode::ShapeInvalid);
  // Un modèle qui n'est pas un DistilDRBA reconnu : refusé, pour que la lecture garde le moteur entier.
  Edit other;
  other.align_corners = 0;
  const Bytes strange = fake_drba(other);
  const fs::path strange_file = dir / "etrange.onnx";
  {
    std::ofstream out(strange_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(strange.data()), static_cast<std::streamsize>(strange.size()));
  }
  const auto refused = derive_drba_fused(strange_file, folder, 128, 192);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldMissing);
  CHECK_FALSE(derive_drba_fused(dir / "absent.onnx", folder, 128, 192));
  std::error_code ignored;
  fs::remove_all(dir, ignored);
}

// Les vrais modèles du lecteur, quand ils sont sur la machine (AJP_DRBA_MODEL_DIR) : DistilDRBA v2 lite,
// réduction 4 (distil_drba_v2_lite.onnx) ou 8 (distil_drba_v2_lite_s05.onnx, flot à l'échelle 0,5).
TEST_CASE("Les DistilDRBA du catalogue se coupent en quatre morceaux", "[prepare][onnx][drba_fused]") {
  const std::filesystem::path dir = AJP_DRBA_MODEL_DIR;
  struct Known {
    std::string_view file;
    int tail;
    int height;
    int width;
  };
  const std::vector<Known> known{{"distil_drba_v2_lite.onnx", 4, 192, 320},
                                 {"distil_drba_v2_lite_s05.onnx", 8, 256, 384}};
  bool any = false;
  for (const Known& each : known) {
    const std::filesystem::path path = dir / each.file;
    if (dir.empty() || !std::filesystem::is_regular_file(path)) continue;
    any = true;
    INFO(path.string());
    const auto bytes = read_onnx_bytes(path);
    REQUIRE(bytes);
    const auto split = analyze_drba(*bytes);
    if (!split) FAIL(split.error().detail);
    CHECK(split->plan.tail == each.tail);
    const auto pieces = split_drba(*bytes, *split, each.height, each.width);
    REQUIRE(pieces);
    REQUIRE(pieces->size() == 4);
    std::size_t nodes = 0;
    for (const DrbaPiece& piece : *pieces) {
      const auto graph = read_onnx_graph(piece.model);
      REQUIRE(graph);
      nodes += graph->nodes.size();
      for (const OnnxFullNode& n : graph->nodes) {
        // Aucune opération pleine résolution dans les morceaux : tout cela est refait par les noyaux.
        CHECK(n.op_type != "GridSample");
        CHECK(n.op_type != "Resize");
        CHECK(n.op_type != "DepthToSpace");
        CHECK(n.op_type != "Slice");
      }
    }
    CHECK(nodes > 60);
  }
  if (!any) SKIP("modèles DistilDRBA du lecteur absents (AJP_DRBA_MODEL_DIR)");
}

}  // namespace
}  // namespace ajp::prepare
