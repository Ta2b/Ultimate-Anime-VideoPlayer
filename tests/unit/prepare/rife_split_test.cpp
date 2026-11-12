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

#include "core/model_index.h"
#include "prepare/model_source.h"
#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"
#include "rife_fake.h"

#ifndef AJP_RIFE_MODEL_DIR
#define AJP_RIFE_MODEL_DIR ""
#endif

namespace ajp::prepare {
namespace {

using Bytes = std::vector<std::byte>;

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
  CHECK(parsed->constants.contains(
      "w2"));  // l'initialiseur qui sert est repris, celui qui ne sert pas est coupé
  CHECK_FALSE(parsed->constants.contains("w_unused"));
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
  CHECK(encode->inputs[0].dims == std::vector<std::int64_t>{1, 3, 128, 256});
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

TEST_CASE("Un graphe aux noms de RIFE mais d'une autre colle n'est pas coupé (B04-01)",
          "[prepare][onnx][rife_fused]") {
  // Les noms de modules et les constantes sont ceux de RIFE ; ce sont les maillons que les noyaux refont qui
  // changent. Chacun doit être vu : sinon la découpe donnerait, sans erreur, des images intermédiaires
  // fausses.
  Fake fake;
  std::string_view which;
  bool by_glue_check = true;  // ces défauts sont refusés plus tôt, par la reconnaissance du masque ou du flot
  SECTION("encodage non warpé") {
    fake.unwarped_encodings = true;
    which = "warpée";
  }
  SECTION("flot non accumulé") {
    fake.flow_not_accumulated = true;
    which = "flot";
    by_glue_check = false;  // le Mul 1/d du flot est déjà refusé avant la colle
  }
  SECTION("mélange sans masque") {
    fake.blend_without_mask = true;
    which = "masque";
    by_glue_check = false;
  }
  SECTION("grille indépendante du flot") {
    fake.grid_ignores_flow = true;
    which = "grille";
  }
  const auto split = analyze_rife(fake_rife(fake));
  REQUIRE_FALSE(split.has_value());
  CHECK(split.error().code == RefusalCode::FieldMissing);
  if (by_glue_check) CHECK(split.error().detail.find("colle des noyaux") != std::string::npos);
  CHECK(split.error().detail.find(which) != std::string::npos);
  // Le même graphe, sans défaut, se reconnaît (le contrôle ne refuse pas tout).
  CHECK(analyze_rife(fake_rife(Fake{})).has_value());
}

TEST_CASE("Les constantes d'un tenseur se lisent aussi dans int64_data, int32_data et float_data (B04-03)",
          "[prepare][onnx]") {
  // Un outil qui ré-enregistre un ONNX range parfois ses constantes ailleurs que dans raw_data.
  const auto model_with = [](const Bytes& init) {
    Bytes graph;
    delimited(graph, 1, node("n", "Mul", {"x", "c"}, {"y"}));
    delimited(graph, 5, init);
    Bytes out;
    integer(out, 1, 8);
    delimited(out, 7, graph);
    return out;
  };
  const auto constant_of = [&](const Bytes& init, const char* name) -> std::vector<double> {
    const auto graph = read_onnx_graph(model_with(init));
    REQUIRE(graph);
    const auto found = graph->constants.find(name);
    return found == graph->constants.end() ? std::vector<double>{} : found->second;
  };
  // int64_data empaqueté : dims {4}, type INT64, valeurs 6 1 2 2.
  Bytes packed64;
  integer(packed64, 1, 4);
  integer(packed64, 2, 7);
  Bytes values;
  for (const std::uint64_t v : {6U, 1U, 2U, 2U}) varint(values, v);
  delimited(packed64, 7, values);
  delimited(packed64, 8, text("c"));
  CHECK(constant_of(packed64, "c") == std::vector<double>{6, 1, 2, 2});
  // int64_data non empaqueté : un champ par valeur.
  Bytes loose64;
  integer(loose64, 1, 2);
  integer(loose64, 2, 7);
  integer(loose64, 7, 4);
  integer(loose64, 7, 1);
  delimited(loose64, 8, text("c"));
  CHECK(constant_of(loose64, "c") == std::vector<double>{4, 1});
  // int32_data : les bits d'un FP16 (0x3C00 = 1,0 ; 0x4000 = 2,0).
  Bytes halves;
  integer(halves, 1, 2);
  integer(halves, 2, 10);
  Bytes bits;
  varint(bits, 0x3C00);
  varint(bits, 0x4000);
  delimited(halves, 5, bits);
  delimited(halves, 8, text("c"));
  CHECK(constant_of(halves, "c") == std::vector<double>{1.0, 2.0});
  // float_data non empaqueté : un fixed32 par valeur (étiquette 0x25, puis 4 octets).
  Bytes loose_floats;
  integer(loose_floats, 1, 2);
  integer(loose_floats, 2, 1);
  for (const float v : {0.5F, 3.0F}) {
    loose_floats.push_back(static_cast<std::byte>(0x25));
    const auto raw = std::bit_cast<std::uint32_t>(v);
    for (unsigned i = 0; i < 4; ++i) loose_floats.push_back(static_cast<std::byte>((raw >> (8 * i)) & 0xFFU));
  }
  delimited(loose_floats, 8, text("c"));
  CHECK(constant_of(loose_floats, "c") == std::vector<double>{0.5, 3.0});
  // Une dimension empaquetée tronquée (dernier octet avec bit de suite) est refusée, pas lue à moitié.
  Bytes truncated;
  Bytes dims;
  dims.push_back(static_cast<std::byte>(0x84));  // « 4 » avec bit de suite, sans la suite
  delimited(truncated, 1, dims);
  integer(truncated, 2, 7);
  delimited(truncated, 8, text("c"));
  const auto refused = read_onnx_graph(model_with(truncated));
  REQUIRE_FALSE(refused.has_value());
  CHECK(refused.error().code == RefusalCode::ModelUnreadable);
}

// 4.26 (1/16) et 4.25 lite (1/32, colle entrelacée), convertis par tools/rife/convert_rife_fp16.py (Cast
// autour des Resize).
TEST_CASE("Les variantes de RIFE 4.26 et 4.25 lite se reconnaissent", "[prepare][onnx][rife_fused]") {
  Fake v426;
  v426.reduce = {16, 8, 4, 2, 1};
  v426.casts = true;
  const auto split = analyze_rife(fake_rife(v426));
  REQUIRE(split);
  CHECK(split->plan == kernels::RifeFusedPlan{.blocks = 5, .reduce = {16, 8, 4, 2, 1}});
  CHECK_FALSE(split->interleaved);
  CHECK(split->block_input[0] == "/block0/Resize_output_0/cast16");

  Fake lite;
  lite.reduce = {32, 16, 8, 4, 1};
  lite.casts = true;
  lite.interleaved = true;
  const Bytes model = fake_rife(lite);
  const auto cut = analyze_rife(model);
  REQUIRE(cut);
  CHECK(cut->plan == kernels::RifeFusedPlan{.blocks = 5, .reduce = {32, 16, 8, 4, 1}});
  CHECK(cut->interleaved);
  CHECK(cut->block_weight[1] == "w28");
  CHECK(split_rife(model, *cut, 128, 256).has_value());
  CHECK(split_rife(model, *cut, 64, 256).error().code == RefusalCode::ShapeInvalid);  // 1/128 : pas entier

  // Au-delà du 1/32, les noyaux ne suivent plus.
  Fake too_deep = lite;
  too_deep.reduce = {64, 16, 8, 4, 1};
  CHECK_FALSE(analyze_rife(fake_rife(too_deep)).has_value());
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
  CHECK(parsed->sizes.empty());  // un manifeste d'avant les tailles se lit encore
  for (const RifeFusedOutput out : {RifeFusedOutput::Image, RifeFusedOutput::ImageCubic}) {
    RifeFusedManifest other = manifest;
    other.output = out;
    CHECK(parse_manifest(format_manifest(other))->output == out);
  }
  manifest.sizes = {10, 20, 30, 40, 50};
  const auto sized = parse_manifest(format_manifest(manifest));
  REQUIRE(sized);
  CHECK(sized->sizes == manifest.sizes);
  RifeFusedManifest uneven = manifest;
  uneven.sizes.pop_back();  // autant de tailles que de morceaux, ou aucune
  CHECK_FALSE(parse_manifest(format_manifest(uneven)));
  manifest.sizes.clear();
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
  CHECK(manifest->filename().string().find("fusion3-128x256") != std::string::npos);
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
  // Le manifeste porte la taille de chaque morceau ; un morceau tronqué depuis (disque, antivirus) est
  // réécrit à la dérivation suivante, au lieu d'être gardé « parce qu'il existe » (B04-02).
  REQUIRE(parsed->sizes.size() == parsed->pieces.size());
  const auto whole = fs::file_size(folder / parsed->pieces[3]);
  CHECK(whole == parsed->sizes[3]);
  fs::resize_file(folder / parsed->pieces[3], whole / 2);
  REQUIRE(derive_rife_fused(source, folder, 128, 256, RifeFusedOutput::Image));
  CHECK(fs::file_size(folder / parsed->pieces[3]) == whole);
  // Un manifeste d'avant les tailles ne se croit pas : les morceaux sont refaits et le manifeste réécrit
  // avec.
  {
    std::ofstream old(*manifest, std::ios::trunc);
    old << format_manifest(RifeFusedManifest{.output = RifeFusedOutput::Image,
                                             .height = 128,
                                             .width = 256,
                                             .plan = parsed->plan,
                                             .pieces = parsed->pieces});
  }
  fs::resize_file(folder / parsed->pieces[1], 5);
  REQUIRE(derive_rife_fused(source, folder, 128, 256, RifeFusedOutput::Image));
  std::string rewritten;
  {
    std::ifstream text(*manifest);
    rewritten.assign(std::istreambuf_iterator<char>(text), std::istreambuf_iterator<char>());
  }
  CHECK(parse_manifest(rewritten)->sizes.size() == 5);
  CHECK(fs::file_size(folder / parsed->pieces[1]) == parsed->sizes[1]);
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
// blocs réduits 8, 8, 4, 2, 1, 4.22 lite en 4 blocs réduits 8, 4, 2, 1 ; et les candidats s'ils y sont
// (tools/rife/candidats.sh) : 4.26 réduit 16, 8, 4, 2, 1, 4.25 lite 32, 16, 8, 4, 1 à colle entrelacée.
TEST_CASE("Les RIFE du catalogue se coupent en six morceaux", "[prepare][onnx][rife_fused]") {
  const std::filesystem::path dir = AJP_RIFE_MODEL_DIR;
  struct Known {
    std::string_view file;
    std::vector<int> reduce;
  };
  const std::vector<Known> known{{"rife_v4.25.onnx", {8, 8, 4, 2, 1}},
                                 {"rife_v4.22_lite.onnx", {8, 4, 2, 1}},
                                 {"rife_v4.26.onnx", {16, 8, 4, 2, 1}},
                                 {"rife_v4.25_lite.onnx", {32, 16, 8, 4, 1}}};
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
    CHECK(split->interleaved == each.file.starts_with("rife_v4.25_lite"));
    const auto pieces = split_rife(*bytes, *split, 256, 384);  // multiples de 128 : 4.25 lite
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

// A04-07 : un morceau dérivé déjà présent n'est jamais réécrit tant que kRifeFusedDerivation ne change pas.
// Ce test hache les morceaux d'un graphe fabriqué : si la découpe produit autre chose (extract_subgraph,
// l'analyse, value_info), il échoue jusqu'à ce que kRifeFusedDerivation monte, et l'empreinte attendue avec.
// Sans cela, une installation existante garderait les anciens morceaux et une neuve aurait les nouveaux.
TEST_CASE("Les morceaux RIFE ne changent pas sans que leur version monte (A04-07)", "[prepare][rife_fused]") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "ajp-rife-empreinte";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const Bytes model = fake_rife(Fake{});
  const fs::path source = dir / "modele.onnx";
  {
    std::ofstream out(source, std::ios::binary);
    out.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
  }
  const auto manifest = derive_rife_fused(source, dir / "modeles", 128, 256, RifeFusedOutput::Image);
  REQUIRE(manifest);
  std::string content;
  {
    std::ifstream text(*manifest);
    content.assign(std::istreambuf_iterator<char>(text), std::istreambuf_iterator<char>());
  }
  const auto parsed = parse_manifest(content);
  REQUIRE(parsed);
  std::string hashes;
  for (const std::string& piece : parsed->pieces) {
    const auto sha = core::sha256_file(manifest->parent_path() / piece);
    REQUIRE(sha);
    hashes += *sha;
  }
  const std::string fingerprint = core::sha256_hex(hashes);
  INFO("empreinte des morceaux : " << fingerprint);
  CHECK(std::string(kRifeFusedDerivation) == "fusion3");
  CHECK(fingerprint == "e48e872256365e95505145e9ab4bfffdff35d120054f1bcf31953c672ef2329e");
  std::error_code ignored;
  fs::remove_all(dir, ignored);
}

}  // namespace
}  // namespace ajp::prepare
