// Les clés d'Artifact des moteurs fusionnés du backend TensorRT (gpu/cuda/backend/fused_keys.h), relues sans
// CUDA (B03-09) : une clé écrite se relit à l'identique, une clé d'un autre moteur, abîmée ou qui sortirait
// du dossier du cache est refusée.
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "gpu/cuda/backend/fused_keys.h"

namespace kirari::gpu::cuda {

TEST_CASE("RIFE fusionné (TensorRT) : la clé d'artefact se relit", "[gpu][fused_keys]") {
  const RifeFusedKey key{.height = 1088,
                         .width = 1920,
                         .motion = true,
                         .plan = {.blocks = 4, .reduce = {8, 4, 2, 1}},
                         .engines = {"enc.engine", "b0.engine", "b1.engine", "b2.engine", "b3.engine"}};
  const std::string text = rife_fused_artifact_key(key);
  CHECK(text == "rife-fused:1088x1920:motion:8,4,2,1:enc.engine,b0.engine,b1.engine,b2.engine,b3.engine");
  const auto parsed = rife_fused_artifact_key_parse(text);
  REQUIRE(parsed);
  CHECK(parsed->height == 1088);
  CHECK(parsed->width == 1920);
  CHECK(parsed->motion);
  CHECK(parsed->plan == key.plan);
  CHECK(parsed->engines == key.engines);
}

TEST_CASE("RIFE fusionné (TensorRT) : la clé du warp Catmull-Rom se relit (C03-09)", "[gpu][fused_keys]") {
  const RifeFusedKey key{.height = 576,
                         .width = 960,
                         .motion = false,
                         .warp = kernels::RifeWarp::CatmullRom,
                         .plan = {.blocks = 4, .reduce = {8, 4, 2, 1}},
                         .engines = {"enc.engine", "b0.engine", "b1.engine", "b2.engine", "b3.engine"}};
  const std::string text = rife_fused_artifact_key(key);
  CHECK(text == "rife-fused:576x960:image-cubic:8,4,2,1:enc.engine,b0.engine,b1.engine,b2.engine,b3.engine");
  const auto parsed = rife_fused_artifact_key_parse(text);
  REQUIRE(parsed);
  CHECK_FALSE(parsed->motion);
  CHECK(parsed->warp == kernels::RifeWarp::CatmullRom);
  // Le warp bilinéaire, par défaut, se relit aussi tel quel.
  RifeFusedKey bilinear = key;
  bilinear.warp = kernels::RifeWarp::Bilinear;
  const auto plain = rife_fused_artifact_key_parse(rife_fused_artifact_key(bilinear));
  REQUIRE(plain);
  CHECK(plain->warp == kernels::RifeWarp::Bilinear);
}

TEST_CASE("RIFE fusionné (TensorRT) : une clé abîmée ou hors du cache est refusée", "[gpu][fused_keys]") {
  CHECK_FALSE(rife_fused_artifact_key_parse("0123456789abcdef"));  // un moteur ordinaire
  CHECK_FALSE(rife_fused_artifact_key_parse("drba-fused:1088x1920:4:a,b,c,d"));
  // Un moteur de trop, de moins, un découpage invalide, une taille que les noyaux ne prennent pas.
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused:1088x1920:image:8,4,2,1:a,b,c,d"));
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused:1088x1920:image:8,4,2,3:a,b,c,d,e"));
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused:1000x1920:image:8,4,2,1:a,b,c,d,e"));
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused:1088x1920:autre:8,4,2,1:a,b,c,d,e"));
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused:1088x19x20:image:8,4,2,1:a,b,c,d,e"));
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused:1088x1920:image:8,4,2,1,1,1:a,b,c,d,e,f,g"));
  // Le nom d'un moteur ne sort jamais du dossier du cache.
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused:1088x1920:image:8,4,2,1:../a,b,c,d,e"));
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused:1088x1920:image:8,4,2,1:a,b\\c,c,d,e"));
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused:1088x1920:image:8,4,2,1:..,b,c,d,e"));
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused:1088x1920:image:8,4,2,1:a,,c,d,e"));
}

TEST_CASE("DistilDRBA fusionné (TensorRT) : la clé d'artefact se relit, une clé abîmée est refusée",
          "[gpu][fused_keys]") {
  const DrbaFusedKey key{
      .height = 1088, .width = 1920, .plan = {.tail = 4}, .engines = {"enc", "b0", "b1", "b2"}};
  const std::string text = drba_fused_artifact_key(key);
  CHECK(text == "drba-fused:1088x1920:4:enc,b0,b1,b2");
  const auto parsed = drba_fused_artifact_key_parse(text);
  REQUIRE(parsed);
  CHECK(parsed->height == 1088);
  CHECK(parsed->width == 1920);
  CHECK(parsed->plan == key.plan);
  CHECK(parsed->engines == key.engines);
  CHECK_FALSE(drba_fused_artifact_key_parse("rife-fused:1088x1920:image:8,4,2,1:a,b,c,d,e"));
  CHECK_FALSE(drba_fused_artifact_key_parse("drba-fused:1088x1920:4:enc,b0,b1"));      // un moteur de moins
  CHECK_FALSE(drba_fused_artifact_key_parse("drba-fused:1088x1920:5:enc,b0,b1,b2"));   // découpage
  CHECK_FALSE(drba_fused_artifact_key_parse("drba-fused:1080x1920:4:enc,b0,b1,b2"));   // 1080 % 32
  CHECK_FALSE(drba_fused_artifact_key_parse("drba-fused:1088x1920:4:enc,b0,b1,/b2"));  // hors du cache
  CHECK_FALSE(drba_fused_artifact_key_parse("drba-fused:1088x1920:4x:enc,b0,b1,b2"));
}

TEST_CASE("TemporalFix fusionné (TensorRT) : la clé d'artefact se relit, une clé abîmée est refusée",
          "[gpu][fused_keys]") {
  const TfixFusedKey key{
      .window = true, .height = 540, .width = 960, .half = true, .engines = {"b0", "b1", "b2", "c"}};
  const std::string text = tfix_fused_artifact_key(key);
  CHECK(text == "tfix-fused:window:540x960:1:b0,b1,b2,c");
  const auto parsed = tfix_fused_artifact_key_parse(text);
  REQUIRE(parsed);
  CHECK(parsed->window);
  CHECK(parsed->half);
  CHECK(parsed->height == 540);
  CHECK(parsed->engines == key.engines);
  CHECK(tfix_fused_artifact_key_parse("tfix-fused:encoder:1080x1920:0:enc"));
  CHECK_FALSE(tfix_fused_artifact_key_parse("tfix-fused:encoder:1080x1920:0:enc,b0"));   // un moteur de trop
  CHECK_FALSE(tfix_fused_artifact_key_parse("tfix-fused:window:541x960:0:b0,b1,b2,c"));  // impaire
  CHECK_FALSE(tfix_fused_artifact_key_parse("tfix-fused:window:540x960:2:b0,b1,b2,c"));  // demi-taille
  CHECK_FALSE(tfix_fused_artifact_key_parse("tfix-fused:window:540x960:0:b0,b1,b2,../c"));  // hors du cache
  CHECK_FALSE(tfix_fused_artifact_key_parse("drba-fused:1088x1920:4:enc,b0,b1,b2"));
}

TEST_CASE("GMFSS natif (TensorRT) : la clé d'artefact se relit, une clé abîmée est refusée",
          "[gpu][fused_keys]") {
  const GmfssFusedKey key{.height = 384, .width = 640, .engines = {"fusion.engine", "ifnet.engine"}};
  const std::string text = gmfss_fused_artifact_key(key);
  CHECK(text == "gmfss-fused:384x640:fusion.engine,ifnet.engine");
  const auto parsed = gmfss_fused_artifact_key_parse(text);
  REQUIRE(parsed);
  CHECK(parsed->height == 384);
  CHECK(parsed->width == 640);
  CHECK(parsed->engines == key.engines);
  CHECK_FALSE(gmfss_fused_artifact_key_parse("gmfss-fused:384x640:fusion"));        // l'IFNet manque
  CHECK_FALSE(gmfss_fused_artifact_key_parse("gmfss-fused:380x640:fusion,ifnet"));  // pas un multiple de 32
  CHECK_FALSE(gmfss_fused_artifact_key_parse("gmfss-fused:384x640:fusion,../ifnet"));  // hors du cache
  CHECK_FALSE(gmfss_fused_artifact_key_parse("tfix-fused:encoder:1080x1920:0:enc"));
}

TEST_CASE("GMFSS natif (TensorRT) : la clé de la paire se relit, et ne se prend pas pour celle de la fusion",
          "[gpu][fused_keys]") {
  const GmfssFusedKey key{.height = 384, .width = 640, .engines = {"pair.engine", "flow.engine"}};
  const std::string text = gmfss_pair_artifact_key(key);
  CHECK(text == "gmfss-pair:384x640:pair.engine,flow.engine");
  const auto parsed = gmfss_pair_artifact_key_parse(text);
  REQUIRE(parsed);
  CHECK(parsed->height == 384);
  CHECK(parsed->width == 640);
  CHECK(parsed->engines == key.engines);
  CHECK_FALSE(gmfss_fused_artifact_key_parse(text));  // le backend charge chaque clé par son préfixe
  CHECK_FALSE(gmfss_pair_artifact_key_parse(gmfss_fused_artifact_key(key)));
  CHECK_FALSE(gmfss_pair_artifact_key_parse("gmfss-pair:384x640:pair"));        // le flot manque
  CHECK_FALSE(gmfss_pair_artifact_key_parse("gmfss-pair:384x630:pair,flow"));   // pas un multiple de 32
  CHECK_FALSE(gmfss_pair_artifact_key_parse("gmfss-pair:384x640:pair,/flow"));  // hors du cache
}

}  // namespace kirari::gpu::cuda
