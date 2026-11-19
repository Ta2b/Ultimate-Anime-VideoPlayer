// Le verdict gardé de l'essai du RIFE natif (rife_native_verdict.h) : le fichier relu tel qu'écrit, un
// fichier abîmé ou d'un autre modèle refusé, un nom par identité.
#include "gpu/cuda/backend/rife_native_verdict.h"

#include <limits>
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

namespace verdict = kirari::gpu::cuda::rife_verdict;

TEST_CASE("rife natif, verdict : relu tel qu'écrit") {
  const verdict::Verdict kept{
      .options = {4, 0, 0, 0, 1, 1}, .fused_end = true, .trt_db = 48.512, .apart = {3, 4}};
  const auto read = verdict::parse(verdict::text_of(kept), 6, 4);
  REQUIRE(read.has_value());
  CHECK(read->options == kept.options);
  CHECK(read->fused_end);
  CHECK(read->trt_db == 48.512);
  CHECK(read->apart == kept.apart);
  CHECK_FALSE(verdict::all_trt(*read));

  const verdict::Verdict trt{.options = {0, 0, 0, 0, 0}, .fused_end = false, .trt_db = 0.0};
  const auto back = verdict::parse(verdict::text_of(trt), 5, 4);
  REQUIRE(back.has_value());
  CHECK(verdict::all_trt(*back));
  CHECK_FALSE(back->fused_end);
  CHECK(back->apart.empty());
}

TEST_CASE("rife natif, verdict : un fichier abîmé ou d'un autre modèle ne se lit pas") {
  const std::string good = verdict::text_of({.options = {4, 0, 1}, .fused_end = true, .trt_db = 47.6});
  CHECK_FALSE(verdict::parse(good, 4, 4).has_value());  // un autre nombre de morceaux
  CHECK_FALSE(verdict::parse(good, 3, 3).has_value());  // une option hors de l'échelle
  CHECK_FALSE(verdict::parse(good.substr(0, good.size() / 2), 3, 4).has_value());  // tronqué
  CHECK_FALSE(verdict::parse("", 3, 4).has_value());
  const auto text = [](std::string_view version, std::string_view options, std::string_view fused,
                       std::string_view db, std::string_view apart) {
    return std::string("kirari-rife-natif-verdict ") + std::string(version) + "\noptions " +
           std::string(options) + "\nfused_end " + std::string(fused) + "\ntrt_db " + std::string(db) +
           "\napart " + std::string(apart) + "\n";
  };
  CHECK(verdict::parse(text("2", "1 0 1", "1", "47.6", "-"), 3, 4).has_value());
  CHECK(verdict::parse(text("2", "1 0 1", "1", "47.6", "1"), 3, 4).has_value());
  // Le format d'avant (sans les blocs à part) ne se relit pas : l'essai se refait.
  CHECK_FALSE(verdict::parse("kirari-rife-natif-verdict 1\noptions 1 0 1\nfused_end 1\ntrt_db 47.6\n", 3, 4)
                  .has_value());
  CHECK_FALSE(verdict::parse(text("1", "1 0 1", "1", "47.6", "-"), 3, 4).has_value());
  CHECK_FALSE(verdict::parse(text("2", "1 0 x", "1", "47.6", "-"), 3, 4).has_value());
  CHECK_FALSE(verdict::parse(text("2", "1 0 1", "2", "47.6", "-"), 3, 4).has_value());
  CHECK_FALSE(verdict::parse(text("2", "1 0 1", "1", "nan", "-"), 3, 4).has_value());
  CHECK_FALSE(verdict::parse(text("2", "1  1", "1", "47.6", "-"), 3, 4).has_value());
  CHECK_FALSE(verdict::parse(text("2", "1 0 1", "1", "47.6", ""), 3, 4).has_value());
  CHECK_FALSE(verdict::parse(text("2", "1 0 1", "1", "47.6", "2"), 3, 4).has_value());    // pas de bloc 2
  CHECK_FALSE(verdict::parse(text("2", "1 0 1", "1", "47.6", "0"), 3, 4).has_value());    // bloc 0 : TensorRT
  CHECK_FALSE(verdict::parse(text("2", "1 1 1", "1", "47.6", "1 0"), 3, 4).has_value());  // désordre
  CHECK_FALSE(verdict::parse(text("2", "1 1 1", "1", "47.6", "1 1"), 3, 4).has_value());  // doublon
  CHECK(verdict::parse(text("2", "1 1 1", "1", "47.6", "0 1"), 3, 4).has_value());
}

TEST_CASE("rife natif, verdict : un nom par identité, la chaîne reprise rejugée") {
  const std::string a = verdict::file_name("RTX 4070 Ti|sm8.9|CUDA 13.0|moteurs a|1920x1088|flot");
  const std::string b = verdict::file_name("RTX 4070 Ti|sm8.9|CUDA 13.1|moteurs a|1920x1088|flot");
  CHECK(a != b);
  CHECK(a == verdict::file_name("RTX 4070 Ti|sm8.9|CUDA 13.0|moteurs a|1920x1088|flot"));
  CHECK(a.starts_with("rife-natif-"));
  CHECK(a.ends_with(".verdict"));
  CHECK(a.size() == std::string("rife-natif-").size() + 16 + std::string(".verdict").size());

  const verdict::Verdict kept{.options = {1}, .fused_end = true, .trt_db = 48.5};
  CHECK(verdict::still_holds(kept, 48.5));
  CHECK(verdict::still_holds(kept, 47.6));
  CHECK_FALSE(verdict::still_holds(kept, 47.4));
  CHECK_FALSE(verdict::still_holds(kept, -std::numeric_limits<double>::infinity()));
}
