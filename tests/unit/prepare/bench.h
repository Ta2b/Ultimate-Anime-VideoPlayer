// Banc commun des tests de construction : un faux trtexec (helper.cpp copié en « trtexec-<mode> »), un modèle
// et un dossier de moteurs.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "prepare/builder.h"
#include "prepare/sha256.h"
#include "support.h"

namespace ajp::prepare::test_bench {

using namespace std::chrono_literals;
using ajp::gpu::ShapeSet;
using ajp::gpu::TensorType;

inline ShapeSet fixed(std::int64_t channels, std::int64_t h, std::int64_t w) {
  const ajp::gpu::Shape shape{{1, channels, h, w}};
  return {.min = shape, .opt = shape, .max = shape};
}

struct Steps {
  std::vector<std::pair<float, std::string>> seen;
  ajp::gpu::Progress progress() {
    return {.report = [this](float f, const std::string& stage) { seen.emplace_back(f, stage); }, .stop = {}};
  }
};

class Bench {
 public:
  explicit Bench(std::string_view mode, const test::FakeModel& graph = {},
                 TensorType precision = TensorType::F16) {
    const auto fake =
        dir_ / "bin" / ("trtexec-" + std::string(mode) + test::helper_path().extension().string());
    test::fs::create_directories(fake.parent_path());
    test::fs::copy_file(test::helper_path(), fake);
    test::fs::permissions(fake, test::fs::perms::owner_all, test::fs::perm_options::add);
    const std::string onnx = test::onnx(graph);
    test::write_file(dir_ / "modeles/modele.onnx", onnx);
    model = {.id = "modele",
             .onnx_path = dir_ / "modeles/modele.onnx",
             .sha256 = sha256_hex(onnx),
             .precision = precision};
    config = {.trtexec = fake,
              .cache_dir = dir_ / "moteurs",
              .runtime = {.major = 11,
                          .minor = 3,
                          .patch = 0,
                          .headers_build = 99,
                          .cuda_runtime = 13000,
                          .cuda_driver = 13000},
              .device = {.name = "RTX de test", .sm_major = 8, .sm_minor = 9, .uuid = {}},
              .timeout = 10s};
  }

  // Ce que la construction de shapes produirait, calculé sans la lancer.
  BuildPlan plan(const ShapeSet& shapes) const {
    return plan_build(config, model, read_model(model, "input").value(), shapes).value();
  }
  // Lancements de trtexec, comptés par le faux trtexec à côté du modèle.
  std::size_t builder_calls() const { return test::read_lines(model.onnx_path.string() + ".calls").size(); }
  std::vector<std::string> timing_cache() const { return test::read_lines(device_timing_cache(config)); }
  std::vector<std::string> logged_args(const ShapeSet& shapes) const {
    std::vector<std::string> args;
    for (const std::string& line : test::read_lines(plan(shapes).log_file)) {
      if (line.starts_with("[I] arg ")) args.push_back(line.substr(8));
    }
    return args;
  }
  const test::fs::path& cache() const { return config.cache_dir; }

  test::TempDir dir_;
  ajp::gpu::ModelSpec model;
  TrtexecConfig config;
};

inline const ShapeSet kHd = fixed(3, 1080, 1920);

}  // namespace ajp::prepare::test_bench
