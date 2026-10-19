// Construction d'un moteur réel par prepare::TrtexecBuilder, comme le thread
// preparer la lancera : clé du fork, trtexec en sous-processus, progression,
// timing cache, puis vérification par désérialisation avec le runtime.
//
//   ajp_trt_build --trtexec <trtexec> --cache <dossier> --onnx <modèle.onnx> --shape 1x3x720x1280
//                 [--min 1x3x..] [--max 1x3x..] [--precision f16|f32] [--id <nom>] [--plan-only]
//
// --plan-only écrit le nom du moteur et la ligne de commande sans rien lancer :
// le nom se compare à celui d'un moteur du lecteur actuel (même clé, même cache).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "gpu/backend.h"
#include "gpu/cuda/trt/cuda_objects.h"
#include "gpu/cuda/trt/log.h"
#include "gpu/cuda/trt/runtime.h"
#include "gpu/cuda/trt/tool_args.h"
#include "prepare/builder.h"
#include "prepare/model_source.h"
#include "prepare/sha256.h"

namespace {

namespace fs = std::filesystem;
using ajp::Refusal;
using ajp::RefusalCode;

struct Options {
  fs::path trtexec;
  fs::path cache;
  fs::path onnx;
  std::string id;
  std::string opt;
  std::string min;
  std::string max;
  ajp::gpu::TensorType precision = ajp::gpu::TensorType::F16;
  bool plan_only = false;
};

void out(const std::string& text) {
  std::fputs(text.c_str(), stdout);
  std::fflush(stdout);
}

Refusal bad_argument(std::string detail) {
  return Refusal{RefusalCode::OptionUnknown, std::move(detail)};
}

std::expected<Options, Refusal> parse(int argc, const char* const* argv) {
  Options o;
  for (int k = 1; k < argc; ++k) {
    const std::string_view arg = argv[k];
    if (arg == "--plan-only") {
      o.plan_only = true;
      continue;
    }
    if (k + 1 >= argc) return std::unexpected(bad_argument(std::format("{} sans valeur", arg)));
    const std::string_view value = argv[++k];
    if (arg == "--trtexec") {
      o.trtexec = fs::path(value);
    } else if (arg == "--cache") {
      o.cache = fs::path(value);
    } else if (arg == "--onnx") {
      o.onnx = fs::path(value);
    } else if (arg == "--id") {
      o.id = value;
    } else if (arg == "--shape") {
      o.opt = value;
    } else if (arg == "--min") {
      o.min = value;
    } else if (arg == "--max") {
      o.max = value;
    } else if (arg == "--precision" && (value == "f16" || value == "f32")) {
      o.precision = value == "f16" ? ajp::gpu::TensorType::F16 : ajp::gpu::TensorType::F32;
    } else {
      return std::unexpected(bad_argument(std::format("argument inattendu « {} {} »", arg, value)));
    }
  }
  if (o.onnx.empty() || o.opt.empty() || o.cache.empty() || (o.trtexec.empty() && !o.plan_only)) {
    return std::unexpected(bad_argument("--onnx, --shape, --cache et --trtexec sont obligatoires"));
  }
  if (o.id.empty()) o.id = o.onnx.stem().string();
  return o;
}

// « 1x3x720x1280 » vers une forme NCHW.
std::expected<ajp::gpu::Shape, Refusal> parse_shape(std::string_view text) {
  const auto dims = ajp::gpu::trt::tool::parse_dims(text);
  ajp::gpu::Shape shape;
  if (!dims || dims->size() != shape.dims.size()) {
    return std::unexpected(bad_argument(std::format("forme NCHW AxBxCxD attendue, reçu « {} »", text)));
  }
  std::ranges::copy(*dims, shape.dims.begin());
  return shape;
}

std::expected<ajp::gpu::ShapeSet, Refusal> shapes_of(const Options& o) {
  auto opt = parse_shape(o.opt);
  auto min = parse_shape(o.min.empty() ? o.opt : o.min);
  auto max = parse_shape(o.max.empty() ? o.opt : o.max);
  if (!opt || !min || !max) return std::unexpected(!opt ? opt.error() : !min ? min.error() : max.error());
  return ajp::gpu::ShapeSet{.min = *min, .opt = *opt, .max = *max};
}

std::expected<void, Refusal> build(const Options& o) {
  auto sha256 = ajp::prepare::sha256_file(o.onnx);
  if (!sha256) return std::unexpected(sha256.error());
  auto shapes = shapes_of(o);
  if (!shapes) return std::unexpected(shapes.error());
  auto device = ajp::gpu::trt::current_device();
  if (!device) return std::unexpected(device.error());
  auto runtime = ajp::gpu::trt::TrtRuntime::create(ajp::gpu::trt::make_logger());
  if (!runtime) return std::unexpected(runtime.error());
  const ajp::gpu::trt::LibraryVersion version = runtime->version();

  const ajp::gpu::ModelSpec model{
      .id = o.id, .onnx_path = o.onnx, .sha256 = *sha256, .precision = o.precision};
  const ajp::prepare::TrtexecConfig config{
      .trtexec = o.trtexec,
      .cache_dir = o.cache,
      .runtime = {.major = version.major,
                  .minor = version.minor,
                  .patch = version.patch,
                  .headers_build = version.headers_build,
                  .cuda_runtime = device->runtime_version,
                  .cuda_driver = device->driver_version},
      .device = {.name = device->name,
                 .sm_major = device->sm_major,
                 .sm_minor = device->sm_minor,
                 .uuid = device->uuid},
  };
  auto source = ajp::prepare::read_model(model, config.input_name);
  if (!source) return std::unexpected(source.error());
  auto plan = ajp::prepare::plan_build(config, model, *source, *shapes);
  if (!plan) return std::unexpected(plan.error());
  out(std::format(
      "modèle {} ({})\nmoteur {}\nréglages hachés : {}\nligne de commande : {}\njeton : {} / {}\n", model.id,
      *sha256, plan->engine_file, plan->settings, plan->command.kept, config.runtime.token(),
      config.device.token()));
  if (o.plan_only) return {};

  const auto started = std::chrono::steady_clock::now();
  const auto seconds = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  };
  const ajp::prepare::TrtexecBuilder builder(config,
                                             [&](const fs::path& engine) -> std::expected<void, Refusal> {
                                               auto loaded = runtime->load_file(engine);
                                               if (!loaded) return std::unexpected(loaded.error());
                                               return {};
                                             });
  ajp::gpu::Progress progress{.report = [&](float fraction, const std::string& stage) {
    out(std::format("[{:7.2f} s] {:3.0f} % {}\n", seconds(), fraction * 100.0F, stage));
  }};
  auto artifact = builder.build(model, *shapes, progress);
  if (!artifact) return std::unexpected(artifact.error());
  std::error_code error;
  out(std::format("construit en {:.2f} s : {} ({} octets)\n", seconds(), artifact->path.string(),
                  fs::file_size(artifact->path, error)));
  return {};
}

int fail(const Refusal& refusal) {
  std::fputs(std::format("ajp_trt_build : {} : {}\n", ajp::name(refusal.code), refusal.detail).c_str(),
             stderr);
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  auto options = parse(argc, argv);
  if (!options) return fail(options.error());
  if (auto done = build(*options); !done) return fail(done.error());
  return 0;
}
