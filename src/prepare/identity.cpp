// Repris de animejanai-inference-rtx@118c272 src/program_identity.h:6-80,
// src/content_identity.h:84-113, src/model_identity.cpp:20-27 et 139-155,
// src/aji_trt.cpp:542-582. Chaque chaîne hachée est celle du fork.
#include "prepare/identity.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <system_error>
#include <utility>

#include "prepare/sha256.h"
#include "prepare/utf8.h"

namespace ajp::prepare {
namespace {

namespace fs = std::filesystem;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::string_view format_name(ProgramFormat format) {
  switch (format) {
    case ProgramFormat::OnnxExport:
      return "onnx-export";
    case ProgramFormat::TensorRt:
      return "tensorrt-plan";
    case ProgramFormat::CoreMl:
      return "coreml-program";
    case ProgramFormat::OrtProvider:
      return "ort-provider-program";
    case ProgramFormat::MiGraphX:
      return "migraphx-program";
  }
  return "";
}

// Ce que program_key exige du programme, dans l'ordre des vérifications du fork.
std::expected<void, Refusal> check_program(const Program& p) {
  if (format_name(p.format).empty())
    return refuse(RefusalCode::FieldOutOfRange, "program_key : format inconnu");
  if (p.source_scheme.empty()) return refuse(RefusalCode::FieldMissing, "program_key : source_scheme absent");
  if (!is_sha256_hex(p.source_digest)) {
    return refuse(RefusalCode::FieldOutOfRange, "program_key : source_digest n'est pas un SHA-256");
  }
  if (p.converter.empty() || p.converter_version == 0) {
    return refuse(RefusalCode::FieldMissing, "program_key : convertisseur ou version absents");
  }
  if ((p.io_policy == IoPolicy::SourceDefined) != p.bindings.empty()) {
    return refuse(RefusalCode::FieldOutOfRange,
                  "program_key : seule une spécialisation déclare ses liaisons");
  }
  const bool portable = p.format == ProgramFormat::OnnxExport;
  const bool any_target = !p.compiler.empty() || !p.runtime.empty() || !p.device.empty();
  const bool full_target = !p.compiler.empty() && !p.runtime.empty() && !p.device.empty();
  if (portable && any_target) {
    return refuse(RefusalCode::FieldOutOfRange, "program_key : un export ONNX ne dépend d'aucun appareil");
  }
  if (!portable && !full_target) {
    return refuse(RefusalCode::FieldMissing, "program_key : compilateur, runtime ou appareil absent");
  }
  return {};
}

}  // namespace

std::expected<std::string, Refusal> program_key(Program p) {
  if (auto ok = check_program(p); !ok) return std::unexpected(ok.error());
  std::ranges::sort(p.bindings, {}, &Binding::name);
  Sha256 hash;
  hash.text("animejanai-program-key-v3");
  hash.text(format_name(p.format));
  hash.text(p.source_scheme);
  hash.text(p.source_digest);
  hash.text(p.converter);
  hash.integer(p.converter_version);
  hash.text(p.options);
  hash.text(p.build_environment);
  hash.text(p.io_policy == IoPolicy::SourceDefined ? "source-defined" : "specialized");
  hash.integer(p.bindings.size());
  const std::string* previous = nullptr;
  for (const Binding& binding : p.bindings) {
    if (binding.name.empty() || (previous && binding.name == *previous) || binding.type.empty() ||
        binding.dimensions.empty()) {
      return refuse(RefusalCode::FieldOutOfRange, "program_key : liaison invalide ou en double");
    }
    previous = &binding.name;
    hash.text(binding.name);
    hash.text(binding.type);
    hash.integer(binding.dimensions.size());
    for (const std::int64_t dim : binding.dimensions) {
      if (dim != -1 && dim <= 0)
        return refuse(RefusalCode::FieldOutOfRange, "program_key : dimension invalide");
      hash.integer(static_cast<std::uint64_t>(dim));
    }
  }
  hash.text(p.compiler);
  hash.text(p.runtime);
  hash.text(p.device);
  return hash.hex();
}

std::expected<std::string, Refusal> files_digest(std::string_view domain, std::vector<NamedFile> files) {
  std::ranges::sort(files, {}, &NamedFile::name);
  Sha256 hash;
  hash.text("animejanai-content-v1");
  hash.text(domain);
  hash.integer(files.size());
  std::vector<char> buffer(1 << 16);
  for (std::size_t i = 0; i < files.size(); ++i) {
    const NamedFile& file = files[i];
    if (i > 0 && file.name == files[i - 1].name) {
      return refuse(RefusalCode::FieldOutOfRange, "identité en double : " + file.name);
    }
    std::error_code error;
    if (!fs::is_regular_file(file.path, error)) {
      return refuse(RefusalCode::ModelUnreadable, "ressource absente : " + utf8_path(file.path));
    }
    // Un code d'erreur par appel : last_write_time effacerait l'erreur de file_size.
    std::error_code size_error;
    std::error_code time_error;
    const auto size = fs::file_size(file.path, size_error);
    const auto modified = fs::last_write_time(file.path, time_error);
    std::ifstream stream(file.path, std::ios::binary);
    if (size_error || time_error || !stream)
      return refuse(RefusalCode::ModelUnreadable, "ressource illisible : " + utf8_path(file.path));
    hash.text(file.name);
    hash.integer(size);
    std::uintmax_t read = 0;
    while (stream) {
      stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      const auto count = static_cast<std::size_t>(stream.gcount());
      hash.update(std::string_view(buffer.data(), count));
      read += count;
    }
    const bool unchanged = stream.eof() && read == size && fs::file_size(file.path, size_error) == size &&
                           !size_error && fs::last_write_time(file.path, time_error) == modified &&
                           !time_error;
    if (!unchanged)
      return refuse(RefusalCode::ModelMismatch,
                    "ressource modifiée pendant sa lecture : " + utf8_path(file.path));
  }
  return hash.hex();
}

std::expected<std::string, Refusal> onnx_source_key(std::span<const std::byte> graph, const fs::path& folder,
                                                    std::span<const std::string> locations) {
  std::vector<NamedFile> weights;
  weights.reserve(locations.size());
  for (const std::string& location : locations) {
    weights.push_back({.name = "external/" + location, .path = folder / path_from_utf8(location)});
  }
  auto weights_digest = files_digest("onnx-external-weights-v1", std::move(weights));
  if (!weights_digest) return weights_digest;
  Sha256 hash;
  hash.text(kOnnxSourceScheme);
  hash.integer(graph.size());
  hash.update(graph);
  hash.text(*weights_digest);
  return hash.hex();
}

std::string TrtRuntime::token() const {
  return std::format("{}.{}.{}-headers-build{}-cuda{}-driver{}", major, minor, patch, headers_build,
                     cuda_runtime, cuda_driver);
}

TrtRuntime decode_infer_lib_version(std::int32_t version) {
  if (version < 10000)
    return {.major = version / 1000, .minor = (version / 100) % 10, .patch = version % 100};
  return {.major = version / 10000, .minor = (version / 100) % 100, .patch = version % 100};
}

std::string CudaDevice::token() const {
  std::string uuid_hex;
  for (const std::uint8_t byte : uuid) uuid_hex += std::format("{:02x}", byte);
  return std::format("{}-sm{}.{}-{}", name, sm_major, sm_minor, uuid_hex);
}

std::expected<std::string, Refusal> engine_name(const EngineIdentity& identity) {
  // Le condensé ONNX lie les types, les contraintes et les formes de sortie ;
  // les réglages exacts lient les profils min/opt/max. La résolution de lecture
  // n'est pas une dépendance d'un moteur dynamique réutilisable.
  auto key = program_key({
      .format = ProgramFormat::TensorRt,
      .source_scheme = identity.source_scheme,
      .source_digest = identity.content,
      .converter = "onnx-to-tensorrt",
      .converter_version = 1,
      .options = identity.settings,
      .build_environment = identity.build_environment,
      .io_policy = IoPolicy::SourceDefined,
      .compiler = "trtexec/tensorrt:" + identity.runtime,
      .runtime = identity.runtime,
      .device = identity.device,
  });
  if (!key) return key;
  return "aji-v3-" + *key + ".engine";
}

}  // namespace ajp::prepare
