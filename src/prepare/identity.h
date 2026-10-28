// Identité d'un modèle et nom de son moteur dans le cache, au bit près ceux du fork.
//
// Repris de animejanai-inference-rtx@118c272 :
//   src/program_identity.h:6-80   Program et program_key (clé « animejanai-program-key-v3 »)
//   src/content_identity.h:84-113 files (encadrement « animejanai-content-v1 »)
//   src/model_identity.cpp:20-27  clé de contenu ONNX (« onnx-source-v2 »)
//   src/model_identity.cpp:139-155 engine_name (« aji-v3-<clé>.engine »)
//   src/aji_trt.cpp:542-582       jetons de version TensorRT et d'appareil CUDA
// Les exceptions du fork deviennent des refus ; les clés ne changent pas.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"

namespace ajp::prepare {

enum class ProgramFormat : std::uint8_t { OnnxExport, TensorRt, CoreMl, OrtProvider, MiGraphX };
enum class IoPolicy : std::uint8_t { SourceDefined, Specialized };

struct Binding {
  std::string name;
  std::string type;
  std::vector<std::int64_t> dimensions;  // -1 : dimension symbolique
};

// Tout ce dont dépend un programme compilé : source, conversion, options, outil, appareil.
struct Program {
  ProgramFormat format = ProgramFormat::OnnxExport;
  std::string source_scheme;
  std::string source_digest;  // graphe ET chaque ressource externe
  std::string converter;
  std::uint64_t converter_version = 0;
  std::string options;            // options effectives exactes, précision comprise
  std::string build_environment;  // préfixe d'environnement du constructeur, s'il y en a un
  IoPolicy io_policy = IoPolicy::SourceDefined;
  std::vector<Binding> bindings;  // seulement en IoPolicy::Specialized
  std::string compiler;
  std::string runtime;
  std::string device;
};

std::expected<std::string, Refusal> program_key(Program program);

// Un fichier sous son nom logique : seuls ce nom, la taille et les octets comptent.
struct NamedFile {
  std::string name;
  std::filesystem::path path;
};

std::expected<std::string, Refusal> files_digest(std::string_view domain, std::vector<NamedFile> files);

inline constexpr std::string_view kOnnxSourceScheme = "onnx-source-v2";

// Clé de contenu d'un ONNX : son graphe et chacun de ses poids externes, rangés
// sous folder aux emplacements donnés (relatifs, déjà validés).
std::expected<std::string, Refusal> onnx_source_key(std::span<const std::byte> graph,
                                                    const std::filesystem::path& folder,
                                                    std::span<const std::string> locations);

// Version de TensorRT et de CUDA du constructeur, jeton « 11.3.0-headers-build99-cuda13000-driver13000 ».
struct TrtRuntime {
  int major = 0;
  int minor = 0;
  int patch = 0;
  int headers_build = 0;  // NV_TENSORRT_BUILD des en-têtes
  int cuda_runtime = 0;   // cudaRuntimeGetVersion
  int cuda_driver = 0;    // cudaDriverGetVersion
  std::string token() const;
};

// Décode getInferLibVersion() : 110300 → 11.3.0, et l'ancien codage 8601 → 8.6.1.
TrtRuntime decode_infer_lib_version(std::int32_t version);

// Appareil CUDA, jeton « NVIDIA GeForce RTX 4070 Ti-sm8.9-<uuid en hexadécimal> ».
struct CudaDevice {
  std::string name;
  int sm_major = 0;
  int sm_minor = 0;
  gpu::DeviceUuid uuid{};
  std::string token() const;
};

struct EngineIdentity {
  std::string content;   // onnx_source_key
  std::string settings;  // réglages exacts, avant nettoyage pour TensorRT 11
  std::string runtime;   // TrtRuntime::token
  std::string device;    // CudaDevice::token
  std::string build_environment;
  std::string source_scheme = std::string(kOnnxSourceScheme);
};

// Nom du fichier moteur : « aji-v3-<program_key>.engine ».
std::expected<std::string, Refusal> engine_name(const EngineIdentity& identity);

}  // namespace ajp::prepare
