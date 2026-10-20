// Réglages de construction TensorRT et ligne de commande de trtexec.
//
// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp :
//   202-208   réglages par défaut des agrandisseurs (TensorRT 11)
//   812-860   sanitize_settings_trt11 : drapeaux du typage faible retirés
//   862-890   make_build_spec : --onnx, --saveEngine, timing cache par moteur
//   1313-1320 réglages de RIFE : construction fortement typée
// et de AnimeJaNaiManager-HDR (manager-v2) AnimeJaNaiConfEditor/ViewModels/
// MainWindowViewModel.cs:1581-1610 : ordre des profils du préréglage « Dynamic ».
// Les réglages hachés dans la clé sont ceux du fork, au caractère près ; seule
// la ligne de commande passe par le nettoyage de TensorRT 11.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"

namespace ajp::prepare {

enum class Recipe : std::uint8_t {
  // Agrandisseurs (SPAN…) : types du réseau, niveau d'optimisation 5.
  Upscaler,
  // RIFE : graphe FP16 dont les GridSample et le calcul de leur grille restent
  // en îlots FP32 (tools/convert_rife_fp16.py du fork). --stronglyTyped fait
  // suivre ces types au constructeur au lieu de tout passer en FP16.
  Fp32Islands,
  // Le même, petit graphe : niveau d'optimisation 5 en plus (model_source.h, kTunedMaxNodes).
  Fp32IslandsTuned,
};

// « 1x3x1080x1920 », l'ordre NCHW de trtexec.
std::string shape_text(const gpu::Shape& shape);

// Largeur x hauteur pour l'OSD : « 1920x1080 », ou « 1280x720-1920x1080 » si min ≠ max.
std::string shapes_label(const gpu::ShapeSet& shapes);

// Chaque dimension positive, et min ≤ opt ≤ max dimension par dimension.
std::expected<void, Refusal> check_shapes(const gpu::ShapeSet& shapes);

// Réglages hachés dans la clé. Sans formes (ONNX statique), aucun profil :
// trtexec 11 refuse même un profil égal aux formes déclarées.
std::string engine_settings(Recipe recipe, std::string_view input_name,
                            const std::optional<gpu::ShapeSet>& shapes);

struct SanitizedSettings {
  std::string kept;                  // réglages pour la ligne de commande
  std::vector<std::string> removed;  // drapeaux retirés ou réécrits, à journaliser
};

// trtexec 11 a abandonné le typage faible : --fp16, --bf16, --int8, --best et
// --buildOnly disparaissent, ainsi que --inputIOFormats, --outputIOFormats et
// --tacticSources ; --workspace=N devient --memPoolSize=workspace:NM.
SanitizedSettings sanitize_trt11(std::string_view settings);

struct TrtexecPaths {
  std::filesystem::path onnx;
  std::filesystem::path engine;        // sortie, dans le dossier de travail
  std::filesystem::path timing_cache;  // copie de travail du cache persistant (builder.cpp)
};

// « <moteur>.timing.cache », persistant à côté du moteur publié : les temps des
// tactiques servent aux constructions suivantes.
std::filesystem::path timing_cache_for(const std::filesystem::path& engine);

// Arguments de trtexec, sans argv[0] ni guillemets : aucun shell ne les relit.
std::vector<std::string> trtexec_arguments(const TrtexecPaths& paths, std::string_view command_settings);

}  // namespace ajp::prepare
