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
  // suivre ces types au constructeur au lieu de tout passer en FP16. Gardé pour un RIFE ou un modèle à
  // GridSample de plus de kTunedMaxNodes nœuds donné par --rife-model ou --model : aucun modèle livré n'y va
  // depuis le retrait de GMFSS, mais la branche n'est pas morte pour un fichier de l'utilisateur (B05-09).
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
  std::filesystem::path timing_cache;  // copie de travail du cache de l'appareil (builder.h)
  std::filesystem::path layer_info;    // rapport des couches (--exportLayerInfo) ; vide : aucun
};

// « <moteur>.layers.json » : le rapport des couches de la dernière construction, à côté du moteur.
std::filesystem::path layer_info_for(const std::filesystem::path& engine);

// Version de la ligne de commande de trtexec, hachée dans la clé du moteur (build_environment). Ni le
// nettoyage de TensorRT 11 (sanitize_trt11) ni les arguments que trtexec_arguments ajoute n'entrent dans les
// réglages hachés : qui change l'un d'eux, au point de changer les tactiques ou la précision d'un moteur,
// monte cette constante, sinon les moteurs déjà en cache, construits autrement, seraient repris tels quels
// (C05-09). Un test fige la ligne actuelle (trtexec_command_test.cpp) et rappelle de la monter. La version 1
// est celle d'avant la constante : sa clé reste vide, les moteurs déjà construits restent valables.
inline constexpr int kCommandLineVersion = 1;
inline std::string command_line_version() {
  return kCommandLineVersion == 1 ? std::string() : "trtexec-command-v" + std::to_string(kCommandLineVersion);
}

// Arguments de trtexec, sans argv[0] ni guillemets : aucun shell ne les relit. Avec un rapport des
// couches (étude 4, 6.1), --profilingVerbosity=detailed et --exportLayerInfo, hors de la clé : ils
// ne changent ni les tactiques ni la précision, et un moteur déjà construit reste valable.
// --avgTiming n'y est pas : trtexec 11 en fait déjà 8 mesures par tactique (sampleOptions.h,
// defaultAvgTiming), ce que l'étude croyait à 1.
// cuda_device (C37-01) : l'ordinal CUDA de la carte du rendu, que le backend a choisie par UUID ; trtexec
// construit sinon sur l'appareil 0, qui n'est pas forcément elle (deux cartes NVIDIA, eGPU) : le moteur,
// nommé pour la carte du rendu, était refusé au chargement et le cache de temps rempli des tactiques de
// l'autre. trtexec hérite de l'environnement du lecteur (CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER) : il
// numérote les cartes comme lui. Négatif : aucun appareil demandé (outils, tests).
std::vector<std::string> trtexec_arguments(const TrtexecPaths& paths, std::string_view command_settings,
                                           int cuda_device = -1);

// La carte sur laquelle trtexec dit construire (« [I] Selected Device: NVIDIA GeForce RTX 4070 ») ; rien si
// la ligne n'est pas celle-là. Le constructeur la compare à la carte du rendu (C37-01).
std::optional<std::string> trtexec_selected_device(std::string_view line);

// Le trtexec que le moteur lance (P1.21) : celui posé à côté du module, pris seulement si son SHA-256
// est celui relevé à la construction du module ; sinon `fallback`, le trtexec du poste de
// développement, vide dans un build Release. Vide : aucun trtexec, le moteur ne construit rien.
std::filesystem::path pick_trtexec(const std::filesystem::path& beside, std::string_view expected_sha256,
                                   const std::filesystem::path& fallback);

// Ce que dit un rapport des couches de trtexec (JSON de IEngineInspector) : couches, dont les
// reformatages (LayerType « Reformat ») ; le détail, couches FP32 hors îlots comprises, est dans
// tools/trt/layer_report.py. Lu sans analyseur JSON : seules les clés LayerType comptent.
struct LayerCounts {
  int layers = 0;
  int reformats = 0;
};
LayerCounts layer_counts(std::string_view json);

}  // namespace ajp::prepare
