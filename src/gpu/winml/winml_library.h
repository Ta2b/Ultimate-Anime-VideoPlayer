// Les bibliothèques de Windows ML livrées avec le module (ajp_backend_migraphx.dll) : ONNX Runtime
// (onnxruntime.dll, celui de Windows ML) et le catalogue des fournisseurs d'exécution
// (Microsoft.Windows.AI.MachineLearning.dll, son API C WinMLEpCatalog). Chargées par leur chemin,
// à côté du module, jamais par leur nom : un onnxruntime.dll du système ou d'une autre application
// ne s'y substitue pas. Rien n'est lié à l'import, et un module sans elles se charge quand même :
// make_runtime refuse alors, et le lecteur prend Vulkan.
//
// Le fournisseur MIGraphX d'AMD n'est pas livré : Windows le distribue (Windows Update) aux
// machines qui remplissent ses conditions (GPU AMD RDNA 3 ou plus récent, pilote AMD récent,
// Windows 11 24H2), et le catalogue l'installe à la demande.
#pragma once

#include <expected>
#include <filesystem>
#include <functional>
#include <stop_token>
#include <string>

#include "core/types.h"

namespace ajp::gpu::winml_rt {

// Le nom du fournisseur MIGraphX dans le catalogue et dans ONNX Runtime.
inline constexpr const char* kMigraphxProvider = "MIGraphXExecutionProvider";

// Le fournisseur MIGraphX du catalogue.
struct Provider {
  std::string version;            // celle de son paquet (MSIX)
  bool ready = false;             // installé et utilisable
  std::filesystem::path library;  // sa bibliothèque, à enregistrer auprès d'ONNX Runtime (prêt)
};

// Charge les deux bibliothèques une fois par processus (elles restent chargées) et initialise
// l'API d'ONNX Runtime pour Ort::. Refus LibraryUnavailable : absentes, symboles manquants, ou
// hors Windows.
std::expected<void, Refusal> load_libraries();

// La version d'ONNX Runtime chargé (« 1.27.1 »). Après load_libraries.
std::string ort_version();

// Le fournisseur MIGraphX que le catalogue propose sur cette machine. Refus LibraryUnavailable :
// le catalogue ne le propose pas (conditions de Windows ML non remplies), ou ne s'ouvre pas.
std::expected<Provider, Refusal> find_migraphx();

// Le fournisseur prêt : téléchargé et installé par Windows au besoin, progress (0 à 1) appelé du
// fil de l'appelant ; stop l'annule (Cancelled). Refus DeviceUnavailable : l'installation échoue.
std::expected<Provider, Refusal> ensure_migraphx(const std::function<void(double)>& progress,
                                                 const std::stop_token& stop);

}  // namespace ajp::gpu::winml_rt
