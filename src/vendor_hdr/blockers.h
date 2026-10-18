// Ordre des bloqueurs TrueHDR : la première cause qui empêche TrueHDR, pour l'OSD, le panneau
// Moteurs et resolve(). Fonction pure, sans fichier ni GPU : app réunit les faits.
//
// Repris de manager-v2 AnimeJaNaiConfEditor/ViewModels/OptionalSdkViewModel.cs:39-48 :
// système, GPU non NVIDIA, moteur non TensorRT, SDK, module, adaptateur, sortie non PQ.
// Le Manager ne connaissait que Linux ; ici Windows (RTX Video) et Linux (Maxine) ont chacun
// leur SDK, et seul macOS bloque au premier rang.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "core/types.h"
#include "vendor_hdr/sdk_kind.h"

namespace ajp::vendor_hdr {

// Dans l'ordre de priorité : la première qui s'applique est la cause affichée.
enum class TrueHdrBlocker : std::uint8_t {
  UnsupportedOs,      // macOS : aucun SDK TrueHDR
  NotNvidiaGpu,       // GPU de rendu d'un autre fabricant
  EngineNotTensorRt,  // TrueHDR suit le moteur TensorRT, seul à partager le contexte CUDA
  SdkMissing,         // SDK absent, incomplet, abîmé ou d'une autre version
  ModuleMissing,      // bibliothèque du backend TensorRT non installée
  AdapterMissing,     // backend installé sans l'interop CUDA dont TrueHDR a besoin
  OutputNotPq,        // TrueHDR n'agit qu'en sortie PQ
};

// Les faits, tels que app les connaît au moment de résoudre.
struct TrueHdrEnvironment {
  HostOs os = HostOs::Windows;
  bool nvidia_gpu = false;
  bool tensorrt_engine = false;  // moteur choisi pour la lecture
  bool sdk_ready = false;        // SdkStore::status() vaut Ready
  std::string sdk_reason;        // raison lisible du magasin quand le SDK n'est pas prêt
  bool module_present = false;   // ajp_backend_tensorrt installé
  bool adapter_present = false;  // le backend expose le flux CUDA de TrueHDR
  bool output_pq = false;        // présentation en HDR10 PQ
};

struct Blocked {
  TrueHdrBlocker blocker;
  std::string reason;  // phrase française, affichée telle quelle
};

// La première cause dans l'ordre ci-dessus, ou rien si TrueHDR peut être tenté.
std::optional<Blocked> first_blocker(const TrueHdrEnvironment& environment);

// Le refus que resolve() renvoie pour un profil hdr = "truehdr" bloqué.
Refusal to_refusal(const Blocked& blocked);

}  // namespace ajp::vendor_hdr
