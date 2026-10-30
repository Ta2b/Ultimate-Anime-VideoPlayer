// Pont aji_rtx.dll, côté Windows, vers le chemin CUDA du NVIDIA RTX Video SDK 1.1. Le SDK
// n'a pas de DLL d'entrée : son API (NGX) se lie statiquement ; le pont `aji_rtx` (source
// D:/AnimeJaNai-dev/aji-rtx, exemple NVIDIA modifié, lié à nvsdk_ngx_s.lib, cuda.lib et
// cudart.lib) l'enveloppe en trois fonctions C et charge à l'exécution nvngx_truehdr.dll et
// nvngx_vsr.dll, copiées du SDK à côté de lui, avec cudart64_12.dll. Aucun fichier NVIDIA
// n'est dans le dépôt ni dans les versions : l'utilisateur fournit ce dossier.
// Origines : relevé de la source du pont (lecture seule) et appels du moteur
// animejanai-inference-rtx src/aji_trt.cpp (création avec le contexte et le flux 0, entrée
// RGB 10 bits tassée, R dans les bits de poids faible, sortie HDR10 PQ BT.2020 au même
// format). Non exécuté ici : à vérifier sur le PC Windows RTX.
// Le pont ne borne rien : les réglages sont bornés par l'appelant (contraste et saturation
// 0..200, 100 neutre ; gris moyen 10..100, 50 par défaut ; pic 400..2000 nits).
// Une seule instance existe à la fois dans le processus.
#pragma once

namespace ajp::vendor_hdr::rtx_video {

// Contexte CUDA courant sur le thread appelant, flux nul (flux 0). thdr et vsr : fonctions à
// créer (TrueHDR avant Super Resolution). Renvoie 1 si tout est prêt, 0 sinon.
using CreateFn = int(void* cuda_context, void* cuda_stream, int thdr, int vsr);

// Pointeurs de périphérique de pixels uint32 serrés (pas = 4 × largeur) : R bits 0-9, G 10-19,
// B 20-29, alpha 3. Sans Super Resolution, la sortie a la taille de l'entrée. Renvoie 1 si OK.
using EvaluateFn = int(unsigned long long input, unsigned long long output, int input_width, int input_height,
                       int output_width, int output_height, int vsr_quality, int contrast, int saturation,
                       int middle_gray, int max_luminance);

using ShutdownFn = void();

inline constexpr const char* kCreate = "aji_rtx_create";
inline constexpr const char* kEvaluate = "aji_rtx_eval";
inline constexpr const char* kShutdown = "aji_rtx_shutdown";

}  // namespace ajp::vendor_hdr::rtx_video
