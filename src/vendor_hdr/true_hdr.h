// TrueHdr (RTX HDR) : une source SDR devient HDR10 (PQ, BT.2020) par le SDK NVIDIA que
// l'utilisateur a importé (RTX Video SDK 1.1 sous Windows, Maxine VFX TrueHDR 1.3 sous Linux).
// Sans thread : gpu appelle process() depuis le sien, entre deux noyaux de son graphe.
//
// Contrat de flux (plan, « Le cœur résout, le GPU exécute » ; AH-19). Le flux de gpu est non
// bloquant : rien ne l'ordonne avec le flux 0 hérité (legacy). TrueHDR tourne sur ce flux 0,
// et c'est le seul travail admis dessus. process() l'encadre d'événements, sans attente hôte :
//   1. GpuStream::hand_to_legacy()   le flux 0 attend l'entrée produite sur le flux de gpu ;
//   2. appel du SDK sur le flux 0 ;
//   3. GpuStream::take_from_legacy() le flux de gpu attend la sortie écrite par le flux 0.
// L'étape 3 a lieu même si le SDK échoue : aucun tampon n'est rendu tant que le flux 0 peut
// encore y écrire. L'appel du SDK bloque l'hôte (NvVFX_Run synchrone) : son temps est mesuré
// (vendor_time()) pour être compté à part des host_waits de gpu, jusqu'à l'ADR de AH-19.
//
// Contrat d'entrée (AH-371, AH-392). gpu convertit la source en RGB avec la vraie matrice et la
// vraie plage de la source, jamais une BT.709 limitée supposée ; une entrée 8 bits (Maxine) est
// tramée avant d'être quantifiée, sinon les quasi-noirs se crantent avant même le pied du
// grade. L'entrée déclare les trois (InputPreparation) et process() refuse ce qui ne les
// respecte pas.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string_view>

#include "core/frame.h"
#include "core/types.h"
#include "vendor_hdr/refusal.h"

namespace ajp::vendor_hdr {

// Plage incluse d'un réglage entier, avec sa valeur par défaut.
struct ParamRange {
  int lo = 0;
  int hi = 0;
  int fallback = 0;
  constexpr bool contains(int value) const { return lo <= value && value <= hi; }
};

// Plages du plan, identiques à celles du SDK Maxine 1.3.0.0 (nvVFXTrueHDR.h:17-20) : sur la RTX,
// une valeur hors plage y est refusée (NVCV_ERR_PARAMETER) et l'ancienne gardée. Seul le pic
// par défaut diffère : 650 nits dans le SDK, 1000 ici (plan). Le lecteur passe toujours chaque
// réglage avant le chargement du modèle et ne dépend jamais d'un défaut du SDK (le faux SDK des
// tests refuse NvVFX_Load sinon ; real_true_hdr_test.cpp vérifie le pic sur la RTX).
namespace limits {
inline constexpr ParamRange kContrast{0, 200, 100};
inline constexpr ParamRange kSaturation{0, 200, 100};
inline constexpr ParamRange kMiddleGray{10, 100, 50};
inline constexpr ParamRange kPeakNits{400, 2000, 1000};
// Côté d'image accepté ; TrueHDR ne change pas la taille.
inline constexpr ParamRange kSide{2, 16384, 0};
// Pas de ligne le plus grand : NvCVImage le porte dans un int.
inline constexpr std::size_t kMaxPitch = 0x7FFFFFFF;
}  // namespace limits

// Réglages TrueHDR, communs au SDK et à son émulation sur la route native.
struct TrueHdrParams {
  int contrast = limits::kContrast.fallback;
  int saturation = limits::kSaturation.fallback;
  int middle_gray = limits::kMiddleGray.fallback;
  int peak_nits = limits::kPeakNits.fallback;
  // Passe DL-Debander de TrueHDR (DebandingOff de Maxine 1.3, nvVFXTrueHDR.h:21) : active par
  // défaut, qualité d'abord. La couper allège TrueHDR mais recharge le modèle au changement de
  // ce réglage, jamais à chaque image. RTX Video ne sait pas la
  // couper : il refuse false.
  bool debanding = true;
  friend bool operator==(const TrueHdrParams&, const TrueHdrParams&) = default;
};

// Refuse un réglage hors plage, en le nommant ; ne corrige jamais.
std::expected<TrueHdrParams, Refusal> validate(const TrueHdrParams& params);

// Pic à donner à TrueHDR depuis la luminance détectée de l'écran : bornée à 400-2000 nits,
// 1000 quand elle est inconnue (absente, nulle, négative ou non finie). app passe
// core::DisplayInfo::peak_nits quand sa provenance n'est pas Default : la crête vue du contenu,
// soit sous KWin la marge HDR de SDL3 (relative au blanc de référence de KWin) × 203 nits,
// 1000 / 265 × 203 = 766 nits sur l'écran de référence (~/ajp-reports/display.md).
int peak_from_display(std::optional<double> detected_nits);

// Disposition d'un tampon TrueHDR : entrelacé, 4 octets par pixel.
enum class TrueHdrPixel : std::uint8_t {
  Rgba8,    // R, G, B, A sur 8 bits : entrée de Maxine
  Rgb10A2,  // R bits 0-9, G 10-19, B 20-29, A 30-31 : sortie des deux SDK, entrée de RTX Video
};

std::string_view name(TrueHdrPixel pixel);

// Tampon linéaire en mémoire CUDA, sans propriété.
struct TrueHdrImage {
  std::uintptr_t device = 0;  // CUdeviceptr
  std::size_t pitch = 0;      // octets par ligne : multiple de 4, au moins 4 × largeur
  int width = 0;
  int height = 0;
  TrueHdrPixel pixel = TrueHdrPixel::Rgba8;
};

// Comment gpu a produit l'entrée RGB : déclaré par lui, vérifié par process().
struct InputPreparation {
  ColorMatrix matrix = ColorMatrix::Bt709;  // matrice utilisée pour YUV vers RGB (AH-371)
  ColorRange range = ColorRange::Limited;   // plage lue à la conversion (AH-371)
  bool dithered = false;                    // tramage avant la quantification en 8 bits (AH-392)
};

struct TrueHdrInput {
  TrueHdrImage image;
  InputPreparation prepared;
};

// Ce pour quoi une instance TrueHdr est créée ; la taille ne change plus ensuite.
struct TrueHdrConfig {
  int width = 0;
  int height = 0;
  ColorInfo source;  // couleur de la source décodée : SDR seulement
  TrueHdrParams params;
  // CUcontext du thread gpu, courant pendant les appels. Exigé par RTX Video, qui le reçoit à
  // la création ; Maxine prend le contexte courant.
  std::uintptr_t cuda_context = 0;
};

// Temps hôte passé dans les appels du SDK depuis la création, à compter à part des host_waits.
struct VendorTime {
  std::uint64_t calls = 0;
  std::chrono::nanoseconds host{0};
};

// Refuse une taille hors plage, une source déjà HDR ou des réglages hors plage.
Done check(const TrueHdrConfig& config);

// Flux de calcul du thread gpu, vu par TrueHDR. gpu l'implémente autour de son CUstream non
// bloquant et de deux CUevent créés une fois (CU_EVENT_DISABLE_TIMING) :
//   hand_to_legacy()   cuEventRecord(entrée, flux) puis cuStreamWaitEvent(flux 0, entrée)
//   take_from_legacy() cuEventRecord(sortie, flux 0) puis cuStreamWaitEvent(flux, sortie)
class GpuStream {
 public:
  virtual ~GpuStream() = default;
  virtual Done hand_to_legacy() = 0;
  virtual Done take_from_legacy() = 0;
};

class TrueHdr {
 public:
  virtual ~TrueHdr() = default;
  virtual const TrueHdrConfig& config() const = 0;
  // Disposition attendue en entrée ; la sortie est toujours Rgb10A2.
  virtual TrueHdrPixel input_pixel() const = 0;
  // Refuse un réglage hors plage sans rien changer ; une erreur du SDK est rendue telle quelle.
  virtual Done set_params(const TrueHdrParams& params) = 0;
  virtual Done process(const TrueHdrInput& in, const TrueHdrImage& out, GpuStream& stream) = 0;
  // Appels du SDK faits par process(), réussis ou non, et leur temps hôte cumulé.
  virtual VendorTime vendor_time() const = 0;
};

// Ce que chaque SDK fournit : un effet déjà créé pour la taille du TrueHdrConfig, qui ne
// soumet du travail que sur le flux 0. Les chargeurs (maxine_vfx, rtx_video) l'implémentent.
class VendorEffect {
 public:
  virtual ~VendorEffect() = default;
  virtual TrueHdrPixel input_pixel() const = 0;
  virtual Done apply(const TrueHdrParams& params) = 0;
  virtual Done run_on_legacy(const TrueHdrImage& in, const TrueHdrImage& out) = 0;
};

// Vérifie la configuration, applique ses réglages à l'effet et l'enveloppe dans les contrats
// de flux et d'entrée décrits plus haut.
std::expected<std::unique_ptr<TrueHdr>, Refusal> make_true_hdr(const TrueHdrConfig& config,
                                                               std::unique_ptr<VendorEffect> effect);

}  // namespace ajp::vendor_hdr
