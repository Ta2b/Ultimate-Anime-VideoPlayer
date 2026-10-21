// L'étage TrueHDR tel que gpu l'exécute, avec son repli : si TrueHDR ne s'ouvre pas ou échoue,
// la route native (libplacebo en PQ, puis le grade) prend le relais pour le reste de la lecture,
// et l'OSD l'annonce au moins 3 s (plan, « HDR » et « Interface »). Ce module expose l'événement
// de bascule ; render et app l'affichent.
#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>

#include "core/types.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::vendor_hdr {

// Durée minimale d'affichage d'une bascule à l'OSD.
inline constexpr std::chrono::milliseconds kFallbackNotice{3000};

enum class HdrPath : std::uint8_t {
  TrueHdr,  // la sortie porte l'image HDR10 de TrueHDR
  Native,   // la sortie n'est pas écrite : render convertit l'entrée par la route native
};

// Bascule à annoncer : une fois par étage, au moment où elle a lieu.
struct TrueHdrFallback {
  Refusal cause;
  std::chrono::milliseconds notice = kFallbackNotice;
};

class TrueHdrStage {
 public:
  // opened : le résultat de open_true_hdr(). Un refus bascule tout de suite sur la route native.
  explicit TrueHdrStage(std::expected<std::unique_ptr<TrueHdr>, Refusal> opened);

  HdrPath path() const { return sdk_ ? HdrPath::TrueHdr : HdrPath::Native; }

  // Route prise pour cette image. Une erreur de TrueHDR ferme le SDK et bascule pour de bon.
  HdrPath process(const TrueHdrInput& in, const TrueHdrImage& out, GpuStream& stream);

  // Un réglage hors plage est refusé sans rien changer ; une erreur du SDK bascule.
  std::expected<HdrPath, Refusal> set_params(const TrueHdrParams& params);

  // La bascule survenue depuis le dernier appel, une seule fois ; rien sinon.
  std::optional<TrueHdrFallback> take_fallback();

  // Temps hôte des appels du SDK depuis l'ouverture, gardé après une bascule.
  VendorTime vendor_time() const { return sdk_ ? sdk_->vendor_time() : closed_time_; }

 private:
  void fall_back(Refusal cause);

  std::unique_ptr<TrueHdr> sdk_;
  std::optional<TrueHdrFallback> pending_;
  VendorTime closed_time_;
};

}  // namespace ajp::vendor_hdr
