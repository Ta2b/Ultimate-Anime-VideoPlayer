#include "vendor_hdr/true_hdr.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <utility>

namespace ajp::vendor_hdr {
namespace {

Done check_param(std::string_view field, int value, ParamRange range) {
  if (range.contains(value)) return {};
  return refuse(RefusalCode::FieldOutOfRange,
                std::format("truehdr.{} = {}, hors de {}-{}", field, value, range.lo, range.hi));
}

std::string_view matrix_name(ColorMatrix matrix) {
  switch (matrix) {
    case ColorMatrix::Bt601:
      return "BT.601";
    case ColorMatrix::Bt709:
      return "BT.709";
    case ColorMatrix::Bt2020Ncl:
      return "BT.2020";
    case ColorMatrix::Rgb:
      return "RGB";
  }
  return "inconnue";
}

bool sdr_transfer(Transfer transfer) {
  return transfer == Transfer::Bt1886 || transfer == Transfer::Srgb;
}

bool sdr_primaries(Primaries primaries) {
  return primaries == Primaries::Bt709 || primaries == Primaries::Bt601_625 ||
         primaries == Primaries::Bt601_525;
}

std::string_view range_name(ColorRange range) {
  return range == ColorRange::Full ? "pleine" : "limitée";
}

// Octets qu'une image occupe, de sa première ligne à la fin des pixels de la dernière. Les
// bornes de check_image (côté ≤ 16384, pas < 2^31) le gardent sous 2^46.
std::uintptr_t extent(const TrueHdrImage& image) {
  const auto rows = static_cast<std::uintptr_t>(image.height - 1);
  return (rows * image.pitch) + (static_cast<std::uintptr_t>(image.width) * 4);
}

bool overlap(const TrueHdrImage& a, const TrueHdrImage& b) {
  return a.device < b.device + extent(b) && b.device < a.device + extent(a);
}

Done check_image(std::string_view role, const TrueHdrImage& image, const TrueHdrConfig& config,
                 TrueHdrPixel expected) {
  if (image.device == 0) return refuse(RefusalCode::InputMismatch, std::format("{} : tampon nul", role));
  if (image.width != config.width || image.height != config.height) {
    return refuse(RefusalCode::InputMismatch, std::format("{} : {}x{} au lieu de {}x{}", role, image.width,
                                                          image.height, config.width, config.height));
  }
  if (image.pixel != expected) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : {} au lieu de {}", role, name(image.pixel), name(expected)));
  }
  const auto row = static_cast<std::size_t>(image.width) * 4;
  if (image.pitch < row || image.pitch > limits::kMaxPitch || image.pitch % 4 != 0) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : pas de {} octets, attendu un multiple de 4 entre {} et {}", role,
                              image.pitch, row, limits::kMaxPitch));
  }
  if (image.device > UINTPTR_MAX - extent(image)) {
    return refuse(RefusalCode::InputMismatch, std::format("{} : tampon hors de l'espace d'adresses", role));
  }
  return {};
}

// Enveloppe un effet constructeur dans les contrats de flux et d'entrée de true_hdr.h.
class BracketedTrueHdr final : public TrueHdr {
 public:
  BracketedTrueHdr(const TrueHdrConfig& config, std::unique_ptr<VendorEffect> effect)
      : config_(config), effect_(std::move(effect)) {}

  const TrueHdrConfig& config() const override { return config_; }
  TrueHdrPixel input_pixel() const override { return effect_->input_pixel(); }

  Done set_params(const TrueHdrParams& params) override {
    if (auto valid = validate(params); !valid) return std::unexpected(valid.error());
    if (Done applied = effect_->apply(params); !applied) return applied;
    config_.params = params;
    return {};
  }

  Done process(const TrueHdrInput& in, const TrueHdrImage& out, GpuStream& stream) override {
    if (Done valid = check_input(in); !valid) return valid;
    if (Done valid = check_image("sortie", out, config_, TrueHdrPixel::Rgb10A2); !valid) return valid;
    if (overlap(in.image, out)) {
      return refuse(RefusalCode::InputMismatch, "entrée et sortie se recouvrent dans le même tampon");
    }
    if (Done handed = stream.hand_to_legacy(); !handed) return handed;
    const auto start = std::chrono::steady_clock::now();
    const Done ran = effect_->run_on_legacy(in.image, out);
    time_.host +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
    ++time_.calls;
    // Toujours rendre la main au flux de gpu : le flux 0 peut encore écrire dans la sortie.
    const Done taken = stream.take_from_legacy();
    return ran ? taken : ran;
  }

  VendorTime vendor_time() const override { return time_; }

 private:
  Done check_input(const TrueHdrInput& in) const {
    if (Done valid = check_image("entrée", in.image, config_, effect_->input_pixel()); !valid) return valid;
    if (in.prepared.matrix != config_.source.matrix) {
      return refuse(RefusalCode::InputMismatch,
                    std::format("entrée convertie avec la matrice {}, la source est en {} (AH-371)",
                                matrix_name(in.prepared.matrix), matrix_name(config_.source.matrix)));
    }
    if (in.prepared.range != config_.source.range) {
      return refuse(RefusalCode::InputMismatch,
                    std::format("entrée convertie en plage {}, la source est en plage {} (AH-371)",
                                range_name(in.prepared.range), range_name(config_.source.range)));
    }
    if (in.image.pixel == TrueHdrPixel::Rgba8 && !in.prepared.dithered) {
      return refuse(RefusalCode::InputMismatch, "entrée 8 bits quantifiée sans tramage (AH-392)");
    }
    return {};
  }

  TrueHdrConfig config_;
  std::unique_ptr<VendorEffect> effect_;
  VendorTime time_;
};

}  // namespace

std::expected<TrueHdrParams, Refusal> validate(const TrueHdrParams& params) {
  for (const Done& field : {check_param("contrast", params.contrast, limits::kContrast),
                            check_param("saturation", params.saturation, limits::kSaturation),
                            check_param("middle_gray", params.middle_gray, limits::kMiddleGray),
                            check_param("peak", params.peak_nits, limits::kPeakNits)}) {
    if (!field) return std::unexpected(field.error());
  }
  return params;
}

int peak_from_display(std::optional<double> detected_nits) {
  if (!detected_nits || !std::isfinite(*detected_nits) || *detected_nits <= 0.0) {
    return limits::kPeakNits.fallback;
  }
  const double bounded = std::clamp(*detected_nits, static_cast<double>(limits::kPeakNits.lo),
                                    static_cast<double>(limits::kPeakNits.hi));
  return static_cast<int>(std::lround(bounded));
}

std::string_view name(TrueHdrPixel pixel) {
  switch (pixel) {
    case TrueHdrPixel::Rgba8:
      return "RGBA 8 bits";
    case TrueHdrPixel::Rgb10A2:
      return "RGB10A2";
  }
  return "inconnu";
}

Done check(const TrueHdrConfig& config) {
  if (!limits::kSide.contains(config.width) || !limits::kSide.contains(config.height)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("truehdr : {}x{}, chaque côté doit tenir dans {}-{}", config.width,
                              config.height, limits::kSide.lo, limits::kSide.hi));
  }
  const ColorInfo& source = config.source;
  if (!sdr_transfer(source.transfer) || !sdr_primaries(source.primaries) ||
      source.matrix == ColorMatrix::Bt2020Ncl) {
    // Une source déjà HDR n'est jamais convertie deux fois (aji_pipeline.cpp:144-147 du moteur).
    return refuse(RefusalCode::HdrUnavailable, "TrueHDR n'accepte qu'une source SDR (BT.709 ou BT.601)");
  }
  if (auto valid = validate(config.params); !valid) return std::unexpected(valid.error());
  return {};
}

std::expected<std::unique_ptr<TrueHdr>, Refusal> make_true_hdr(const TrueHdrConfig& config,
                                                               std::unique_ptr<VendorEffect> effect) {
  if (!effect) return refuse(RefusalCode::LibraryUnavailable, "aucun effet TrueHDR à envelopper");
  if (Done valid = check(config); !valid) return std::unexpected(valid.error());
  if (Done applied = effect->apply(config.params); !applied) return std::unexpected(applied.error());
  return std::make_unique<BracketedTrueHdr>(config, std::move(effect));
}

}  // namespace ajp::vendor_hdr
