// La fenêtre de la stabilisation temporelle dans l'étage gpu (upscale_temporal.cpp) : les images de travail
// ingérées telles quelles (« brutes »), gardées par ordre de pts, et les sept qui entourent une image. Rien
// ici ne touche au GPU hors des tampons : le choix des images et le reflet aux bords se testent seuls
// (tests/unit/app/temporal_window_test.cpp).
//
// Bords : une époque qui commence n'a pas d'images d'avant, la fin du flux pas d'images d'après. Comme
// vs_temporalfix (gen_shifts), la fenêtre se reflète alors autour de l'image : l'image d'avant manquante k
// est l'image d'après k, et inversement ; sans l'une ni l'autre, la plus proche déjà prise, puis l'image
// elle-même.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "app/play/temporal_fix.h"
#include "core/frame.h"
#include "gpu/kernel_ops.h"
#include "gpu/tensor.h"

namespace kirari::app::play {

class TemporalWindow {
 public:
  // Ce qu'un tampon brut garde : l'image, sa réduction (une autre réduction, changée en direct, demande une
  // autre ingestion) et son débandage.
  struct Raw {
    SourceId id;
    std::uint32_t epoch = 0;
    std::int64_t pts_us = kNoPts;
    gpu::Reduction reduce;
    bool debanded = false;
    bool filled = false;
  };

  // capacity tampons : de l'image la plus ancienne qu'une fenêtre lit (la tête moins kTemporalFixRadius) à la
  // plus lointaine que l'étage lit en avance, plus kTemporalFixRadius.
  explicit TemporalWindow(std::size_t capacity = 0) : raws_(capacity) {}

  std::size_t capacity() const { return raws_.size(); }
  const Raw& raw(std::size_t index) const { return raws_[index]; }

  // Le tampon qui garde déjà cette image à cette réduction.
  std::optional<std::size_t> find(const DecodedFrame& frame, const gpu::Reduction& reduce) const;
  // Le tampon où ingérer cette image : un libre ou d'une autre réduction, sinon celui du plus petit pts. Les
  // tampons d'une autre époque sont oubliés (un seek). L'image y est notée, remplie.
  std::size_t claim(const DecodedFrame& frame, const gpu::Reduction& reduce, bool debanded);
  void set_debanded(std::size_t index, bool debanded) { raws_[index].debanded = debanded; }
  // Les kTemporalFixFrames tampons autour de center, dans l'ordre du temps (le milieu : center), pris parmi
  // ceux de la même époque et de la même réduction, reflétés aux bords. mirrored : vrai si un bord a servi.
  std::array<std::size_t, kTemporalFixFrames> window(std::size_t center, bool* mirrored = nullptr) const;
  // Tout oublier (la chaîne change de réduction, un refus).
  void clear();

 private:
  std::vector<Raw> raws_;
};

// Les tenseurs de la fenêtre, un par tampon : l'image brute et ses traits, encodés une fois à l'ingestion
// (prepare/onnx_cache.h) ; et ses compteurs pour le journal de fin.
struct TemporalBuffers {
  TemporalWindow window;
  std::vector<std::unique_ptr<gpu::TensorBuffer>> rgb;
  std::vector<std::unique_ptr<gpu::TensorBuffer>> features;
  // L'entrée k du moteur lit l'image de rang inputs[k] de la fenêtre, ou les traits du rang
  // inputs[k] - kTemporalFixFrames.
  std::array<std::size_t, 2 * kTemporalFixFrames> inputs{};
  std::vector<std::size_t> fresh;  // tampons ingérés pour la passe en cours, encodés juste avant elle
  std::uint64_t runs = 0;
  std::uint64_t mirrored = 0;  // passes dont la fenêtre s'est reflétée à un bord
  std::uint64_t raws = 0;      // ingestions brutes, chacune encodée une fois
};

}  // namespace kirari::app::play
