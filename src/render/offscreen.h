// OffscreenTarget : la cible hors écran de l'export (et des tests). Une pl_tex RGBA16 UNORM à
// la taille de sortie, dans l'espace d'un DisplayInfo (PQ BT.2020 pour l'export : grade
// compris, sans OSD), que Compositor::compose remplit comme une image de swapchain, puis
// qu'une copie asynchrone (pl_tex_download vers un pl_buf) rend lisible par l'hôte.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>

#include "core/resolve.h"
#include "core/types.h"

struct pl_frame;
struct pl_tex_t;

namespace ajp::render {

class RenderDevice;

struct OffscreenOptions {
  int width = 0;
  int height = 0;
  // L'espace de la cible : HDR actif en PQ pour l'export (pic du grade), sRGB sinon.
  core::DisplayInfo display;
};

class OffscreenTarget {
 public:
  // FieldOutOfRange : taille ; FeatureUnavailable : rgba16 non rendable ; RenderFailed :
  // texture ou tampon refusés.
  static std::expected<OffscreenTarget, Refusal> create(const RenderDevice& device,
                                                        const OffscreenOptions& options);

  OffscreenTarget(OffscreenTarget&&) noexcept;
  OffscreenTarget& operator=(OffscreenTarget&&) noexcept;
  OffscreenTarget(const OffscreenTarget&) = delete;
  OffscreenTarget& operator=(const OffscreenTarget&) = delete;
  ~OffscreenTarget();

  // La cible à passer à Compositor::compose ; son espace est fixé par options.display.
  const pl_frame& frame() const;
  const pl_tex_t* texture() const;
  int width() const;
  int height() const;

  // Lance la copie de l'image courante vers l'hôte, derrière le rendu déjà soumis.
  // SequenceOrder : une copie est encore en cours ; RenderFailed : copie refusée.
  std::expected<void, Refusal> start_readback();
  // Sans attente par défaut : true quand la copie lancée est arrivée.
  bool readback_ready(std::uint64_t timeout_ns = 0);
  // Les pixels de la dernière copie arrivée, RGBA16 UNORM, ligne après ligne (width × 4
  // valeurs par ligne) ; valables jusqu'au prochain start_readback(). SequenceOrder : aucune
  // copie arrivée.
  std::expected<std::span<const std::uint16_t>, Refusal> pixels();

 private:
  struct Impl;
  explicit OffscreenTarget(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::render
