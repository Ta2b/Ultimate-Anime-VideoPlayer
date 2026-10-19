// Outils des tests de render : appareil sans écran (RTX ou lavapipe), slots de contenu connu,
// lecture d'une cible hors écran, encodage PQ de référence.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <libplacebo/gpu.h>

#include "core/resolve.h"
#include "render/offscreen.h"
#include "render/render_device.h"

namespace ajp::render::test {

// Appareil sans écran, logiciel accepté ; nullopt avec la raison quand Vulkan manque.
std::optional<RenderDevice> make_device(std::string& reason);

// Écran SDR inconnu, et écran HDR10 en PQ au pic donné (mesuré par le système).
core::DisplayInfo sdr_display();
core::DisplayInfo pq_display(double peak_nits);

// Texture RGBA16 UNORM échantillonnable remplie de pixels (largeur × hauteur × 4 valeurs).
pl_tex make_texture(pl_gpu gpu, int width, int height, std::span<const std::uint16_t> pixels);
// Une couleur unie, en valeurs 0 à 1.
std::vector<std::uint16_t> solid(int width, int height, float r, float g, float b);

// Copie la cible vers l'hôte et attend son arrivée.
std::vector<std::uint16_t> read_back(OffscreenTarget& target);

// Valeur 0 à 1 du canal c du pixel (x, y) d'une image RGBA16.
float channel(std::span<const std::uint16_t> pixels, int width, int x, int y, int c);

// ST 2084 : signal PQ de nits.
double pq_encode(double nits);

}  // namespace ajp::render::test
