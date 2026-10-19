// Dessin de l'interface Dear ImGui par libplacebo (pl_dispatch_vertex), en dernier, par-dessus
// l'image composée. Le contexte ImGui et l'interface appartiennent à ui.cpp : cette couche ne fait
// que le « renderer backend », textures comprises (ImGuiBackendFlags_RendererHasTextures, 1.92).
// Reprise de render/imgui_layer (pc/render), le jour où render remplace l'écran provisoire.
//
// Les couleurs d'ImGui sont du sRGB 8 bits en alpha droit ; dans une cible HDR, leur blanc vaut
// white_nits. Le mélange se fait dans l'encodage de la cible, comme le fait ImGui partout.
#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include <libplacebo/dispatch.h>
#include <libplacebo/renderer.h>

#include "core/types.h"

struct ImDrawData;
struct ImGuiIO;
struct ImGuiPlatformIO;
struct ImTextureData;

namespace ajp::app::play {

class ImGuiLayer {
 public:
  ImGuiLayer(pl_gpu gpu, pl_log log);
  ImGuiLayer(const ImGuiLayer&) = delete;
  ImGuiLayer& operator=(const ImGuiLayer&) = delete;
  ~ImGuiLayer();

  // Annonce le backend à ImGui (nom, RendererHasTextures, VtxOffset) ; ui.cpp l'appelle une fois.
  static void configure(ImGuiIO& io);

  // Honore les demandes de textures de data, puis dessine ses listes sur target.planes[0], avec
  // les rectangles de découpe. FormatUnsupported : cible non mélangeable ou format de sommet
  // absent ; VendorCallFailed : texture ou passe refusées ; InputMismatch : DisplayPos ou échelle
  // autres que (0, 0) et 1 (l'interface travaille en pixels de la sortie).
  std::expected<void, Refusal> draw(const ImDrawData& data, const pl_frame& target, float white_nits);

  // Détruit les textures que cette couche a créées pour les textures d'ImGui et les marque
  // détruites : avant de détruire la couche si le contexte ImGui lui survit.
  void release_textures(ImGuiPlatformIO& platform);

 private:
  std::expected<void, Refusal> update_texture(ImTextureData& texture);
  // ImTextureID d'une texture de cette couche : son rang dans textures_ plus un (0 : aucune).
  pl_tex texture_of(std::uint64_t id) const;
  std::uint64_t adopt(pl_tex tex);
  void destroy(ImTextureData& texture);

  pl_gpu gpu_;
  pl_dispatch dispatch_ = nullptr;
  std::vector<pl_tex> textures_;  // créées par cette couche, détruites avec elle ; nullptr : place libre
};

}  // namespace ajp::app::play
