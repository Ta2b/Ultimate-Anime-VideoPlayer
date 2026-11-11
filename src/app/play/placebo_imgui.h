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
  //
  // serial non nul (UiFrame::serial) : l'interface de ce numéro est gardée dans une texture (B32-05). Tant
  // que ni elle ni la cible (taille, couleurs) ne changent, elle se mélange d'un seul passage, sur le
  // rectangle qu'elle couvre, au lieu d'un passage par commande ImGui. Le mélange « over » étant associatif,
  // le résultat est celui du dessin direct (texture 16 bits par composante). 0 : dessin direct.
  std::expected<void, Refusal> draw(const ImDrawData& data, const pl_frame& target, float white_nits,
                                    std::uint64_t serial = 0);
  // Rend la texture gardée (interface cachée) : jusqu'à 66 Mo de mémoire vidéo en 4K.
  void drop_cache();

  // Détruit les textures que cette couche a créées pour les textures d'ImGui et les marque
  // détruites : avant de détruire la couche si le contexte ImGui lui survit.
  void release_textures(ImGuiPlatformIO& platform);

 private:
  std::expected<void, Refusal> update_texture(ImTextureData& texture);
  // ImTextureID d'une texture de cette couche : son rang dans textures_ plus un (0 : aucune).
  pl_tex texture_of(std::uint64_t id) const;
  std::uint64_t adopt(pl_tex tex);
  void destroy(ImTextureData& texture);

  // Dessine les listes de data dans `into` (la cible, ou la texture gardée, de même taille) ; bounds reçoit
  // l'union des rectangles de découpe, en coordonnées de la texture.
  std::expected<void, Refusal> draw_lists(const ImDrawData& data, const pl_frame& target, pl_tex into,
                                          float white_nits, pl_rect2d* bounds);
  std::expected<void, Refusal> blend_cache(pl_tex target_tex);
  bool cache_matches(std::uint64_t serial, const pl_frame& target, float white_nits) const;

  pl_gpu gpu_;
  pl_dispatch dispatch_ = nullptr;
  std::vector<pl_tex> textures_;  // créées par cette couche, détruites avec elle ; nullptr : place libre
  // B32-05 : l'interface gardée, en alpha prémultiplié, dans l'encodage de la cible.
  pl_tex cache_ = nullptr;
  std::uint64_t cache_serial_ = 0;  // 0 : rien de gardé
  bool cache_flipped_ = false;
  float cache_white_ = 0.0F;
  pl_color_space cache_color_{};
  pl_color_repr cache_repr_{};
  pl_rect2d cache_bounds_{};
};

}  // namespace ajp::app::play
