// Familles de GPU et liste d'autorisation des unités matricielles. La famille se déduit des
// identifiants PCI (constructeur, appareil), jamais du nom ; le pilote se lit dans
// VkPhysicalDeviceDriverProperties. Les unités matricielles ne servent que sur une famille
// et un pilote qualifiés : un pilote qui se trompe ne casse pas l'image.
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace kirari::gpu::vk {

enum class GpuFamily : std::uint8_t {
  Unknown,
  Software,  // lavapipe, SwiftShader : processeur
  NvidiaTuring,
  NvidiaAmpere,
  NvidiaAda,
  NvidiaBlackwell,
  AmdBeforeRdna3,
  AmdRdna3,
  AmdRdna4,
  IntelXeLp,   // Tiger Lake à Raptor Lake (UHD 770, Iris Xe), DG1
  IntelXeLpg,  // Meteor Lake et Arrow Lake (Arc intégré des Core Ultra 100 et 200)
  IntelAlchemist,
  IntelXe2,
  Apple,  // puces M d'Apple sous MoltenVK
};

std::string_view name(GpuFamily family);

// cpu : VkPhysicalDeviceType vaut VK_PHYSICAL_DEVICE_TYPE_CPU. device_name : un appareil AMD hors
// des plages connues se reconnaît à la puce que RADV nomme (« (RADV NAVI48) », « (RADV GFX1150) »).
GpuFamily detect_family(std::uint32_t vendor_id, std::uint32_t device_id, bool cpu,
                        std::string_view device_name = {});

// La famille revue d'après les tuiles que le pilote annonce, pour un appareil AMD hors des plages PCI que
// RADV ne nomme pas (pilote AMD de Windows, puce plus récente que les tables) : detect_family le range avant
// RDNA3, sans unités matricielles. Des tuiles FP8 E4M3 n'existent qu'à partir de RDNA 4 : il y est rangé. La
// seule tuile float16 16 × 16 × 16 en fait une famille inconnue : les unités matricielles sont proposées à
// l'autotuner, qui les vérifie et les mesure (matrix_units_unlisted), au lieu d'être écartées. Une puce que
// RADV nomme garde sa famille.
GpuFamily family_from_tiles(GpuFamily family, std::string_view device_name, bool f16_tile, bool e4m3_tile);

// Pilotes connus, depuis VkPhysicalDeviceDriverProperties::driverName.
enum class DriverKind : std::uint8_t {
  Other,
  NvidiaProprietary,
  Radv,
  AmdProprietary,
  IntelMesa,
  IntelWindows
};
DriverKind driver_kind(std::string_view driver_name);

// Unités matricielles (P2.46) : décidées sur les tuiles que le pilote annonce
// (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR, device_profile.cpp) et sur la sortie de chaque couche,
// vérifiée par l'autotuner ; une seule liste d'exclusion écarte les appareils et pilotes connus pour les
// annoncer à tort (AMD avant RDNA3, Intel sans XMX, processeur, NVIDIA hors pilote propriétaire...).
// matrix_units_excluded en donne la raison. Intel Alchemist et Xe2 ne sont pas exclus. Les Xe2 annoncent sous
// ANV une tuile 8×16×16 : la variante coopmat_m8 (net_conv_m8.h, Target::matrix_m8), que l'autotuner seul
// retient, en sous-groupes de 16. Les Arc A (DG2) y annoncent 8×8×16 (N = 8), qu'aucun noyau ne tuile encore.
std::optional<std::string_view> matrix_units_excluded(GpuFamily family, DriverKind driver);
bool matrix_units_allowed(GpuFamily family, DriverKind driver);

// Les tuiles FP8 (coopmat_fp8) : jamais sur une RTX 50 (Blackwell) tant qu'aucune mesure ne les justifie
// (intention du produit : pas de FP8 sans mesure) ; le pilote les annonce pourtant. Pas non plus sur une
// famille inconnue, qui peut être une RTX 50 d'un identifiant hors des tables (C45-05). À rouvrir avec une
// mesure.
bool matrix_f8_allowed(GpuFamily family);

// Les graphes INT8 (variante INT8 du SPAN, cran Performance) : seulement là où les tuiles INT8 vont deux fois
// plus vite que les tuiles float16, d'après les fiches des constructeurs : RDNA 4 et GeForce Turing à Ada.
// Les RDNA 3 font l'INT8 au rythme du float16 (122,8 TOPS INT8 comme TFLOPS FP16 sur RX 7900 XTX) : l'INT8 y
// perdrait de la qualité sans rien gagner. Pas de RTX 50 ni de famille inconnue, comme pour le FP8. Les
// tuiles INT8 restent proposées à l'autotuner partout où le pilote les annonce (Target::matrix_i8).
bool int8_graphs_allowed(GpuFamily family);

// Permises sur une famille inconnue (puce hors des plages PCI) : proposées et vérifiées par l'autotuner,
// jamais en tête des candidats.
bool matrix_units_unlisted(GpuFamily family, DriverKind driver);

}  // namespace kirari::gpu::vk
