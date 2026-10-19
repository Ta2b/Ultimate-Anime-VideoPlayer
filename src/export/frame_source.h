// Entrée de l'export : les images finales de render, hors écran, avec leur pts.
//
// render compose une image RGBA16 PQ BT.2020 à la taille de sortie, grade compris, sans OSD
// (plan : « Export : le même Plan, sans horloge »). L'export la lit par ExportFrameSource, à la
// place de present. Deux voies :
//   - hôte (HostFrame) : render relit l'image en mémoire hôte ; l'export la convertit par
//     libswscale vers le format de l'encodeur. C'est la voie des encodeurs logiciels et de la
//     voie de secours ; une copie hôte est permise hors temps réel.
//   - appareil (DeviceFrame) : le noyau d'écriture du backend convertit l'image en P010 sur le
//     GPU de rendu, et l'encodeur matériel la lit sans copie hôte. Cette voie est décrite ici
//     et branchée par le backend ; run_export la refuse tant qu'il ne l'a pas fait.
// Aucune dépendance hors core_types : render implémente cette interface sans tirer FFmpeg.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <variant>

#include "core/frame.h"
#include "core/types.h"

namespace ajp::exporter {

// Une image RGBA16 en mémoire hôte : quatre canaux de 16 bits non signés par pixel, R, G, B
// et A dans cet ordre, en boutisme natif. R, G et B portent le signal PQ (SMPTE ST 2084) en
// primaires BT.2020, pleine échelle (0 à 65535) ; A est ignoré.
struct HostFrame {
  int width = 0;
  int height = 0;
  std::size_t stride = 0;               // octets entre deux lignes, au moins width * 8
  std::span<const std::uint16_t> rgba;  // height lignes de stride octets
  std::int64_t pts_us = kNoPts;         // horodatage de présentation, microsecondes
};

// Mesure de lumière faite par le backend sur la voie appareil (MaxCLL et MaxFALL).
struct FrameLight {
  double max_nits = 0.0;      // plus grande composante R, G ou B de l'image, en cd/m²
  double average_nits = 0.0;  // moyenne sur l'image de la plus grande composante de chaque pixel
};

// Une image P010 sur le GPU de rendu, écrite par le noyau d'écriture du backend.
struct DeviceFrame {
  int width = 0;
  int height = 0;
  std::uint64_t image = 0;        // VkImage G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
  std::uint64_t semaphore = 0;    // VkSemaphore timeline à attendre avant lecture
  std::uint64_t ready_value = 0;  // valeur de ce sémaphore
  FrameLight light;               // mesurée sur le GPU : l'export ne relit pas l'image
  std::int64_t pts_us = kNoPts;
};

using SourceFrame = std::variant<HostFrame, DeviceFrame>;

// Ce que l'export lit. next() rend l'image suivante, std::nullopt à la fin du flux, ou un
// refus (Cancelled quand l'arrêt est demandé). L'image rendue reste valable jusqu'à l'appel
// suivant de next() ou jusqu'à la destruction de la source.
class ExportFrameSource {
 public:
  virtual ~ExportFrameSource() = default;

  virtual std::expected<std::optional<SourceFrame>, Refusal> next() = 0;

  // Nombre d'images attendues, s'il est connu : sert à la progression et au temps restant.
  virtual std::optional<std::int64_t> total_frames() const { return std::nullopt; }

 protected:
  ExportFrameSource() = default;
  ExportFrameSource(const ExportFrameSource&) = default;
  ExportFrameSource& operator=(const ExportFrameSource&) = default;
};

}  // namespace ajp::exporter
