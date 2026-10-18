// Ring de sortie : les images finales que gpu écrit et que render (thread main) présente.
// Fait partie de core_types : bibliothèque standard seulement.
//
// Protocole d'un slot. Chaque slot a son sémaphore timeline, dont les valeurs alternent :
//   1. gpu attend la valeur de libération (0 au premier tour), écrit le slot, puis signale
//      ready_value, impaire, et envoie RingFrame au thread main ;
//   2. main attend ready_value côté GPU, échantillonne le slot, signale ready_value + 1 à la
//      fin de son rendu, puis renvoie SlotFree{slot, ready_value + 1, generation} à gpu ;
//   3. gpu attend cette valeur avant de réécrire le slot.
// Un seul écrivain à la fois par slot : la suite des valeurs croît donc toujours, comme un
// sémaphore timeline l'exige, sans qu'aucun côté n'attende sur l'hôte.
//
// Un ring recréé (taille ou format de sortie changés par un nouveau Plan) prend la
// génération suivante : un RingFrame ou un SlotFree d'une autre génération est ignoré.
#pragma once

#include <cstdint>

#include "core/frame.h"
#include "core/types.h"

namespace ajp {

// Rgba16Unorm par défaut ; Yuv444p16 est le repli qualifié (--ring-format).
enum class RingFormat : std::uint8_t { Rgba16Unorm, Yuv444p16 };

// Handle système exporté par Vulkan pour une autre API (CUDA, D3D11, autre VkDevice).
enum class HandleKind : std::uint8_t {
  None,
  OpaqueFd,     // Linux : descripteur (VK_EXTERNAL_*_HANDLE_TYPE_OPAQUE_FD_BIT)
  OpaqueWin32,  // Windows : HANDLE NT (VK_EXTERNAL_*_HANDLE_TYPE_OPAQUE_WIN32_BIT)
};

// Le créateur du ring possède l'original et le ferme à la destruction du ring ; un module qui
// l'importe en importe une copie (dup, DuplicateHandle), car l'import d'un fd en prend la
// propriété.
struct ExportedHandle {
  HandleKind kind = HandleKind::None;
  std::uintptr_t value = 0;  // OpaqueFd : le descripteur ; OpaqueWin32 : le HANDLE
};

// Un slot du ring, tel que render l'a alloué et exporté.
struct RingSlotDesc {
  std::uint32_t index = 0;
  std::uint32_t generation = 0;
  int width = 0;
  int height = 0;
  RingFormat format = RingFormat::Rgba16Unorm;
  std::uint64_t image = 0;          // VkImage, handle non dispatchable de 64 bits
  ExportedHandle memory;            // la mémoire de l'image
  std::uint64_t memory_size = 0;    // octets de l'allocation exportée
  std::uint64_t memory_offset = 0;  // début de l'image dans l'allocation
  std::uint64_t semaphore = 0;      // VkSemaphore timeline du slot
  ExportedHandle semaphore_handle;  // le même sémaphore, exporté
};

// Une image écrite dans le ring, envoyée par gpu au thread main.
struct RingFrame {
  std::uint32_t slot = 0;
  std::uint32_t generation = 0;
  std::int64_t pts_us = kNoPts;  // horodatage de présentation, microsecondes
  // La paire d'où vient l'image : left == right pour une source montrée telle quelle.
  SourceId left;
  SourceId right;
  // Instant dans la paire, t = t_num / t_den dans [0, 1[ : 0 est la source left.
  std::int64_t t_num = 0;
  std::int64_t t_den = 1;
  bool repeated = false;          // REPEATED : l'image d'un instant déjà calculé, rien de recalculé
  bool scene_cut = false;         // coupe de scène entre left et right : aucune interpolation
  std::uint32_t epoch = 0;        // une image d'une époque passée est jetée par main
  std::uint64_t ready_value = 0;  // valeur du sémaphore du slot à attendre avant lecture
};

}  // namespace ajp
