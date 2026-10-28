// Programme d'un réseau : un graphe gpu_ir abaissé vers les noyaux GLSL du moteur Vulkan.
//
// Chaque nœud de l'IR reçoit un choix de noyau (KernelChoice : variante et réglages, choisis par
// défaut ou par l'autotuner), puis devient un ou plusieurs dispatchs : shader embarqué,
// constantes de spécialisation, tampons liés, bloc push_constant et nombre de groupes. Les poids
// rangés pour le noyau choisi forment des blobs, envoyés une fois au chargement. L'abaissement
// est déterministe : la même IR et les mêmes choix donnent le même programme, ce que load()
// vérifie contre l'artefact.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu/ir/memory_plan.h"

namespace ajp::gpu::vk::net {

// Ce qu'un dispatch lie à un binding : un tenseur du graphe (entrée, sortie ou activation de
// l'arène), un blob du programme, ou un petit tampon factice pour un binding que le shader
// n'utilise pas.
enum class BindingKind : std::uint8_t { Tensor, Blob, Dummy };

struct Binding {
  BindingKind kind = BindingKind::Dummy;
  std::uint32_t index = 0;  // identifiant de tenseur ou rang du blob
  friend bool operator==(const Binding&, const Binding&) = default;
};

struct Dispatch {
  std::uint32_t node = 0;  // nœud de l'IR d'où il vient
  std::string shader;      // nom embarqué, variante comprise (« net_conv_f16 »)
  std::vector<std::uint32_t> constants;
  std::vector<Binding> bindings;
  std::vector<std::uint32_t> push;  // mots du bloc push_constant
  std::array<std::uint32_t, 3> groups{1, 1, 1};
  std::uint32_t subgroup = 0;  // taille de sous-groupe exigée ; 0 : celle du pilote
  friend bool operator==(const Dispatch&, const Dispatch&) = default;
};

// Noyau choisi pour un nœud : la variante (« direct », « direct_f16p », « coopmat »…) et ses
// réglages (pour Conv directe : quads, largeur du groupe ; pour coopmat : tuiles).
struct KernelChoice {
  std::string variant;
  std::vector<std::uint32_t> tuning;
  friend bool operator==(const KernelChoice&, const KernelChoice&) = default;
};

struct Program {
  std::vector<Dispatch> dispatches;
  std::vector<std::vector<std::byte>> blobs;  // poids rangés, biais, constantes lues par les noyaux
  std::vector<ir::Alias> aliases;             // un par tenseur (net_alias.h), ou vide : aucun
};

// Ce que l'appareil permet aux noyaux.
struct Target {
  bool f16_storage = false;   // tenseurs float16 (variantes « _f16 »)
  bool f16_math = false;      // calcul en fp16 (« _f16p »)
  bool matrix_units = false;  // matrices coopératives proposées (famille et pilote, voir family.h)
  // Et en tête des candidats, choix par défaut sans autotuner : famille de la liste
  // d'autorisation ; faux pour un appareil hors liste (matrix_units_unlisted).
  bool matrix_first = false;
  // Tuiles 16 × 16 × 16 à sommes float16 aussi annoncées (variante coopmat_acc16, deux fois plus
  // rapide sur GeForce, retenue seulement si sa sortie concorde).
  bool matrix_f16_acc = false;
  std::uint32_t subgroup_size = 0;
  // Tailles de sous-groupe qu'un dispatch peut exiger ; 0 : aucune (le pilote choisit).
  std::uint32_t subgroup_min = 0;
  std::uint32_t subgroup_max = 0;
  std::uint32_t max_push_constants = 128;
  std::uint32_t max_shared_memory = 16384;
  std::uint32_t max_workgroup_invocations = 128;
  std::uint32_t max_workgroup_count_x = 65535;  // maxComputeWorkGroupCount[0] : groupes d'un dispatch en x
  std::uint32_t storage_alignment = 256;        // minStorageBufferOffsetAlignment : décalage d'une vue liée
};

}  // namespace ajp::gpu::vk::net
