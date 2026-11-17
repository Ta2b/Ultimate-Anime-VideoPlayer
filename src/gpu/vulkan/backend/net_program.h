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
// l'arène), un blob du programme, un petit tampon factice pour un binding que le shader
// n'utilise pas, ou le tampon de travail du programme (Program::scratch_bytes : les sommes partielles de
// la découpe de K, V1), partagé par tous ses dispatchs.
enum class BindingKind : std::uint8_t { Tensor, Blob, Dummy, Scratch };

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
  std::uint64_t scratch_bytes = 0;            // tampon de travail (BindingKind::Scratch) ; 0 : aucun
};

// Ce que l'appareil permet aux noyaux.
struct Target {
  bool f16_storage = false;  // tenseurs float16 (variantes « _f16 »)
  bool f16_math = false;     // calcul en fp16 (« _f16p »)
  // Matrices coopératives en tuiles 16 × 16 × 16 proposées (famille et pilote, voir family.h).
  bool matrix_units = false;
  // Et en tête des candidats, choix par défaut sans autotuner : famille de la liste
  // d'autorisation ; faux pour un appareil hors liste (matrix_units_unlisted).
  bool matrix_first = false;
  // Tuiles 16 × 16 × 16 à sommes float16 aussi annoncées (variante coopmat_acc16, deux fois plus
  // rapide sur GeForce, retenue seulement si sa sortie concorde).
  bool matrix_f16_acc = false;
  // Tuiles 16 × 16 × 16 FP8 E4M3 à sommes float32 annoncées par le pilote (VK_EXT_shader_float8,
  // RDNA 4) : la variante coopmat_fp8 des Conv quantifiées en FP8 et le SPAN FP8 (net_span8.h).
  bool matrix_f8 = false;
  // Tuiles 16 × 16 × 16 s8 × s8 → s32 annoncées par le pilote, shaderInt8 activé (RDNA 3 et 4) : la
  // variante coopmat_int8 des Conv quantifiées en INT8 (graphes Q/DQ INT8, SPAN INT8).
  bool matrix_i8 = false;
  // Tuiles 8 × 16 × 16 f16 × f16 → f32 par sous-groupe annoncées (M = 8 : Intel Xe2 sous ANV, Lunar Lake et
  // Battlemage), sur un appareil permis qui n'annonce pas la tuile 16 × 16 × 16 et laisse exiger des
  // sous-groupes de 16 : la variante coopmat_m8 (net_conv_m8.h), toujours derrière le noyau direct.
  bool matrix_m8 = false;
  std::uint32_t subgroup_size = 0;
  // Tailles de sous-groupe qu'un dispatch peut exiger ; 0 : aucune (le pilote choisit).
  std::uint32_t subgroup_min = 0;
  std::uint32_t subgroup_max = 0;
  std::uint32_t max_push_constants = 128;
  std::uint32_t max_shared_memory = 16384;
  std::uint32_t max_workgroup_invocations = 128;
  std::uint32_t max_workgroup_count_x = 65535;  // maxComputeWorkGroupCount[0] : groupes d'un dispatch en x
  // Les entrées d'un Concat sur les canaux Nhwc16 écrites par leur Conv directement dans leur tranche
  // (net_alias.h) ; faux : chaque entrée est copiée.
  bool concat_in_place = true;
  // Les sorties d'un Split sur les canaux Nhwc16 lues dans leur entrée (net_alias.h) ; faux : copiées.
  bool split_views = true;
  std::uint32_t storage_alignment = 256;  // minStorageBufferOffsetAlignment : décalage d'une vue liée
};

}  // namespace ajp::gpu::vk::net
