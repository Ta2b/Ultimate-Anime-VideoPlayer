// Profil des dispatchs d'un réseau : temps GPU et nombre de dispatchs par nom de shader.
//
// Option de débogage (DebugFlags::profile_dispatches, BackendOptions::profile_dispatches) : le
// moteur écrit une estampille avant et après chaque dispatch de l'enregistrement, lit les
// résultats une fois l'exécution finie, et journalise un tableau par shader à sa destruction.
// Désactivée, aucun pool n'est créé et l'enregistrement est celui de toujours. Chaque dispatch est
// encadré de deux commandes d'horodatage, donc de deux barrières : la somme des intervalles mesure
// les noyaux, pas les trous entre eux (voir ProfileReport::span_ms pour la durée d'ensemble).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "gpu/ir/ir.h"
#include "gpu/vulkan/backend/net_program.h"

namespace ajp::gpu::vk::net {

// Une exécution mesurée : durée de chaque dispatch (ms, dans l'ordre du programme) et durée de la
// première estampille à la dernière.
struct ProfileRun {
  std::vector<double> dispatch_ms;
  double span_ms = 0.0;
};

struct ShaderTime {
  std::string shader;
  std::uint32_t dispatches = 0;
  double ms = 0.0;     // médiane, sur les exécutions, de la somme de ses dispatchs
  double share = 0.0;  // part de ProfileReport::total_ms, entre 0 et 1
};

struct ProfileReport {
  std::vector<ShaderTime> shaders;  // par temps décroissant
  double total_ms = 0.0;            // médiane de la somme des dispatchs
  double span_ms = 0.0;             // médiane de la durée entre la première et la dernière estampille
  std::size_t runs = 0;
};

// Le rapport des exécutions : médianes par shader. Vide sans exécution.
ProfileReport summarize_profile(std::span<const Dispatch> dispatches, std::span<const ProfileRun> runs);

// Le tableau pour le journal : une ligne par shader, puis le total.
std::string format_profile(const ProfileReport& report);

// Ce que le programme lance, avant toute exécution : par shader, le nombre de dispatchs et les
// octets lus et écrits estimés par les formes (un tenseur lié compte en entier ; sortie du nœud :
// écrit, sinon lu ; un blob : lu).
struct ShaderCount {
  std::string shader;
  std::uint32_t dispatches = 0;
  std::uint64_t read_bytes = 0;
  std::uint64_t written_bytes = 0;
};

// Par nombre de dispatchs décroissant.
std::vector<ShaderCount> static_count(const ir::Graph& graph, const Program& program);

}  // namespace ajp::gpu::vk::net
