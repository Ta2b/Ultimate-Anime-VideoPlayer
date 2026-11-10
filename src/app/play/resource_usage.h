// Les ressources du processus au journal (A33-08) : une fuite par réouverture ou par fichier se voit dans
// ajp.log, en chiffres, avant qu'un utilisateur ne la sente au bout de quelques heures.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <thread>

namespace ajp::app::play {

struct ProcessResources {
  std::optional<std::uint64_t> resident_bytes;  // mémoire résidente (Linux, Windows : ensemble de travail)
  std::optional<int> threads;
  std::optional<int> handles;  // descripteurs ouverts (Linux), poignées (Windows)
};

ProcessResources process_resources();
// « mémoire 812 Mio, 46 threads, 131 descripteurs » ; ce que le système ne dit pas est omis.
std::string describe(const ProcessResources& resources);

// Une ligne « ressources : … » au journal toutes les 5 minutes, jusqu'à la destruction du fil.
std::jthread start_resource_log();

}  // namespace ajp::app::play
