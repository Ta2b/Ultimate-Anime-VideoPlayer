// Les ressources du processus au journal (A33-08) : une fuite par réouverture ou par fichier se voit dans
// ajp.log, en chiffres, avant qu'un utilisateur ne la sente au bout de quelques heures.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace ajp::app::play {

struct ProcessResources {
  std::optional<std::uint64_t> resident_bytes;  // mémoire résidente (Linux, Windows : ensemble de travail)
  std::optional<int> threads;
  std::optional<int> handles;  // descripteurs ouverts (Linux), poignées (Windows)
};

ProcessResources process_resources();
// « mémoire 812 Mio, 46 threads, 131 descripteurs » ; ce que le système ne dit pas est omis.
std::string describe(const ProcessResources& resources);

// B33-05 : la mémoire vidéo de la chaîne ouverte (utilisée, totale), ajoutée à la ligne des 5 minutes ; nul :
// aucune chaîne. Posé par ChainStats à l'ouverture, retiré à sa fermeture (le fil du journal ne lit plus les
// noyaux une fois retiré).
using DeviceMemoryReader = std::function<std::optional<std::pair<std::uint64_t, std::uint64_t>>()>;
void set_device_memory_reader(DeviceMemoryReader reader);

// La dérive à signaler : mémoire résidente ou descripteurs au-delà de la moitié de plus que le premier relevé
// (à 5 min), le texte de l'avertissement ; vide sinon.
std::string resource_drift(const ProcessResources& first, const ProcessResources& now);

// Une ligne « ressources : … » au journal toutes les 5 minutes, jusqu'à la destruction du fil, avec la
// mémoire vidéo et un avertissement de dérive (B33-05 : un essai de 8 h se lit dans ajp.log seul).
std::jthread start_resource_log();

class PlaceboScreen;

// release_ring à la sortie de la portée (RingRelease, A33-01), puis les ressources une fois la chaîne
// démontée au journal (C33-03) : c'est ce relevé qui se compare d'une réouverture à l'autre, pas celui de la
// chaîne encore vivante.
class ResourcesAfterTeardown {
 public:
  explicit ResourcesAfterTeardown(PlaceboScreen& screen) : screen_(screen) {}
  ResourcesAfterTeardown(const ResourcesAfterTeardown&) = delete;
  ResourcesAfterTeardown& operator=(const ResourcesAfterTeardown&) = delete;
  ~ResourcesAfterTeardown();

 private:
  PlaceboScreen& screen_;
};

}  // namespace ajp::app::play
