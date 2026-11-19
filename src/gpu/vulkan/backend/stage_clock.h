// Le profil GPU de la lecture sous Vulkan (KernelOps::gpu_profile, gpu/gpu_profile.h), comme celui de
// TensorRT : chaque étape de la chaîne (ingestion, écriture du ring, moteur SPAN…) bornée par deux
// estampilles, résumée en nombre de mesures, total, médiane et 95e centile.
//
// Une estampille est un petit enregistrement à part (réinitialise sa requête, l'écrit en fin de pipeline),
// glissé dans le lot sur la file de la voie courante, avant puis après les soumissions de l'étape : la
// première attend la fin de ce qui la précède sur cette file, la seconde celle de l'étape. Sans barrière en
// plus dans les enregistrements de la chaîne, sans attente hôte : une paire se relit quand ses deux
// estampilles sont finies, constaté sans attendre, au début d'une étape suivante ou à profile().
//
// Comme sous CUDA (GpuTiming), une étape qui attend un sémaphore d'ailleurs (image décodée, slot du ring)
// compte cette attente. Toutes les paires en vol : l'étape n'est pas mesurée, rien n'attend. Une étape
// fermée sur une autre file que celle de son ouverture est abandonnée (ses deux estampilles ne se
// comparent pas). Sans horodatage sur la file de calcul, create() rend nullptr : gpu_profile reste vide.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "core/sample_reservoir.h"
#include "gpu/gpu_profile.h"
#include "gpu/vulkan/backend/net_query.h"
#include "gpu/vulkan/backend/queue.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/recording.h"

namespace kirari::gpu::vk {

class StageClock {
 public:
  // Paires d'estampilles en vol au plus ; au-delà, une étape n'est pas mesurée.
  static constexpr std::size_t kPairs = 64;
  // Mesures gardées par étape pour les centiles ; toutes comptent dans le nombre et le total.
  static constexpr std::size_t kMaxSamples = std::size_t{1} << 16U;
  static constexpr std::size_t kNone = ~std::size_t{0};

  // nullptr : la file de calcul n'a pas d'horodatage (timestampValidBits à 0), ou le pool de requêtes
  // n'a pas pu se créer. Le contexte, l'allocateur et la file doivent survivre à l'horloge.
  static std::unique_ptr<StageClock> create(VkContext& context, MemoryAllocator& memory, BackendQueue& queue);

  StageClock(const StageClock&) = delete;
  StageClock& operator=(const StageClock&) = delete;
  StageClock(StageClock&&) = delete;
  StageClock& operator=(StageClock&&) = delete;
  // Attend la fin des estampilles en vol (~Recording) avant de détruire le pool.
  ~StageClock();

  // Ouvre la mesure de l'étape name sur la file de la voie courante : la paire, ou kNone quand l'étape
  // n'est pas mesurée (toutes en vol, estampille refusée).
  std::size_t open(std::string_view name);
  // Ferme la paire ouverte par open ; kNone : rien.
  void close(std::size_t pair);

  // Les étapes mesurées jusqu'ici, par nom ; seules les paires finies comptent. Appelable depuis un autre
  // fil que celui de la file.
  std::vector<GpuStageSummary> profile();

  // Ferme la mesure à la fin de la portée : une étape à plusieurs soumissions (RIFE fusionné) en une.
  class Scope {
   public:
    Scope(StageClock* clock, std::string_view name)
        : clock_(clock), pair_(clock != nullptr ? clock->open(name) : kNone) {}
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope(Scope&&) = delete;
    Scope& operator=(Scope&&) = delete;
    ~Scope() {
      if (clock_ != nullptr) clock_->close(pair_);
    }

   private:
    StageClock* clock_;
    std::size_t pair_;
  };

 private:
  enum class State : std::uint8_t { Free, Open, Closed, Abandoned };
  struct Pair {
    std::unique_ptr<Recording> begin;  // requête 2 i
    std::unique_ptr<Recording> end;    // requête 2 i + 1
    State state = State::Free;
    std::size_t stage = 0;  // indice dans stages_
    std::size_t file = 0;   // la file de l'ouverture
  };
  struct Stage {
    std::string name;
    core::SampleReservoir<float> all{kMaxSamples};
  };

  StageClock(VkContext& context, MemoryAllocator& memory, BackendQueue& queue, net::TimestampSupport support,
             std::unique_ptr<net::QueryPool> pool);
  // La paire i prête à servir : ses deux estampilles enregistrées (la première fois), sinon refus.
  Done ready(std::size_t i);
  std::expected<std::unique_ptr<Recording>, Refusal> stamp(std::uint32_t query);
  // Relit les paires finies et libère les abandonnées finies, sans attendre (verrou tenu).
  void collect();
  std::size_t stage_index(std::string_view name);

  VkContext& context_;
  MemoryAllocator& memory_;
  BackendQueue& queue_;
  net::TimestampSupport support_;
  std::unique_ptr<net::QueryPool> pool_;  // avant pairs_ : détruit après les estampilles
  std::mutex mutex_;
  std::vector<Pair> pairs_;
  std::vector<Stage> stages_;
  std::map<std::string, std::size_t, std::less<>> by_name_;
};

}  // namespace kirari::gpu::vk
