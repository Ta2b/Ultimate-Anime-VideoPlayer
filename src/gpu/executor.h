// Exécuteur d'un Program : de chaque image décodée aux images finales à écrire dans le ring.
//
// Par image décodée : ingestion (et préparation de RIFE), mesure de la paire qu'elle ferme
// avec la précédente, puis les niveaux. La mesure est relue sans attente : une source attend
// sa mesure dans une petite file avant d'entrer au premier niveau, et une mesure perdue
// (le backend n'en rend que la dernière) vaut « nouveau dessin, pas de coupe ».
// Chaque niveau décide par son LevelScheduler (cadence et dessins de core) ; les images se
// calculent au moment où une décision les cite, et une image montrée telle quelle partage le
// contenu de son entrée. Rien n'attend côté hôte : tout s'ajoute à la file du backend.
//
// Un seul thread (gpu). Les réserves de tenseurs et de contenus se remplissent aux premières
// images puis se recyclent : aucune allocation hôte par image en régime. Les noyaux gardent
// eux-mêmes l'image décodée qu'ils lisent jusqu'à la fin de leur lecture (kernel_ops.h).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <vector>

#include "core/frame.h"
#include "core/ring.h"
#include "core/types.h"
#include "gpu/image_store.h"
#include "gpu/level_scheduler.h"
#include "gpu/program.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu {

// Une image finale, prête pour le ring : RingFrame sans slot, génération ni valeur.
struct FinalImage {
  RingFrame frame;
  ContentRef content;  // content->high : l'image à écrire
};

struct ExecutorCounters {
  std::uint64_t sources = 0;        // images décodées entrées
  std::uint64_t outputs = 0;        // images finales
  std::uint64_t interpolated = 0;   // images finales calculées entre deux entrées
  std::uint64_t repeated = 0;       // images finales REPEATED
  std::uint64_t scene_cuts = 0;     // coupes de scène mesurées entre deux sources
  std::uint64_t lost_measures = 0;  // mesures jamais relues : dessin neuf par défaut
};

class Executor {
 public:
  // Bornes : sources en attente de leur mesure, images finales en file, entrées gardées par
  // niveau, tenseurs et contenus vivants.
  static constexpr std::size_t kMaxWaiting = 2;
  static constexpr std::size_t kMaxFinal = 128;
  static constexpr std::size_t kStore = LevelScheduler::kMaxLive + 4;
  static constexpr std::size_t kMaxTensors = 256;
  static constexpr std::size_t kMaxContents = 256;

  explicit Executor(std::shared_ptr<const Program> program);
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;
  ~Executor();

  const Program& program() const { return *program_; }

  // L'image décodée suivante. Une image d'une époque passée est ignorée ; une époque plus
  // récente vaut un Flush. Refus : identité dérivée, taille que le Plan n'accepte pas,
  // refus d'un noyau, d'un moteur ou d'un niveau (qui repart alors de l'image suivante).
  std::expected<void, Refusal> push(const DecodedFrame& frame);

  // Comme push, sans rien émettre : la dernière source d'un Program précédent devient
  // l'origine de celui-ci, déjà montrée.
  std::expected<void, Refusal> seed(const DecodedFrame& frame);

  // Fin du flux : les mesures manquantes valent « dessin neuf », les anticipations « aucun
  // dessin suivant ».
  std::expected<void, Refusal> finish();

  // Seek ou nouveau Plan : tout ce qui précède `epoch` est jeté, sans attente.
  void flush(std::uint32_t epoch);

  bool has_output() const { return !finals_.empty(); }
  FinalImage& front() { return finals_.front(); }
  void pop() { finals_.pop(); }
  std::size_t pending_outputs() const { return finals_.size(); }

  const ExecutorCounters& counters() const { return counters_; }
  std::size_t tensors_created() const { return tensors_.created(); }

 private:
  struct Picture {
    SourceId id;
    ContentRef content;
  };
  struct Waiting {
    LevelInput input;
    ContentRef content;
    bool measured = false;
  };
  struct MotionCache {
    bool valid = false;
    SourceId a;
    SourceId b;
    TensorRef flow;
    TensorRef mask;
  };
  // Tailles de la séquence en cours, déduites de la première image et du Plan.
  struct Geometry {
    int width = 0;  // source
    int height = 0;
    int work_width = 0;  // après réduction, avant agrandissement
    int work_height = 0;
    kernels::RifeGeometry rife;
  };

  std::expected<void, Refusal> accept(const DecodedFrame& frame);
  std::expected<Geometry, Refusal> geometry_of(const DecodedFrame& frame) const;
  std::expected<ContentRef, Refusal> ingest(const DecodedFrame& frame);
  void poll_scores();
  std::expected<void, Refusal> drain_waiting(bool all);

  std::expected<void, Refusal> feed(std::size_t level, const LevelInput& input, ContentRef content);
  std::expected<void, Refusal> take(std::size_t level, std::span<const LevelOutput> outputs);
  std::expected<ContentRef, Refusal> materialize(std::size_t level, const LevelOutput& output);
  std::expected<ContentRef, Refusal> interpolate(std::size_t level, const ContentRef& a, const ContentRef& b,
                                                 core::Ratio u);
  std::expected<ContentRef, Refusal> warp(std::size_t level, const LevelOutput& output, const ContentRef& a,
                                          const ContentRef& b);
  std::expected<void, Refusal> finish_content(Content& content);
  RingFrame ring_frame(const LevelOutput& output);
  std::expected<void, Refusal> emit(const RingFrame& frame, ContentRef content);

  std::expected<TensorRef, Refusal> image(int width, int height, int channels = 3);
  std::expected<ContentRef, Refusal> new_content();
  const Picture* find(std::size_t level, SourceId id) const;
  void prune(std::size_t level);

  std::shared_ptr<const Program> program_;
  TensorPool tensors_;     // avant les contenus, qui tiennent des tenseurs
  ContentSlots contents_;  // avant tout ce qui tient des contenus
  std::vector<LevelScheduler> levels_;
  std::array<std::array<Picture, kStore>, kMaxLevels> stores_{};
  FixedQueue<Waiting, kMaxWaiting + 1> waiting_;
  FixedQueue<FinalImage, kMaxFinal> finals_;
  MotionCache motion_;
  Geometry geometry_;
  bool sized_ = false;                    // geometry_ vaut pour la séquence en cours
  std::optional<DecodedFrame> previous_;  // la source précédente de la séquence, pour la mesure
  std::uint32_t epoch_ = 0;
  bool seeding_ = false;
  ExecutorCounters counters_;
};

}  // namespace ajp::gpu
