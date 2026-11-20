// La carte de la scène dans l'étage gpu (upscale_stage.h) : lance le flux de NVIDIA sur une paire d'images
// de travail (KernelOps::scene_flow_start), le relit sans jamais attendre (scene_flow_take) et en tire la
// carte (kernels/ref/scene_flow.h, scene_motion_of), posée dans PairFacts::scene pour la paire qu'elle
// décrit.
//
// Deux cadences. Avec l'image suivante ingérée en avance (RIFE, GMFSS : looks_ahead ; DistilDRBA :
// drba_next), la paire (N, N+1) est lancée pendant l'image N et sa carte est là quand l'étage décide de cette
// paire, avant l'interpolation et SPAN. Sans elle (sans interpolation, ou image suivante pas encore décodée),
// la paire (N−1, N) est lancée à l'image N et sa carte n'arrive qu'à l'image suivante : latest() la garde,
// avec sa paire, mais PairFacts ne la reçoit pas. Une paire n'est lancée qu'une fois (started), et jamais à
// cheval sur deux époques.
//
// NVOF encore occupé (plus lent que la chaîne de deux paires) : la paire n'est pas lancée, rien n'attend, et
// la suivante repart sans indices (sautées au journal de fin). Une carte pas encore relue quand l'étage
// décide de sa paire compte « trop tard » ; une relecture remplacée ou périmée, « perdue ».
//
// La caméra en deux temps (porte 2.1, kernels/spec/scene_camera.h) : ajustée sur les vecteurs de NVOF
// (ref::scene_camera_of), affinée par Lucas-Kanade sur les lumas de la paire, sur la file de NVOF
// (KernelOps::scene_camera_start) ; la carte n'est tirée qu'avec la caméra affinée, ou, si l'étage décide de
// la paire avant, avec celle de NVOF (« affinage trop tard »). L'affinage part dès le lancement de la paire,
// enfilé derrière son flux, depuis la dernière caméra du même plan : sans attendre que l'hôte
// relise les tuiles, il est fini avec elles. Sans caméra du plan (début, coupe), ou si sa vérification sur
// l'image passe de loin celle du plan (kEarlyFloor, kEarlyFactor), il repart des tuiles relues, depuis la
// caméra de NVOF.
// advance(), appelé à chaque image de sortie, fait avancer le tout sans attendre. Un affinage encore en vol
// quand l'étage décide de sa paire est relu plus tard : last_refined() le donne aux calques, qui reprennent
// la caméra d'une paire voisine.
//
// Absente (--scene-map éteint, backend sans NVOF, Turing, refus) : rien n'est lancé, chaque paire garde
// scene vide, et la chaîne fait exactement ce qu'elle faisait. Un refus éteint la carte pour la lecture, avec
// une ligne au journal.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "kernels/ref/scene_flow.h"
#include "kernels/spec/scene_flow.h"

namespace kirari::app::play {

class SceneMap {
 public:
  explicit SceneMap(bool wanted, bool roundtrip = false) : on_(wanted), roundtrip_(roundtrip) {}
  bool on() const { return on_; }

  // Lance la paire (left, right) des images de travail a et b, d'époque epoch. Les indices temporels de
  // NVOF repartent si la paire ne suit pas la précédente (début, saut, nouvelle époque) ou après une coupe.
  void start(gpu::KernelOps& ops, const gpu::Tensor& a, const gpu::Tensor& b, SourceId left, SourceId right,
             std::uint32_t epoch);
  // La carte de la paire (left, right) si elle est relue ; sinon rien. Relit sans attente ce qui est fini.
  std::optional<kernels::SceneMotion> for_pair(gpu::KernelOps& ops, SourceId left, SourceId right,
                                               std::uint32_t epoch = 0);
  // La paire (left, right) a-t-elle été lancée (la dernière lancée) ?
  bool started(SourceId left, SourceId right, std::uint32_t epoch) const {
    return last_epoch_ == epoch && last_left_ && *last_left_ == left && last_right_ && *last_right_ == right;
  }
  // La paire (left, right) enjambe une coupe (mesurée par le lecteur) : sa carte, relue ou à venir, n'a que
  // des tuiles incertaines (kernels::ref::mark_cut), et la paire lancée ensuite repart sans indices ni
  // classes.
  void cut(SourceId left, SourceId right, std::uint32_t epoch = 0);
  // Relit ce qui est fini et lance l'affinage des caméras, sans attendre (à chaque image de sortie).
  void advance(gpu::KernelOps& ops) {
    if (on_) poll(ops);
  }

  // La dernière carte relue et sa paire, pour qui accepte une paire de retard.
  struct Latest {
    SourceId left;
    SourceId right;
    std::uint64_t ticket = 0;
    std::uint32_t epoch = 0;
    int flow_width = 0;
    int flow_height = 0;
    int divisor = 1;
    kernels::SceneMotion motion;
    std::vector<std::uint8_t> photo_edge_a;  // attente du retour pour ce ticket
  };
  const std::optional<Latest>& latest() const { return latest_; }

  // La dernière caméra affinée relue, à l'heure ou après que l'étage a décidé de sa paire, en pixels de
  // travail (ref::scene_camera_work, comme SceneMotion::camera et residual).
  struct Refined {
    SourceId left;
    SourceId right;
    std::array<double, 6> camera{};
    double residual = 0.0;
    int residual_tiles = 0;
  };
  const std::optional<Refined>& last_refined() const { return last_refined_; }

  // La ligne du journal de fin : paires lancées, relues, posées, sautées, trop tard, perdues, coupes et
  // fondus vus.
  void log_summary() const;

 private:
  struct Pending {
    std::uint64_t ticket = 0;
    SourceId left;
    SourceId right;
    std::uint32_t epoch = 0;
    bool restart = false;     // les classes des paires d'avant ne valent pas pour elle
    bool across_cut = false;  // elle enjambe une coupe dite par le lecteur (cut)
    std::uint64_t shot = 0;   // le plan (shot_ à son lancement)
    // Après la relecture des tuiles : la grille, la caméra, et son affinage en vol (0 : aucun), parti
    // derrière le flux depuis la caméra du plan (early_limit : la vérification qu'il ne doit pas dépasser)
    // ou depuis celle de NVOF (early_limit 0).
    std::optional<kernels::SceneFlowTiles> tiles;
    kernels::ref::SceneCamera camera;
    std::uint64_t refine = 0;
    double early_limit = 0.0;
  };
  // Un affinage encore en vol quand l'étage a décidé de sa paire : relu plus tard pour last_refined().
  struct Late {
    std::uint64_t refine = 0;
    double early_limit = 0.0;
    std::uint64_t shot = 0;
    SourceId left;
    SourceId right;
    kernels::ref::SceneCamera camera;  // celle de NVOF
    int flow_width = 0;
    int flow_height = 0;
    int divisor = 1;
  };
  // La dernière caméra connue d'un plan (de NVOF, puis affinée), d'où part l'affinage de la paire suivante,
  // et la vérification sur l'image de sa dernière caméra affinée (pixels du flux ; négative : aucune).
  struct Guess {
    std::uint64_t shot = 0;
    kernels::ref::SceneCamera camera;
    double check = -1.0;
  };
  void poll(gpu::KernelOps& ops);
  // Les tuiles de la paire de tête : faux si elles ne sont pas encore là, ou si la paire est perdue
  // (retirée).
  bool take_tiles(gpu::KernelOps& ops);
  // L'affinage derrière le flux, au lancement de la paire, depuis la caméra de son plan (guess_).
  void request_early(gpu::KernelOps& ops, Pending& pair);
  void request_refine(gpu::KernelOps& ops, Pending& pair);
  // Vrai quand l'affinage de la paire est fini, posé sur sa caméra, ou perdu ; un affinage parti de la caméra
  // du plan qui n'est pas gardé repart depuis celle de NVOF (faux tant qu'il est en vol).
  bool collect_refine(gpu::KernelOps& ops, Pending& pair);
  // L'affinage relu de camera (celle de NVOF) : faux s'il est parti de la caméra du plan (limit > 0) et que
  // sa vérification passe limit, ou qu'il s'écarte de camera de plus de kEarlyGap ; camera reste alors
  // intacte. Sinon posé ; last_refined_ et la vérification du plan suivent.
  bool apply_refined(kernels::ref::SceneCamera& camera, const kernels::SceneCameraRefined& refined,
                     double limit, const Late& where);
  void poll_late(gpu::KernelOps& ops);
  static Late late_of(const Pending& pair);  // la paire, sa grille et son plan (refine 0)
  void keep_late(Pending& pair);             // son affinage en vol passe dans late_
  void finish(Pending& pair);                // la carte de la paire de tête, avec sa caméra telle qu'elle est
  // L'étage décide de la paire (left, right) : elle, et celles d'avant que l'affinage retient encore, partent
  // avec la caméra de NVOF, dans l'ordre (l'historique des classes les suit une à une).
  void settle(gpu::KernelOps& ops, SourceId left, SourceId right, std::uint32_t epoch);
  void stop(const Refusal& refusal, const char* where);
  void stop_refine(const Refusal& refusal, const char* where);

  bool on_ = false;
  bool roundtrip_ = false;
  std::uint64_t roundtrip_kept_ = 0;
  std::vector<Pending> pending_;
  std::vector<Late> late_;
  std::optional<Latest> latest_;
  std::optional<Refined> last_refined_;
  std::optional<Guess> guess_;
  std::uint64_t shot_ = 0;  // un de plus à chaque paire qui repart sans indices
  kernels::ref::SceneHistory history_;
  std::optional<SourceId> last_left_;
  std::optional<SourceId> last_right_;
  std::uint32_t last_epoch_ = 0;
  bool restart_ = true;
  std::uint64_t started_ = 0;
  std::uint64_t read_ = 0;
  std::uint64_t attached_ = 0;
  std::uint64_t busy_ = 0;    // pas lancées : NVOF occupé
  std::uint64_t missed_ = 0;  // lancées, pas encore relues quand l'étage a décidé de leur paire
  std::uint64_t lost_ = 0;    // relectures remplacées ou périmées
  std::uint64_t cuts_ = 0;
  std::uint64_t fades_ = 0;
  std::uint64_t camera_fits_ = 0;
  bool refine_ = true;               // le backend sait affiner la caméra
  std::uint64_t refined_ = 0;        // caméras affinées
  std::uint64_t refine_late_ = 0;    // affinage pas fini quand l'étage a décidé de la paire
  std::uint64_t refine_missed_ = 0;  // pas lancé : lumas réécrites, ou relectures toutes en vol
  std::uint64_t refine_early_ = 0;   // lancés derrière le flux, depuis la caméra du plan
  std::uint64_t refine_off_ = 0;     // de ceux-là, écartés : vérification trop haute, ou trop loin de NVOF
  std::uint64_t refine_after_ = 0;   // relus après que l'étage a décidé de leur paire
  // Temps de l'hôte (le thread du lecteur), µs : lancer une paire (scene_flow_start) et tirer une carte
  // (scene_camera_of, scene_motion_of). Avec les lumas, les tuiles et l'affinage sur les SM, c'est ce que la
  // carte ajoute à chaque image.
  double start_us_ = 0.0;
  double map_us_ = 0.0;
};

}  // namespace kirari::app::play
