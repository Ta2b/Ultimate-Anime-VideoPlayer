// Thread preparer : construit les moteurs demandés, un à la fois, publie leur
// progression pour l'OSD et rend un Artifact ou un refus pour chaque demande.
//
// Il n'attend que sa boîte (demandes et crédits) et, pendant une construction,
// le constructeur lui-même. Une demande s'annule par son propre stop_token ;
// l'arrêt du thread annule la construction en cours, dont le refus Cancelled
// part encore au puits, puis abandonne la file sans rien construire.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <stop_token>
#include <string>
#include <thread>

#include "core/inbox.h"
#include "core/messages.h"
#include "core/types.h"
#include "gpu/backend.h"

namespace ajp::prepare {

struct BuildRequest {
  std::uint64_t id = 0;  // rendu tel quel dans BuildProgress et BuildDone
  gpu::ModelSpec model;
  gpu::ShapeSet shapes;
  gpu::Backend* backend = nullptr;  // non possédé : vit plus longtemps que le Preparer
  std::stop_token cancel;           // annule la demande, en attente ou en cours
};

struct BuildProgress {
  std::uint64_t id = 0;
  std::string model;
  float fraction = 0.0F;  // 0 à 1
  std::string stage;
  std::size_t queued = 0;  // demandes en attente derrière celle-ci
};

struct BuildDone {
  std::uint64_t id = 0;
  std::string model;
  std::expected<gpu::Artifact, Refusal> result;
};

// Ce que le preparer appelle pour publier ; app l'implémente par sa boîte.
class PreparerSink {
 public:
  virtual ~PreparerSink() = default;
  // false : progression perdue, la suivante la remplacera.
  virtual bool publish(const BuildProgress& progress) = 0;
  // false : borne atteinte et done intact. Le preparer le garde et réessaie au
  // prochain Credit ou au résultat suivant, toujours dans l'ordre des demandes.
  virtual bool deliver(BuildDone&& done) = 0;
  // Le thread s'est arrêté sur une exception (ThreadFailed) : plus aucune demande ne sera
  // servie. Appelé depuis le thread preparer, une fois au plus.
  virtual void failed(const Refusal& refusal) = 0;
};

using PreparerInbox = Inbox<BuildRequest, Credit>;

struct PreparerBounds {
  std::size_t requests = 32;
  std::size_t credits = 8;
};

class Preparer {
 public:
  // Lance le thread. Refus : thread refusé par le système (ThreadUnavailable).
  static std::expected<std::unique_ptr<Preparer>, Refusal> start(PreparerSink& sink,
                                                                 PreparerBounds bounds = {});
  Preparer(const Preparer&) = delete;
  Preparer& operator=(const Preparer&) = delete;
  ~Preparer() = default;  // jthread : arrêt demandé, puis attente de la fin du thread

  // false si la borne des demandes est atteinte : request reste intacte.
  bool submit(BuildRequest&& request);
  // Le consommateur des BuildDone rend une place.
  bool credit(Credit credit = {});

 private:
  explicit Preparer(PreparerBounds bounds) : inbox_({bounds.requests, bounds.credits}) {}

  PreparerInbox inbox_;
  std::jthread thread_;  // dernier membre : démarre après la boîte, s'arrête avant elle
};

}  // namespace ajp::prepare
