// Le thread gpu vu par app : il reçoit les images décodées de media, écrit le ring et
// envoie chaque RingFrame au thread main.
//
// Interface de la tranche verticale, tenue par une doublure dans les tests ; le module gpu
// (thread, Program, ring) s'y branche par un adaptateur dans player.cpp.
#pragma once

#include <cstdint>

#include "core/frame.h"
#include "core/messages.h"
#include "core/ring.h"

namespace ajp::app::play {

// Où le thread gpu pousse ses images, depuis son thread. false : la boîte de main est
// pleine, le slot reste à gpu qui réessaiera.
class RingSink {
 public:
  virtual ~RingSink() = default;
  virtual bool try_push(const RingFrame& frame) = 0;
};

class VideoStage {
 public:
  virtual ~VideoStage() = default;
  // Le puits des images décodées, que media appelle depuis le thread source.
  virtual FrameSink& frames() = 0;
  virtual bool post(const SlotFree& free) = 0;
  virtual bool post(const Flush& flush) = 0;
  virtual bool post(const Command& command) = 0;
  // Toute image reçue est écrite dans le ring et sa RingFrame postée (ou ne le sera jamais : la
  // dernière image d'un étage qui attend la suivante). Lu par main après StreamEnded : la fin d'un
  // export. Une doublure sans file d'attente l'est toujours.
  virtual bool idle() const { return true; }
};

}  // namespace ajp::app::play
