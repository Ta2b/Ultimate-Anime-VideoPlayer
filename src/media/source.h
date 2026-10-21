// Thread source : lit le fichier, décode la piste vidéo, livre les images à
// gpu par un FrameSink et les paquets audio et sous-titres par des PacketSink.
//
// Protocole avec app :
// - Flush{e} : l'image et le paquet tenus sont jetés ; ceux livrés ensuite
//   portent l'époque e, qui doit croître. app l'envoie aussi à gpu, audio et subs.
//   Rien ne part plus jusqu'au cmd::Seek qui suit : sans cette attente, une
//   image de l'ancienne position partirait sous la nouvelle époque entre les
//   deux messages. app envoie donc toujours Flush puis Seek, même vers la
//   position affichée (changement de Plan) ; l'Inbox garde cet ordre.
// - cmd::Seek : repositionne. Exact : point clé qui s'affiche à ou avant la
//   cible, puis décodage et rejet jusqu'à l'image visible à la cible. Rapide :
//   point clé le plus proche. Un Seek refusé est signalé et la lecture reprend
//   où elle était.
// - cmd::Quit, ou l'arrêt du std::jthread : la boucle se termine.
// - Credit : un consommateur a libéré une place ; le thread retente sa livraison.
// Les autres commandes (pause, pas image, seek relatif) se traduisent dans app,
// qui connaît la position affichée, en un cmd::Seek absolu.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <thread>
#include <vector>

#include "core/frame.h"
#include "core/inbox.h"
#include "core/messages.h"
#include "core/types.h"
#include "media/demuxer.h"
#include "media/packet.h"
#include "media/video_decoder.h"

namespace ajp::media {

using SourceInbox = Inbox<Credit, Command, Flush>;

// Bornes de la boîte : Credit, Command, Flush. Un Credit refusé sur borne
// pleine est sans perte : un autre attend déjà dans la boîte.
inline constexpr SourceInbox::Bounds kSourceInboxBounds{8, 16, 4};

// Ce que le thread source signale, depuis son thread : l'implémentation
// pousse dans la boîte de app.
class SourceListener {
 public:
  virtual ~SourceListener() = default;
  virtual void end_of_stream(std::uint32_t epoch) = 0;
  virtual void failed(const Refusal& refusal) = 0;
};

struct PacketRoute {
  int track = -1;
  PacketSink* sink = nullptr;
};

// Consommateurs du thread : ils doivent vivre plus longtemps que lui.
struct SourceLinks {
  FrameSink* frames = nullptr;
  SourceListener* listener = nullptr;
  std::vector<PacketRoute> packets;  // pistes audio et sous-titres à livrer
};

// Ce que le thread possède : le fichier ouvert et le décodeur de sa piste vidéo.
struct SourceMedia {
  Demuxer demuxer;
  VideoDecoder decoder;
  int video_track = -1;
};

// Ouvre le décodeur de la piste vidéo choisie.
std::expected<SourceMedia, Refusal> open_source(Demuxer demuxer, int video_track,
                                                const DecoderOptions& options);

class SourceThread {
 public:
  // Vérifie les liens (pistes routées audio ou sous-titres, puits non nuls),
  // limite la lecture aux pistes utiles, puis lance le thread. La boîte existe
  // avant lui, pour que les consommateurs sachent où rendre leurs Credit, et
  // doit lui survivre.
  static std::expected<std::unique_ptr<SourceThread>, Refusal> start(SourceMedia media, SourceLinks links,
                                                                     SourceInbox& inbox);

  SourceThread(const SourceThread&) = delete;
  SourceThread& operator=(const SourceThread&) = delete;
  ~SourceThread() = default;  // le std::jthread demande l'arrêt et attend

 private:
  SourceThread() = default;

  std::jthread thread_;
};

}  // namespace ajp::media
