// Paquets compressés des pistes audio et sous-titres, livrés par le thread
// source aux modules qui les décodent. Aucun en-tête FFmpeg ici.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>

#include "core/types.h"
#include "media/track.h"

struct AVPacket;

namespace ajp::media {

// Paquet d'une piste, partagé sans copie avec le module qui le décode.
struct Packet {
  int track = -1;                // index du flux
  std::int64_t pts_us = kNoPts;  // depuis le début du fichier
  std::int64_t dts_us = kNoPts;
  std::int64_t duration_us = 0;
  bool keyframe = false;
  std::uint32_t epoch = 0;             // posé à la livraison, comme DecodedFrame::epoch
  std::shared_ptr<const AVPacket> av;  // pour avcodec_send_packet

  std::span<const std::byte> data() const;
  // Octets tenus en mémoire : les données plus kPacketOverhead.
  std::size_t footprint() const;
};

// Estimation de ce que coûte un paquet hors de ses données (AVPacket, tampon
// partagé, blocs de contrôle) : un paquet vide pèse quand même.
inline constexpr std::size_t kPacketOverhead = 256;

// Ce que le thread source appelle pour livrer un paquet, comme FrameSink pour
// les images. false : la borne est atteinte, le producteur attend un Credit.
//
// Contrat du consommateur (audio, subs, ou app qui fait le pont) : il retire
// ses paquets à son propre rythme, horloge audio ou rendu des sous-titres,
// jamais selon la vidéo. Il ne fait pas dépendre ses retraits de la première
// image d'une époque, ni de l'avance de gpu : la source décode la vidéo dans
// la même boucle que la lecture du fichier, et ne la reprend qu'après le
// Credit d'une file pleine (voir media/source.h).
class PacketSink {
 public:
  virtual ~PacketSink() = default;
  virtual bool try_push(Packet&& packet) = 0;
};

// Bornes d'une PacketQueue : la durée en file (somme des duration_us) et les
// octets en file (footprint). Un paquet qui ferait dépasser l'une des deux est
// refusé, sauf dans une file vide : un paquet passe toujours.
struct PacketQueueLimits {
  // Audio : quelques secondes. std::nullopt : sans borne de durée, pour les
  // sous-titres, dont les durées se chevauchent (karaoké, panneaux).
  std::optional<std::int64_t> max_duration_us = 2'000'000;
  std::size_t max_bytes = std::size_t{8} << 20U;
};

inline constexpr std::int64_t kMaxQueueDurationUs = 60'000'000;
inline constexpr std::size_t kMinQueueBytes = 1024;
inline constexpr std::size_t kMaxQueueBytes = std::size_t{1} << 30U;

// File bornée d'une piste, pour brancher un module qui ne connaît pas media :
// le thread source pousse, le consommateur retire depuis son propre thread.
// Bornée en durée et en octets plutôt qu'en paquets : 2 s d'audio font 100
// paquets AAC ou 2400 paquets TrueHD, et une rafale de sous-titres ASS en
// compte des centaines au même instant.
class PacketQueue final : public PacketSink {
 public:
  // on_space est appelé par try_pop, hors du verrou, quand une place se libère
  // après un refus : le producteur y pousse un Credit dans sa boîte.
  // Refus FieldOutOfRange : durée hors de ]0, kMaxQueueDurationUs], octets
  // hors de [kMinQueueBytes, kMaxQueueBytes].
  static std::expected<std::unique_ptr<PacketQueue>, Refusal> create(const PacketQueueLimits& limits,
                                                                     std::function<void()> on_space);

  bool try_push(Packet&& packet) override;
  std::optional<Packet> try_pop();
  // Jette les paquets d'une époque passée, après un Flush.
  void discard_before(std::uint32_t epoch);
  std::size_t size() const;
  std::int64_t duration_us() const;
  std::size_t bytes() const;

 private:
  PacketQueue(const PacketQueueLimits& limits, std::function<void()> on_space);
  void forget(const Packet& packet);  // sous le verrou

  mutable std::mutex mutex_;
  std::deque<Packet> queue_;
  PacketQueueLimits limits_;
  std::int64_t duration_us_ = 0;
  std::size_t bytes_ = 0;
  bool refused_ = false;  // un try_push a échoué depuis le dernier try_pop
  std::function<void()> on_space_;
};

}  // namespace ajp::media
