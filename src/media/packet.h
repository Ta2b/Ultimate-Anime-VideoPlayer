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
  int track = -1;                      // index du flux
  std::int64_t pts_us = kNoTimestamp;  // depuis le début du fichier
  std::int64_t dts_us = kNoTimestamp;
  std::int64_t duration_us = 0;
  bool keyframe = false;
  std::uint32_t epoch = 0;             // posé à la livraison, comme DecodedFrame::epoch
  std::shared_ptr<const AVPacket> av;  // pour avcodec_send_packet

  std::span<const std::byte> data() const;
};

// Ce que le thread source appelle pour livrer un paquet, comme FrameSink pour
// les images. false : la borne est atteinte, le producteur attend un Credit.
class PacketSink {
 public:
  virtual ~PacketSink() = default;
  virtual bool try_push(Packet&& packet) = 0;
};

// File bornée d'une piste, pour brancher un module qui ne connaît pas media :
// le thread source pousse, le consommateur retire depuis son propre thread.
class PacketQueue final : public PacketSink {
 public:
  // on_space est appelé par try_pop, hors du verrou, quand une place se libère
  // après un refus : le producteur y pousse un Credit dans sa boîte.
  static std::expected<std::unique_ptr<PacketQueue>, Refusal> create(std::size_t bound,
                                                                     std::function<void()> on_space);

  bool try_push(Packet&& packet) override;
  std::optional<Packet> try_pop();
  // Jette les paquets d'une époque passée, après un Flush.
  void discard_before(std::uint32_t epoch);
  std::size_t size() const;

 private:
  PacketQueue(std::size_t bound, std::function<void()> on_space);

  mutable std::mutex mutex_;
  std::deque<Packet> queue_;
  std::size_t bound_;
  bool refused_ = false;  // un try_push a échoué depuis le dernier try_pop
  std::function<void()> on_space_;
};

inline constexpr std::size_t kMaxPacketQueue = 4096;

}  // namespace ajp::media
