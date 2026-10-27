// Démultiplexage d'un fichier avec libavformat : pistes, chapitres, paquets, seek.
//
// Tous les horodatages sortent en microsecondes depuis le début du fichier
// (start_time du conteneur), y compris dans l'AVPacket partagé : un décodeur
// règle pkt_timebase à 1/1 000 000 et ses images sortent dans la même échelle.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "core/types.h"
#include "media/packet.h"
#include "media/track.h"

namespace ajp::media {

enum class SeekTarget : std::uint8_t {
  // Point clé qui s'affiche à ou avant la cible, base du seek exact : toutes les
  // images de la cible jusqu'à la fin se décodent depuis lui, même si l'index du
  // conteneur est en dts (MP4 avec images B) ou le GOP ouvert.
  PreviousKeyframe,
  NearestKeyframe,  // point clé le plus proche, avant ou après : seek rapide
};

class Demuxer {
 public:
  static std::expected<Demuxer, Refusal> open(const std::filesystem::path& path);

  Demuxer(Demuxer&&) noexcept;
  Demuxer& operator=(Demuxer&&) noexcept;
  ~Demuxer();

  // Pistes vidéo, audio, sous-titres et pièces jointes ; les flux de données sont omis.
  const std::vector<TrackInfo>& tracks() const;
  const std::vector<Chapter>& chapters() const;
  const TrackInfo* track(int index) const;  // nullptr si absente
  std::int64_t duration_us() const;         // 0 si inconnue

  // Ne lit plus que ces pistes ; le conteneur saute les autres.
  std::expected<void, Refusal> keep_only(std::span<const int> tracks);

  // Fait échouer, depuis n'importe quel thread, la lecture ou le seek en cours et toutes les
  // suivantes : la fermeture ne reste pas suspendue à un partage réseau qui ne répond plus. Sans
  // retour possible ; le Demuxer ne sert plus qu'à être détruit.
  void interrupt();

  // Paquet suivant dans l'ordre du fichier ; std::nullopt en fin de fichier.
  std::expected<std::optional<Packet>, Refusal> read();

  // Place la lecture sur un point clé de la piste `reference` proche de target_us.
  // Une cible négative ou au-delà de la durée connue est refusée. Les paquets lus
  // pour vérifier le point clé sont rendus ensuite par read(), dans l'ordre.
  std::expected<void, Refusal> seek(std::int64_t target_us, int reference, SeekTarget where);
  // Repositionnements du fichier (av_seek_frame) depuis l'ouverture : chacun coûte des lectures,
  // des allers-retours sur un partage réseau.
  std::uint32_t file_seeks() const;

 private:
  struct State;
  explicit Demuxer(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

}  // namespace ajp::media
