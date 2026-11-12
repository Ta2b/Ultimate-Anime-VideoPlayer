#include "app/play/player_probe.h"

#include <format>
#include <string>
#include <utility>

#include "app/play/visible_error.h"
#include "core/log.h"
#include "core/utf8_path.h"
#include "media/demuxer.h"
#include "media/selection.h"

namespace ajp::app::play {

// La piste vidéo que la source choisira, lue d'avance : sa taille fixe le moteur et le ring.
std::expected<media::TrackInfo, Refusal> probe_once(const std::filesystem::path& file) {
  auto demuxer = media::Demuxer::open(file, {.attachments = false});
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  const media::TrackChoice choice = media::choose_tracks(demuxer->tracks(), {});
  if (!choice.video) {
    // P3.28 : un fichier audio seul est dit tel quel, pas comme un fichier abîmé.
    Refusal refusal{RefusalCode::TrackUnavailable,
                    std::format("{} : aucune piste vidéo", core::to_utf8(file))};
    if (choice.audio) refusal.key = kAudioOnlyKey;
    return std::unexpected(std::move(refusal));
  }
  const media::TrackInfo& video = *demuxer->track(*choice.video);
  // P3.26 : le profil 5 n'a pas de couche de base HDR10 ou SDR (IPTPQc2) ; les profils 7 et 8 se lisent par
  // leur couche de base, sans les métadonnées dynamiques.
  if (video.dolby_vision_profile == 5) {
    Refusal refusal{RefusalCode::FormatUnsupported,
                    std::format("{} : Dolby Vision profil 5", core::to_utf8(file))};
    refusal.key = kDolbyVision5Key;
    return std::unexpected(std::move(refusal));
  }
  // A15-07 : un format ou une couleur que la chaîne refuserait se dit ici, avant de construire la chaîne.
  if (std::optional<Refusal> unreadable = media::unreadable_video(video)) {
    unreadable->detail = std::format("{} : {}", core::to_utf8(file.filename()), unreadable->detail);
    return std::unexpected(std::move(*unreadable));
  }
  if (video.dolby_vision_profile > 0)
    log_channel("app")->info("Dolby Vision profil {} : lu par sa couche de base, sans RPU",
                             video.dolby_vision_profile);
  return video;
}

std::expected<media::TrackInfo, Refusal> probe_video(const std::filesystem::path& file, bool window_open) {
  return run_pumped_for<media::TrackInfo>("probe", kProbeLimit, window_open,
                                          [file] { return probe_once(file); });
}

std::optional<GivenProbe> start_given_probe(const std::filesystem::path& file) {
  auto started = GivenProbe::start("probe", true, [file] { return probe_once(file); });
  if (!started) {
    log_channel("app")->warn("{} ; sondé à l'ouverture", started.error().detail);
    return std::nullopt;
  }
  return std::move(*started);
}

std::expected<media::TrackInfo, Refusal> finish_given_probe(GivenProbe& probe) {
  return probe.wait(kProbeLimit, true, true);
}

}  // namespace ajp::app::play
