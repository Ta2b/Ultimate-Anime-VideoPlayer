// Boîte du thread subs et ses messages.
//
// Protocole :
// - RenderRequest : render demande l'image de sous-titres d'un pts de sortie ;
//   une réponse par demande, dans l'ordre, au plus kMaxPendingRequests en cours.
// - SelectTrack : app (piste externe, aucune) ou media (piste du conteneur)
//   choisit la piste ; les paquets d'une autre piste sont ignorés.
// - TrackPacket : media livre un paquet de la piste choisie.
// - AttachFont : media livre une police jointe, valable pour toutes les pistes,
//   de préférence avant SelectTrack (sinon le moteur est recréé une fois).
//   Pour ces deux messages, si la borne est atteinte, le producteur retente plus tard.
// - Command : cmd::Quit termine la boucle ; les autres ne concernent pas subs.
// - Flush{e} : les demandes d'époque < e sont jetées sans réponse ; e croît.
//   Les événements de la piste restent : libass écarte les doublons revus.
// - AtlasReturned : un atlas n'a plus de lecteur et rentre dans la réserve.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "core/inbox.h"
#include "core/messages.h"
#include "subs/subtitle_frame.h"
#include "subs/subtitle_renderer.h"
#include "subs/subtitle_track.h"

namespace ajp::subs {

struct NoTrack {};
struct AssStream {
  std::string header;  // CodecPrivate du conteneur
};
struct TextStream {};
struct ExternalFile {
  std::filesystem::path path;  // .ass, .ssa ou .srt
};
using TrackSource = std::variant<NoTrack, AssStream, TextStream, ExternalFile>;

struct SelectTrack {
  std::uint32_t id = 0;  // choisi par l'émetteur, repris par les paquets de cette piste
  TrackSource source;
};

struct TrackPacket {
  std::uint32_t track = 0;
  SubtitlePacket packet;
};

struct AttachFont {
  std::string name;  // nom de la pièce jointe
  std::vector<std::byte> data;
};

struct AtlasReturned {
  std::unique_ptr<SubtitleAtlas> atlas;
};

using SubsInbox = Inbox<RenderRequest, SelectTrack, TrackPacket, AttachFont, Command, Flush, AtlasReturned>;

// Bornes, dans l'ordre des types. AtlasReturned ne peut pas déborder : il
// n'existe jamais plus de kMaxAtlasesInFlight atlas.
inline constexpr SubsInbox::Bounds kSubsInboxBounds{kMaxPendingRequests, 4, 1024, 64, 16, 4,
                                                    kMaxAtlasesInFlight};

}  // namespace ajp::subs
