// A17-05, A17-06 : la piste de sous-titres relue en avance par un second démultiplexeur. Le clip est écrit
// par le test (MPEG-4, un point clé toutes les 4 s, ASS) : aucun binaire dans le dépôt.
#include "app/play/subtitle_lead.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <thread>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "clip_writer.h"
#include "media/demuxer.h"

namespace ajp::app::play {
namespace {

using namespace std::chrono_literals;

constexpr int kVideo = 0;
constexpr int kSubtitles = 1;

// 12 s à 25 im/s, points clés à 0, 4 et 8 s. La réplique « Longue » commence à 1 s et dure jusqu'à 7 s : un
// seek à 6 s repart du point clé de 4 s, après son début.
std::filesystem::path clip() {
  static const auto path = media::test::write_clip(media::test::ClipSpec{
      .name = "app-subtitle-lead.mkv",
      .codec = media::test::VideoCodec::Mpeg4,
      .frames = 300,
      .gop = 100,
      .subtitles = {{.language = "fre",
                     .is_default = true,
                     .events = {{.start_ms = 1000, .duration_ms = 6000, .text = "Longue"},
                                {.start_ms = 6500, .duration_ms = 500, .text = "Six"},
                                {.start_ms = 9000, .duration_ms = 500, .text = "Neuf"},
                                {.start_ms = 11000, .duration_ms = 500, .text = "Onze"}}}},
  });
  REQUIRE(path);
  return *path;
}

// 100 s à 25 im/s, un point clé toutes les 4 s. Un générique défile de 5 s à 95 s ; une réplique courte
// vient à 20 s et 80 s, et rien entre 25 s et 75 s.
std::filesystem::path long_clip() {
  static const auto path = media::test::write_clip(media::test::ClipSpec{
      .name = "app-subtitle-lead-long.mkv",
      .codec = media::test::VideoCodec::Mpeg4,
      .frames = 2500,
      .gop = 100,
      .subtitles = {{.language = "fre",
                     .is_default = true,
                     .events = {{.start_ms = 5000, .duration_ms = 90000, .text = "Generique"},
                                {.start_ms = 20000, .duration_ms = 1000, .text = "Vingt"},
                                {.start_ms = 80000, .duration_ms = 1000, .text = "Quatre-vingts"}}}},
      .interleave_subtitles = true,
  });
  REQUIRE(path);
  return *path;
}

struct Rig {
  explicit Rig(std::filesystem::path file = clip())
      : lead(SubtitleLeadConfig{.file = std::move(file), .reference = kVideo, .inbox = inbox}) {}
  std::shared_ptr<subs::SubsInbox> inbox = std::make_shared<subs::SubsInbox>(subs::kSubsInboxBounds);
  SubtitleLead lead;

  // Le thread a tout lu : les débuts des paquets reçus par subs, dans l'ordre.
  std::vector<std::int64_t> starts() {
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!lead.idle() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(1ms);
    REQUIRE(lead.idle());
    std::vector<std::int64_t> out;
    while (auto message = inbox->try_pop()) {
      if (const auto* packet = std::get_if<subs::TrackPacket>(&*message)) {
        CHECK(packet->track == static_cast<std::uint32_t>(kSubtitles) + 1U);
        out.push_back(packet->packet.start_us);
      }
    }
    return out;
  }
};

bool has(const std::vector<std::int64_t>& starts, std::int64_t start_us) {
  return std::ranges::contains(starts, start_us);
}

}  // namespace

// Avant : la source repart du point clé de 4 s ; la réplique commencée à 1 s n'arrivait jamais.
TEST_CASE("Sous-titres en avance : la source seule perd la réplique commencée avant le point clé",
          "[app][subs]") {
  auto demuxer = media::Demuxer::open(clip());
  REQUIRE(demuxer);
  REQUIRE(demuxer->seek(6'000'000, kVideo, media::SeekTarget::PreviousKeyframe));
  std::vector<std::int64_t> starts;
  while (auto packet = demuxer->read()) {
    if (!*packet || (*packet)->pts_us > 7'000'000) break;
    if ((*packet)->track == kSubtitles) starts.push_back((*packet)->pts_us);
  }
  CHECK(has(starts, 6'500'000));
  CHECK_FALSE(has(starts, 1'000'000));
}

TEST_CASE("Sous-titres en avance : un seek relit la réplique commencée avant le point clé", "[app][subs]") {
  Rig rig;
  rig.lead.select(kSubtitles);
  rig.lead.seek(6'000'000);
  const std::vector<std::int64_t> starts = rig.starts();
  CHECK(has(starts, 1'000'000));  // commencée 5 s avant la cible, elle dure jusqu'à 7 s
  CHECK(has(starts, 6'500'000));
  CHECK_FALSE(has(starts, 11'000'000));  // au-delà de la cible + 1 s, la source livre à temps
  CHECK(rig.lead.sent() == starts.size());
}

TEST_CASE("Sous-titres en avance : un délai négatif lit la piste d'autant en avance", "[app][subs]") {
  Rig rig;
  rig.lead.select(kSubtitles);
  // À 2 s avec −8 s, libass lit l'instant 10 s : la piste est lue jusqu'à 11 s.
  rig.lead.follow(2'000'000, -8'000'000);
  const std::vector<std::int64_t> starts = rig.starts();
  CHECK(has(starts, 9'000'000));
  CHECK(has(starts, 11'000'000));
}

TEST_CASE("Sous-titres en avance : rien sans délai négatif, sans seek ni sans piste", "[app][subs]") {
  Rig rig;
  rig.lead.select(kSubtitles);
  rig.lead.follow(2'000'000, 500'000);  // un délai positif : la source livre à temps
  CHECK(rig.starts().empty());
  rig.lead.select(std::nullopt);  // un fichier externe, lu en entier
  rig.lead.seek(6'000'000);
  rig.lead.follow(2'000'000, -8'000'000);
  CHECK(rig.starts().empty());
  CHECK(rig.lead.sent() == 0);
}

// Un changement de piste attend le seek qui le suit (Pipeline::select_subtitles), puis relit la nouvelle.
TEST_CASE("Sous-titres en avance : la piste choisie après un seek se relit au seek suivant", "[app][subs]") {
  Rig rig;
  rig.lead.seek(6'000'000);
  CHECK(rig.starts().empty());
  rig.lead.select(kSubtitles);
  rig.lead.seek(6'000'000);
  CHECK(has(rig.starts(), 1'000'000));
}

// B17-07 (a) : un générique commencé 55 s avant la cible, au-delà de la préroulée de 10 s, est relu en
// second.
TEST_CASE("Sous-titres en avance : un seek relit aussi la réplique commencée bien avant la préroulée",
          "[app][subs]") {
  Rig rig(long_clip());
  rig.lead.select(kSubtitles);
  rig.lead.seek(60'000'000);
  const std::vector<std::int64_t> starts = rig.starts();
  CHECK(has(starts, 5'000'000));  // 55 s avant la cible, encore à l'écran
  CHECK(has(starts, 20'000'000));
}

// Une plage déjà lue n'est pas relue : un second seek tout près ne renvoie rien de plus.
TEST_CASE("Sous-titres en avance : une plage déjà relue n'est pas relue au seek voisin", "[app][subs]") {
  Rig rig(long_clip());
  rig.lead.select(kSubtitles);
  rig.lead.seek(60'000'000);
  REQUIRE_FALSE(rig.starts().empty());
  const std::uint64_t first = rig.lead.sent();
  rig.lead.seek(62'000'000);
  const std::vector<std::int64_t> again = rig.starts();
  CHECK(rig.lead.sent() - first == again.size());
  CHECK_FALSE(has(again, 5'000'000));  // déjà envoyée
}

// B17-07 (b) : sans réplique avant la cible, la lecture s'arrête dans la vidéo au-delà de ce qui est voulu :
// le démultiplexeur ne remonte plus tout le fichier jusqu'au paquet de sous-titres suivant.
TEST_CASE("Sous-titres en avance : la lecture n'attend pas le paquet de sous-titres suivant", "[app][subs]") {
  Rig rig(long_clip());
  rig.lead.select(kSubtitles);
  rig.lead.seek(40'000'000);  // la réplique suivante est à 80 s
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!rig.lead.idle() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(1ms);
  REQUIRE(rig.lead.idle());
  std::vector<std::int64_t> starts;
  while (auto message = rig.inbox->try_pop()) {
    if (const auto* packet = std::get_if<subs::TrackPacket>(&*message))
      starts.push_back(packet->packet.start_us);
  }
  CHECK(has(starts, 5'000'000));
  CHECK(has(starts, 20'000'000));
  CHECK_FALSE(has(starts, 80'000'000));  // lue avant : le fichier mélange les pistes par horodatage
}

}  // namespace ajp::app::play
