#include <memory>
#include <stop_token>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "subs/subs_messages.h"
#include "subs/subs_worker.h"
#include "support.h"

using ajp::Flush;
using ajp::RefusalCode;
using ajp::subs::AtlasReturned;
using ajp::subs::RenderRequest;
using ajp::subs::SubsInbox;
using ajp::subs::SubsWorker;
using ajp::subs::SubtitleFrame;
using ajp::subs::test::RecordingSink;

namespace {

RenderRequest at(double seconds, std::uint32_t epoch = 0) {
  return {.pts_us = static_cast<std::int64_t>(seconds * 1e6),
          .epoch = epoch,
          .target = {.width = 640, .height = 360}};
}

// Le worker sans son thread : la boîte reçoit les atlas rendus, le test les lui passe.
struct Harness {
  RecordingSink sink;
  std::shared_ptr<SubsInbox> inbox = std::make_shared<SubsInbox>(ajp::subs::kSubsInboxBounds);
  std::unique_ptr<SubsWorker> worker;

  Harness() {
    auto created =
        SubsWorker::create({}, sink, [box = inbox](std::unique_ptr<ajp::subs::SubtitleAtlas> atlas) {
          box->try_push(AtlasReturned{std::move(atlas)});
        });
    REQUIRE(created);
    worker = std::move(*created);
  }

  bool send(SubsInbox::Message message) { return worker->handle(std::move(message)); }

  // Passe au worker les atlas revenus dans la boîte.
  void pump() {
    std::stop_source stop;
    while (inbox->pending<AtlasReturned>() > 0) {
      auto message = inbox->wait(stop.get_token());
      REQUIRE(message);
      send(std::move(*message));
    }
  }

  void select_script(std::string_view script, std::uint32_t id = 1) {
    const ajp::subs::test::TempFile file("piste.ass", script);
    send(ajp::subs::SelectTrack{.id = id, .source = ajp::subs::ExternalFile{file.path()}});
  }
};

}  // namespace

TEST_CASE("SubsWorker : une réponse par demande, vide sans piste", "[worker]") {
  Harness harness;
  REQUIRE(harness.send(at(0.5)));
  auto frames = harness.sink.take_frames();
  REQUIRE(frames.size() == 1);
  CHECK(frames.front().atlas == nullptr);
  CHECK(frames.front().pts_us == 500'000);
}

TEST_CASE("SubsWorker : piste externe rendue aux pts demandés", "[worker]") {
  Harness harness;
  harness.select_script(ajp::subs::test::reference_script());
  harness.send(at(0.5));
  harness.send(at(2.5));
  harness.send(at(9.0));
  auto frames = harness.sink.take_frames();
  REQUIRE(frames.size() == 3);
  ajp::subs::test::check_atlas(frames[0]);
  ajp::subs::test::check_atlas(frames[1]);
  CHECK(frames[2].atlas == nullptr);
  CHECK(harness.sink.refusals().empty());
}

TEST_CASE("SubsWorker : le 9e rendu attend qu'un atlas revienne de render", "[worker]") {
  Harness harness;
  harness.select_script(ajp::subs::test::numbered_script(12));
  for (int i = 0; i < 9; ++i) harness.send(at(i + 0.5));
  std::vector<SubtitleFrame> held = harness.sink.take_frames();
  REQUIRE(held.size() == ajp::subs::kMaxAtlasesInFlight);
  CHECK(harness.worker->waiting() == 1);
  CHECK(harness.worker->atlases_in_flight() == ajp::subs::kMaxAtlasesInFlight);

  harness.send(at(9.5));  // la suivante attend derrière, dans l'ordre
  CHECK(harness.worker->waiting() == 2);

  held.erase(held.begin());
  harness.pump();
  auto served = harness.sink.take_frames();
  REQUIRE(served.size() == 1);
  CHECK(served.front().pts_us == 8'500'000);
  ajp::subs::test::check_atlas(served.front());
  CHECK(harness.worker->waiting() == 1);

  held.clear();
  harness.pump();
  served = harness.sink.take_frames();
  REQUIRE(served.size() == 1);
  CHECK(served.front().pts_us == 9'500'000);
  CHECK(harness.worker->waiting() == 0);
}

TEST_CASE("SubsWorker : au-delà de kMaxPendingRequests, réponse vide et refus", "[worker]") {
  Harness harness;
  harness.select_script(ajp::subs::test::numbered_script(20));
  std::vector<SubtitleFrame> held;
  for (int i = 0; i < 8; ++i) harness.send(at(i + 0.5));
  held = harness.sink.take_frames();
  for (std::size_t i = 0; i < ajp::subs::kMaxPendingRequests; ++i) harness.send(at(10.5));
  CHECK(harness.worker->waiting() == ajp::subs::kMaxPendingRequests);
  harness.send(at(11.5));
  auto frames = harness.sink.take_frames();
  REQUIRE(frames.size() == 1);
  CHECK(frames.front().atlas == nullptr);
  REQUIRE(harness.sink.refusals().size() == 1);
  CHECK(harness.sink.refusals().front().code == RefusalCode::SubtitleBacklog);
}

TEST_CASE("SubsWorker : Flush jette les demandes d'une époque passée", "[worker]") {
  Harness harness;
  harness.select_script(ajp::subs::test::numbered_script(12));
  for (int i = 0; i < 9; ++i) harness.send(at(i + 0.5));
  std::vector<SubtitleFrame> held = harness.sink.take_frames();
  CHECK(harness.worker->waiting() == 1);

  harness.send(Flush{1});
  CHECK(harness.worker->waiting() == 0);
  harness.send(at(0.5, 0));  // arrivée en retard : jetée sans réponse
  CHECK(harness.sink.frame_count() == 0);
  held.clear();
  harness.pump();
  harness.send(at(3.5, 1));
  auto frames = harness.sink.take_frames();
  REQUIRE(frames.size() == 1);
  CHECK(frames.front().epoch == 1);

  harness.send(Flush{1});  // une époque ne recule pas
  REQUIRE_FALSE(harness.sink.refusals().empty());
  CHECK(harness.sink.refusals().back().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("SubsWorker : paquets de la piste choisie seulement", "[worker]") {
  Harness harness;
  harness.send(ajp::subs::SelectTrack{.id = 7, .source = ajp::subs::TextStream{}});
  harness.send(
      ajp::subs::TrackPacket{.track = 7, .packet = {.start_us = 0, .duration_us = 1'000'000, .data = "Oui"}});
  harness.send(ajp::subs::TrackPacket{
      .track = 3, .packet = {.start_us = 2'000'000, .duration_us = 1'000'000, .data = "Non"}});
  harness.send(at(0.5));
  harness.send(at(2.5));
  auto frames = harness.sink.take_frames();
  REQUIRE(frames.size() == 2);
  CHECK(frames[0].atlas != nullptr);
  CHECK(frames[1].atlas == nullptr);

  harness.send(ajp::subs::TrackPacket{.track = 7, .packet = {.start_us = 0, .duration_us = -5, .data = "x"}});
  REQUIRE(harness.sink.refusals().size() == 1);
  CHECK(harness.sink.refusals().front().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("SubsWorker : une piste illisible est refusée, la suivante fonctionne", "[worker]") {
  Harness harness;
  harness.send(ajp::subs::SelectTrack{.id = 1, .source = ajp::subs::ExternalFile{"/absent/piste.srt"}});
  REQUIRE(harness.sink.refusals().size() == 1);
  CHECK(harness.sink.refusals().front().code == RefusalCode::SubtitleUnreadable);
  harness.send(
      ajp::subs::SelectTrack{.id = 2, .source = ajp::subs::AssStream{"[Script Info]\nScriptType: v4.00+\n"}});
  harness.send(ajp::subs::TrackPacket{
      .track = 2, .packet = {.start_us = 0, .duration_us = 1'000'000, .data = "0,0,Default,,0,0,0,,A"}});
  harness.send(at(0.5));
  auto frames = harness.sink.take_frames();
  REQUIRE(frames.size() == 1);
  CHECK(frames.front().atlas != nullptr);
}

TEST_CASE("SubsWorker : police jointe sans police système", "[worker]") {
  const std::vector<std::byte> font = ajp::subs::test::system_font_bytes();
  if (font.empty()) SKIP("DejaVu Sans absente du système");
  RecordingSink sink;
  const auto discard = [](std::unique_ptr<ajp::subs::SubtitleAtlas>) {};
  auto worker = SubsWorker::create({.fonts = ajp::subs::FontProvider::None}, sink, discard);
  REQUIRE(worker);
  const ajp::subs::test::TempFile file("piste.ass", ajp::subs::test::reference_script());
  (*worker)->handle(ajp::subs::SelectTrack{.id = 1, .source = ajp::subs::ExternalFile{file.path()}});

  (*worker)->handle(at(0.5));
  (*worker)->handle(ajp::subs::AttachFont{.name = "DejaVuSans.ttf", .data = font});
  (*worker)->handle(at(0.6));
  auto frames = sink.take_frames();
  REQUIRE(frames.size() == 2);
  CHECK(frames[0].atlas == nullptr);  // aucune police connue : rien à dessiner
  ajp::subs::test::check_atlas(frames[1]);
}

TEST_CASE("SubsWorker : cmd::Quit arrête la boucle, les autres commandes non", "[worker]") {
  Harness harness;
  CHECK(harness.send(ajp::Command{ajp::cmd::Pause{}}));
  CHECK_FALSE(harness.send(ajp::Command{ajp::cmd::Quit{}}));
}
