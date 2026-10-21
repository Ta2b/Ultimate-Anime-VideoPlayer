#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "subs/subs_messages.h"
#include "subs/subs_thread.h"
#include "support.h"

using ajp::subs::RenderRequest;
using ajp::subs::SubsInbox;
using ajp::subs::SubsThread;
using ajp::subs::SubtitleFrame;
using ajp::subs::test::RecordingSink;
using namespace std::chrono_literals;

namespace {

constexpr auto kTimeout = 10s;

RenderRequest at(double seconds) {
  return {.pts_us = static_cast<std::int64_t>(seconds * 1e6),
          .epoch = 0,
          .target = {.width = 640, .height = 360}};
}

}  // namespace

TEST_CASE("SubsThread : rend sur son thread et rend la main à cmd::Quit", "[thread]") {
  RecordingSink sink;
  auto inbox = std::make_shared<SubsInbox>(ajp::subs::kSubsInboxBounds);
  auto thread = SubsThread::start({}, sink, inbox);
  REQUIRE(thread);

  const ajp::subs::test::TempFile file("piste.ass", ajp::subs::test::reference_script());
  REQUIRE(inbox->try_push(ajp::subs::SelectTrack{.id = 1, .source = ajp::subs::ExternalFile{file.path()}}));
  REQUIRE(inbox->try_push(at(0.5)));
  REQUIRE(inbox->try_push(at(0.7)));
  REQUIRE(sink.wait_for_frames(2, kTimeout));
  const auto frames = sink.take_frames();
  ajp::subs::test::check_atlas(frames[0]);
  CHECK(frames[1].atlas == frames[0].atlas);  // inchangé : même atlas

  REQUIRE(inbox->try_push(ajp::Command{ajp::cmd::Quit{}}));
  thread->reset();  // attend la fin du thread
}

TEST_CASE("SubsThread : le 9e rendu attend le retour d'un atlas tenu par render", "[thread]") {
  RecordingSink sink;
  auto inbox = std::make_shared<SubsInbox>(ajp::subs::kSubsInboxBounds);
  auto thread = SubsThread::start({}, sink, inbox);
  REQUIRE(thread);

  const ajp::subs::test::TempFile file("piste.ass", ajp::subs::test::numbered_script(10));
  REQUIRE(inbox->try_push(ajp::subs::SelectTrack{.id = 1, .source = ajp::subs::ExternalFile{file.path()}}));
  // Au plus kMaxPendingRequests demandes sans réponse : huit, puis la 9e.
  for (int i = 0; i < 8; ++i) REQUIRE(inbox->try_push(at(i + 0.5)));
  REQUIRE(sink.wait_for_frames(8, kTimeout));
  std::vector<SubtitleFrame> held = sink.take_frames();
  REQUIRE(inbox->try_push(at(8.5)));
  std::this_thread::sleep_for(100ms);
  CHECK(sink.frame_count() == 0);  // le 9e attend

  held.erase(held.begin());  // render relâche une image : son atlas rentre par la boîte
  REQUIRE(sink.wait_for_frames(1, kTimeout));
  const auto ninth = sink.take_frames();
  CHECK(ninth.front().pts_us == 8'500'000);
  ajp::subs::test::check_atlas(ninth.front());
}

TEST_CASE("SubsThread : l'arrêt du std::jthread réveille la boucle", "[thread]") {
  RecordingSink sink;
  auto inbox = std::make_shared<SubsInbox>(ajp::subs::kSubsInboxBounds);
  auto thread = SubsThread::start({}, sink, inbox);
  REQUIRE(thread);
  const auto begin = std::chrono::steady_clock::now();
  thread->reset();
  CHECK(std::chrono::steady_clock::now() - begin < kTimeout);
}

TEST_CASE("SubsThread : un atlas relâché après le thread est simplement détruit", "[thread]") {
  RecordingSink sink;
  std::vector<SubtitleFrame> held;
  {
    auto inbox = std::make_shared<SubsInbox>(ajp::subs::kSubsInboxBounds);
    auto thread = SubsThread::start({}, sink, inbox);
    REQUIRE(thread);
    const ajp::subs::test::TempFile file("piste.ass", ajp::subs::test::reference_script());
    REQUIRE(inbox->try_push(ajp::subs::SelectTrack{.id = 1, .source = ajp::subs::ExternalFile{file.path()}}));
    REQUIRE(inbox->try_push(at(0.5)));
    REQUIRE(sink.wait_for_frames(1, kTimeout));
    held = sink.take_frames();
  }
  REQUIRE(held.front().atlas != nullptr);
  held.clear();  // la boîte n'existe plus : aucun accès, aucune fuite (ASan)
}

TEST_CASE("SubsThread : refuse une boîte absente", "[thread]") {
  RecordingSink sink;
  CHECK(SubsThread::start({}, sink, nullptr).error().code == ajp::RefusalCode::FieldMissing);
}
