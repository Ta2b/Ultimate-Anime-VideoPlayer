#include <memory>
#include <string>
#include <tuple>

#include <catch2/catch_test_macros.hpp>

#include "subs/subs_messages.h"
#include "subs/subs_worker.h"
#include "support.h"

using ajp::subs::AtlasReturned;
using ajp::subs::RenderRequest;
using ajp::subs::SubsInbox;
using ajp::subs::SubsWorker;
using ajp::subs::test::RecordingSink;

namespace {
const char* kHeader =
    "[Script Info]\nScriptType: v4.00+\nPlayResX: 640\nPlayResY: 360\n\n[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
    "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
    "MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,"
    "&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1\n\n[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

struct H {
  RecordingSink sink;
  std::shared_ptr<SubsInbox> inbox = std::make_shared<SubsInbox>(ajp::subs::kSubsInboxBounds);
  std::unique_ptr<SubsWorker> worker;
  std::uint32_t epoch = 0;
  H() {
    auto created = SubsWorker::create({}, sink, [box = inbox](std::unique_ptr<ajp::subs::SubtitleAtlas> a) {
      std::ignore = box->try_push(AtlasReturned{std::move(a)});
    });
    REQUIRE(created);
    worker = std::move(*created);
    worker->handle(ajp::subs::SelectTrack{.id = 1, .source = ajp::subs::AssStream{kHeader}});
  }
  void packet(int order, std::int64_t start_ms, std::int64_t dur_ms, const std::string& text) {
    worker->handle(
        ajp::subs::TrackPacket{.track = 1,
                               .packet = {.start_us = start_ms * 1000,
                                          .duration_us = dur_ms * 1000,
                                          .data = std::to_string(order) + ",0,Default,,0,0,0,," + text}});
  }
  bool shown(double s) {
    worker->handle(RenderRequest{.pts_us = static_cast<std::int64_t>(s * 1e6),
                                 .epoch = epoch,
                                 .target = {.width = 640, .height = 360}});
    auto frames = sink.take_frames();
    REQUIRE(frames.size() == 1);
    const bool drawn = frames.front().atlas && !frames.front().rects.empty();
    if (frames.front().atlas) std::ignore = inbox->try_push(AtlasReturned{});  // rien
    return drawn;
  }
};
}  // namespace

TEST_CASE("Piste en flux : une réplique qui arrive après la demande s'affiche à la demande suivante",
          "[worker][lost]") {
  H h;
  h.packet(0, 1000, 2000, "Un");
  CHECK(h.shown(2.0));
  CHECK_FALSE(h.shown(10.0));
  CHECK_FALSE(h.shown(10.04));
  h.packet(1, 8000, 6000, "Longue");  // livrée en retard, commencée 2 s avant
  CHECK(h.shown(10.08));
  CHECK(h.shown(10.12));
}

TEST_CASE("Piste en flux : seek dans une réplique déjà connue", "[worker][lost]") {
  H h;
  h.packet(0, 1000, 2000, "Un");
  h.packet(1, 8000, 6000, "Longue");
  CHECK(h.shown(9.0));
  CHECK(h.shown(9.04));
  h.epoch = 1;
  h.worker->handle(ajp::Flush{1});
  CHECK(h.shown(2.0));
  CHECK_FALSE(h.shown(5.0));
  CHECK(h.shown(10.0));
}
