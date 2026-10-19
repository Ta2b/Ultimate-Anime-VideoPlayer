// Déterminisme (framemd5) et thread export nourri par sa boîte, crédit par crédit.
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <variant>

#include <catch2/catch_test_macros.hpp>

#include "core/inbox.h"
#include "export/export_thread.h"
#include "export/framemd5.h"
#include "support.h"

namespace ajp::exporter {
namespace {

using namespace test;

constexpr FrameRate kRate{24000, 1001};

std::size_t frame_lines(const std::string& text) {
  std::istringstream lines(text);
  std::size_t count = 0;
  for (std::string line; std::getline(lines, line);)
    count += static_cast<std::size_t>(!line.empty() && line[0] != '#');
  return count;
}

std::string export_md5(const std::filesystem::path& output, const std::vector<OwnedFrame>& frames,
                       PixelFormat format) {
  VectorSource source(frames);
  ExportJob job = ffv1_job(output, frames[0].width, frames[0].height, source);
  job.encoder.format = format;
  REQUIRE(run_export(job).has_value());
  auto md5 = framemd5_of_file(output);
  REQUIRE(md5.has_value());
  return *md5;
}

// La boîte du producteur dans les tests : crédits rendus, fin de l'export.
struct Finished {
  std::optional<ExportSummary> summary;
  std::optional<Refusal> refusal;
};
using ProducerInbox = Inbox<Credit, Finished>;

// Les rappels tournent sur le thread export : pas d'assertion Catch2 là, les bornes de la
// boîte (16 crédits, une fin) suffisent aux tests.
ExportThreadHooks hooks_for(ProducerInbox& producer) {
  return ExportThreadHooks{
      .credit = [&producer](Credit credit) { producer.try_push(credit); },
      .done = [&producer](const ExportSummary& s) { producer.try_push(Finished{.summary = s}); }};
}

ThreadReport report_for(ProducerInbox& producer) {
  return ThreadReport([&producer](const Refusal& r) { producer.try_push(Finished{.refusal = r}); });
}

ExportFrameMsg message_of(const OwnedFrame& frame) {
  auto owned = std::make_shared<const OwnedFrame>(frame);
  return ExportFrameMsg{.frame = owned->view(), .keep_alive = owned};
}

}  // namespace

TEST_CASE("framemd5 identique sur deux exports identiques", "[export][framemd5]") {
  const auto dir = work_dir("framemd5");
  const auto frames = pattern_frames(48, 32, 5, kRate);
  for (PixelFormat format : {PixelFormat::Rgb16, PixelFormat::Yuv420p10}) {
    const std::string first = export_md5(dir / "a.mkv", frames, format);
    const std::string second = export_md5(dir / "b.mkv", frames, format);
    CHECK(first == second);
    CHECK(first.starts_with("#format: frame checksums"));
    CHECK(frame_lines(first) == 5);

    auto changed = frames;
    changed[3].pixels[100] ^= 0x0400;
    CHECK(export_md5(dir / "c.mkv", changed, format) != first);
  }
}

TEST_CASE("framemd5 des images hôte : même texte pour les mêmes images", "[export][framemd5]") {
  const auto frames = pattern_frames(16, 8, 3, kRate);
  auto hash = [&](const std::vector<OwnedFrame>& input) {
    auto md5 = FrameMd5::create(16, 8);
    REQUIRE(md5.has_value());
    for (const auto& f : input) REQUIRE(md5->add(f.view()).has_value());
    auto text = md5->finish();
    REQUIRE(text.has_value());
    return *text;
  };
  const std::string text = hash(frames);
  CHECK(text == hash(frames));
  CHECK(text.find("#codec_id 0: rawvideo") != std::string::npos);
  CHECK(text.find("#dimensions 0: 16x8") != std::string::npos);
  CHECK(frame_lines(text) == 3);
  auto changed = frames;
  changed[0].pixels[0] ^= 1;
  CHECK(hash(changed) != text);

  auto md5 = FrameMd5::create(16, 8);
  REQUIRE(md5.has_value());
  CHECK(md5->add(pattern_frame(8, 8, 0, 0).view()).error().code == RefusalCode::InputMismatch);
  CHECK(FrameMd5::create(0, 8).error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("thread export : images par la boîte, un crédit par image rendue", "[export][thread]") {
  const auto dir = work_dir("thread");
  const auto frames = pattern_frames(32, 18, 8, kRate);
  ProducerInbox producer(ProducerInbox::Bounds{16, 1});
  ExportJob job{
      .output = dir / "out.mkv", .width = 32, .height = 18, .encoder = EncoderSettings{}, .frames_total = 8};
  auto thread = ExportThread::start(job, hooks_for(producer), report_for(producer));
  REQUIRE(thread.has_value());

  std::size_t in_flight = 0;
  std::size_t credits = 0;
  std::optional<Finished> finished;
  const std::stop_source never;
  for (const auto& frame : frames) {
    while (in_flight == kFramesInFlight) {
      auto message = producer.wait(never.get_token());
      REQUIRE(message.has_value());
      REQUIRE(std::holds_alternative<Credit>(*message));
      in_flight -= std::get<Credit>(*message).count;
      ++credits;
    }
    REQUIRE((*thread)->inbox().try_push(message_of(frame)));
    ++in_flight;
  }
  REQUIRE((*thread)->inbox().try_push(ExportEnd{}));
  while (!finished) {
    auto message = producer.wait(never.get_token());
    REQUIRE(message.has_value());
    if (auto* done = std::get_if<Finished>(&*message))
      finished = *done;
    else
      ++credits;
  }
  (*thread)->join();
  REQUIRE(finished->summary.has_value());
  CHECK(finished->summary->frames == 8);
  CHECK(credits == 8);
  CHECK(std::filesystem::exists(dir / "out.mkv"));
}

TEST_CASE("thread export : ExportCancel abandonne sans sortie", "[export][thread]") {
  const auto dir = work_dir("thread_cancel");
  const auto frames = pattern_frames(32, 18, 2, kRate);
  ProducerInbox producer(ProducerInbox::Bounds{16, 1});
  const ExportJob job{.output = dir / "out.mkv", .width = 32, .height = 18};
  auto thread = ExportThread::start(job, hooks_for(producer), report_for(producer));
  REQUIRE(thread.has_value());
  for (const auto& frame : frames) REQUIRE((*thread)->inbox().try_push(message_of(frame)));
  REQUIRE((*thread)->inbox().try_push(ExportCancel{}));
  (*thread)->join();

  std::optional<Finished> finished;
  const std::stop_source never;
  while (!finished) {
    auto message = producer.wait(never.get_token());
    REQUIRE(message.has_value());
    if (auto* done = std::get_if<Finished>(&*message)) finished = *done;
  }
  REQUIRE(finished->refusal.has_value());
  CHECK(finished->refusal->code == RefusalCode::Cancelled);
  CHECK(std::filesystem::is_empty(dir));
}

TEST_CASE("thread export : travail invalide refusé au démarrage", "[export][thread]") {
  ProducerInbox producer(ProducerInbox::Bounds{16, 1});
  const ExportJob odd{.output = "x.mkv",
                      .width = 33,
                      .height = 18,
                      .encoder = EncoderSettings{.format = PixelFormat::Yuv420p10}};
  CHECK(ExportThread::start(odd, hooks_for(producer), report_for(producer)).error().code ==
        RefusalCode::FieldOutOfRange);
  const ExportJob job{.output = "x.mkv", .width = 32, .height = 18};
  CHECK(ExportThread::start(job, ExportThreadHooks{}, report_for(producer)).error().code ==
        RefusalCode::FieldMissing);
  CHECK(ExportThread::start(job, hooks_for(producer), ThreadReport{}).error().code ==
        RefusalCode::FieldMissing);
}

}  // namespace ajp::exporter
