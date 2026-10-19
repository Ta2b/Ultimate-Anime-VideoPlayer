// La boucle complète avec les vrais threads source, audio (pilote SDL « dummy ») et subs,
// un thread gpu de substitution et un écran sans fenêtre qui présente toutes les 5 ms.
// Le clip est généré dans le test (FFV1, PCM, ASS) : aucun binaire dans le dépôt.
#include "app/play/pipeline.h"

#include <algorithm>
#include <chrono>
#include <thread>

#include <SDL3/SDL_hints.h>
#include <SDL3/SDL_init.h>
#include <catch2/catch_test_macros.hpp>

#include "app/play/session.h"
#include "clip_writer.h"
#include "doubles.h"

namespace ajp::app::play {
namespace {

using namespace std::chrono_literals;

// Écran sans fenêtre au rythme d'un écran à 200 Hz, en temps réel.
struct RealtimeScreen final : Screen {
  test::FakeScreen record;
  OutputSize size() const override { return {640, 360}; }
  std::expected<ScreenTiming, Refusal> show(const ShowRequest& request) override {
    std::this_thread::sleep_for(5ms);
    (void)record.show(request);
    return ScreenTiming{.render_us = 100, .present_us = host_now_us()};
  }
  SlotFree retire(const RingFrame& frame) override { return record.retire(frame); }
  void set_fullscreen(bool on) override { record.set_fullscreen(on); }
  bool fullscreen() const override { return record.fullscreen(); }
};

std::filesystem::path clip() {
  static const auto path = media::test::write_clip(media::test::ClipSpec{
      .name = "app-pipeline.mkv",
      .frames = 50,
      .audio = {{.language = "jpn", .is_default = true}},
      .subtitles = {{.language = "fre",
                     .is_default = true,
                     .events = {{.start_ms = 0, .duration_ms = 1500, .text = "Bonjour"}}}},
  });
  REQUIRE(path);
  return *path;
}

struct Playing {
  std::unique_ptr<Pipeline> pipeline;
  RealtimeScreen screen;
  std::optional<Session> session;

  Playing() {
    SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "dummy");
    REQUIRE(SDL_InitSubSystem(SDL_INIT_AUDIO));
    auto opened =
        Pipeline::open(PipelineConfig{.file = clip(), .decoder = {.attempts = {media::HwApi::Software}}},
                       [](VideoStageInit init) { return test::PassthroughStage::start(std::move(init), 4); });
    REQUIRE(opened);
    pipeline = std::move(*opened);
    session.emplace(pipeline->session_config(), pipeline->stages(), screen, pipeline->clock());
    session->begin(host_now_us());
  }
  ~Playing() {
    session->end();
    pipeline.reset();
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
  }

  // Tours de main jusqu'à la fin, ou jusqu'à until() ; faux si le délai passe.
  template <class Until>
  bool run(Until until, std::chrono::seconds limit = 20s) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!session->finished() && !until()) {
      if (std::chrono::steady_clock::now() > deadline) return false;
      pipeline->pump();
      while (auto message = try_take(pipeline->inbox())) session->handle(std::move(*message));
      REQUIRE(session->tick(host_now_us()));
    }
    return true;
  }
};

std::vector<RingFrame> shown_frames(const RealtimeScreen& screen) {
  std::vector<RingFrame> frames;
  for (const auto& shown : screen.record.shown) {
    if (shown.frame && (frames.empty() || frames.back().ready_value != shown.frame->ready_value ||
                        frames.back().slot != shown.frame->slot)) {
      frames.push_back(*shown.frame);
    }
  }
  return frames;
}

TEST_CASE("Lecture de bout en bout : images dans l'ordre, son maître, sous-titres, fin", "[app][pipeline]") {
  Playing playing;
  REQUIRE(playing.run([] { return false; }));
  CHECK(playing.session->finished());

  const auto frames = shown_frames(playing.screen);
  const auto& stats = playing.session->stats();
  CHECK(frames.size() + static_cast<std::size_t>(stats.dropped) == 50);
  CHECK(stats.dropped <= 5);
  for (std::size_t i = 1; i < frames.size(); ++i) CHECK(frames[i].pts_us > frames[i - 1].pts_us);
  bool subtitled = false;
  for (const auto& shown : playing.screen.record.shown)
    subtitled = subtitled || shown.subtitles_pts.has_value();
  CHECK(subtitled);
  // Le son mène : la dernière image paraît environ 2 s après la première.
  CHECK(stats.frame_interval.median() > 30'000);
  CHECK(stats.frame_interval.median() < 50'000);
}

TEST_CASE("Seek en cours de lecture : nouvelle époque, images de la cible", "[app][pipeline]") {
  Playing playing;
  REQUIRE(playing.run([&] { return playing.session->stats().new_frames >= 5; }));
  playing.session->command(cmd::Seek{.target_us = 1'200'000});
  REQUIRE(playing.run([&] {
    const auto frames = shown_frames(playing.screen);
    return !frames.empty() && frames.back().epoch == 1 && frames.back().pts_us >= 1'400'000;
  }));
  const auto frames = shown_frames(playing.screen);
  const auto first_new = std::ranges::find_if(frames, [](const RingFrame& f) { return f.epoch == 1; });
  REQUIRE(first_new != frames.end());
  CHECK(first_new->pts_us == 1'200'000);
  REQUIRE(playing.run([] { return false; }));  // puis la fin du fichier
}

}  // namespace
}  // namespace ajp::app::play
