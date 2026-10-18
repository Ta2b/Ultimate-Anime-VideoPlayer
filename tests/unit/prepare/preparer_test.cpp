// Thread preparer avec un Backend factice qui implémente gpu/backend.h, puis
// de bout en bout avec TrtexecBuilder et le faux trtexec.
#include "prepare/preparer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/builder.h"
#include "prepare/sha256.h"
#include "support.h"

using namespace ajp::prepare;
using ajp::RefusalCode;
using namespace std::chrono_literals;
namespace gpu = ajp::gpu;

namespace {

// Résultats et progression reçus, dans le thread du test.
class Sink final : public PreparerSink {
 public:
  bool publish(const BuildProgress& progress) override {
    const std::scoped_lock lock(mutex_);
    progress_.push_back(progress);
    return true;
  }
  bool deliver(BuildDone&& done) override {
    const std::scoped_lock lock(mutex_);
    if (refusals_ > 0) {
      --refusals_;
      return false;  // done reste intact
    }
    done_.push_back(std::move(done));
    changed_.notify_all();
    return true;
  }
  // Attend n résultats, au plus 10 s.
  std::vector<BuildDone> wait_done(std::size_t n) {
    std::unique_lock lock(mutex_);
    changed_.wait_for(lock, 10s, [&] { return done_.size() >= n; });
    return done_;
  }
  std::vector<BuildProgress> progress() {
    const std::scoped_lock lock(mutex_);
    return progress_;
  }
  void refuse_next(int n) {
    const std::scoped_lock lock(mutex_);
    refusals_ = n;
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  std::vector<BuildProgress> progress_;
  std::vector<BuildDone> done_;
  int refusals_ = 0;
};

// Backend sans GPU : build() rend un Artifact nommé d'après le modèle.
class FakeBackend : public gpu::Backend {
 public:
  gpu::BackendCaps capabilities() const override { return {.name = "factice"}; }
  gpu::DeviceMatch match(const gpu::DeviceUuid&) const override { return gpu::DeviceMatch::Unknown; }

  std::expected<gpu::Artifact, ajp::Refusal> build(const gpu::ModelSpec& model, const gpu::ShapeSet&,
                                                   gpu::Progress& progress) override {
    const int now = ++running_;
    max_running_ = std::max(max_running_.load(), now);
    struct Leave {
      std::atomic<int>& running;
      ~Leave() { --running; }
    } leave{running_};
    ++calls_;
    progress.report(0.5F, "moitié");
    if (model.id == "exception") throw std::runtime_error("panne du backend");
    if (model.id == "refus") return std::unexpected(ajp::Refusal{RefusalCode::BuilderFailed, "refusé"});
    if (model.id == "long") {
      for (int i = 0; i < 1000 && !progress.stop.stop_requested(); ++i) std::this_thread::sleep_for(10ms);
      if (progress.stop.stop_requested())
        return std::unexpected(ajp::Refusal{RefusalCode::Cancelled, "arrêt"});
    }
    if (model.id == "lent") std::this_thread::sleep_for(30ms);
    progress.report(1.0F, "fini");
    return gpu::Artifact{.cache_key = "clé-" + model.id, .path = model.id + ".engine"};
  }

  std::expected<std::unique_ptr<gpu::Engine>, ajp::Refusal> load(const gpu::Artifact&) override {
    return std::unexpected(ajp::Refusal{RefusalCode::ModelUnavailable, "factice"});
  }
  std::expected<gpu::Tensor, ajp::Refusal> import(const gpu::ExternalImage&) override {
    return std::unexpected(ajp::Refusal{RefusalCode::FieldOutOfRange, "factice"});
  }
  void enqueue(gpu::Engine&, std::span<const gpu::Tensor>, std::span<gpu::Tensor>) override {}
  gpu::Fence signal() override { return {}; }
  void wait(const gpu::Fence&) override {}

  int calls() const { return calls_; }
  int max_running() const { return max_running_; }

 private:
  std::atomic<int> calls_ = 0;
  std::atomic<int> running_ = 0;
  std::atomic<int> max_running_ = 0;
};

BuildRequest request(std::uint64_t id, std::string model, gpu::Backend* backend,
                     std::stop_token cancel = {}) {
  return {.id = id,
          .model = {.id = std::move(model)},
          .shapes = {},
          .backend = backend,
          .cancel = std::move(cancel)};
}

}  // namespace

TEST_CASE("Preparer : un moteur à la fois, résultats dans l'ordre des demandes", "[prepare][preparer]") {
  Sink sink;
  FakeBackend backend;
  Preparer preparer(sink);
  REQUIRE(preparer.submit(request(1, "lent", &backend)));
  REQUIRE(preparer.submit(request(2, "lent", &backend)));
  REQUIRE(preparer.submit(request(3, "refus", &backend)));
  const auto done = sink.wait_done(3);
  REQUIRE(done.size() == 3);
  CHECK(done[0].id == 1);
  CHECK(done[1].id == 2);
  CHECK(done[2].id == 3);
  CHECK(done[0].result.value().cache_key == "clé-lent");
  CHECK(test::refusal(done[2].result) == RefusalCode::BuilderFailed);
  CHECK(backend.max_running() == 1);
  const auto progress = sink.progress();
  REQUIRE(progress.size() >= 5);
  CHECK(progress.front().id == 1);
  CHECK(progress.front().model == "lent");
  CHECK(progress.front().stage == "moitié");
}

TEST_CASE("Preparer : exception, backend absent et demande annulée deviennent des refus",
          "[prepare][preparer]") {
  Sink sink;
  FakeBackend backend;
  Preparer preparer(sink);
  std::stop_source cancelled;
  cancelled.request_stop();
  REQUIRE(preparer.submit(request(1, "exception", &backend)));
  REQUIRE(preparer.submit(request(2, "span", nullptr)));
  REQUIRE(preparer.submit(request(3, "span", &backend, cancelled.get_token())));
  REQUIRE(preparer.submit(request(4, "span", &backend)));
  const auto done = sink.wait_done(4);
  REQUIRE(done.size() == 4);
  CHECK(test::refusal(done[0].result) == RefusalCode::BuilderCrashed);
  CHECK(done[0].result.error().detail.find("panne du backend") != std::string::npos);
  CHECK(test::refusal(done[1].result) == RefusalCode::FieldMissing);
  CHECK(test::refusal(done[2].result) == RefusalCode::Cancelled);
  CHECK(done[3].result.has_value());  // le thread a survécu à l'exception
  CHECK(backend.calls() == 2);        // la demande annulée n'a rien construit
}

TEST_CASE("Preparer : annulation de la construction en cours", "[prepare][preparer]") {
  Sink sink;
  FakeBackend backend;
  Preparer preparer(sink);
  std::stop_source cancel;
  REQUIRE(preparer.submit(request(1, "long", &backend, cancel.get_token())));
  REQUIRE(preparer.submit(request(2, "span", &backend)));
  std::this_thread::sleep_for(50ms);
  cancel.request_stop();
  const auto done = sink.wait_done(2);
  REQUIRE(done.size() == 2);
  CHECK(test::refusal(done[0].result) == RefusalCode::Cancelled);
  CHECK(done[1].result.has_value());
}

TEST_CASE("Preparer : l'arrêt du thread annule la construction en cours", "[prepare][preparer]") {
  Sink sink;
  FakeBackend backend;
  const auto start = std::chrono::steady_clock::now();
  {
    Preparer preparer(sink);
    REQUIRE(preparer.submit(request(1, "long", &backend)));
    std::this_thread::sleep_for(50ms);
  }
  CHECK(std::chrono::steady_clock::now() - start < 5s);
  CHECK(backend.calls() == 1);
}

TEST_CASE("Preparer : un résultat refusé par le puits attend le prochain crédit", "[prepare][preparer]") {
  Sink sink;
  FakeBackend backend;
  sink.refuse_next(2);  // le premier résultat, puis sa nouvelle tentative avec le second
  Preparer preparer(sink);
  REQUIRE(preparer.submit(request(1, "span", &backend)));
  REQUIRE(preparer.submit(request(2, "span", &backend)));
  std::this_thread::sleep_for(100ms);
  CHECK(sink.wait_done(0).empty());  // le second attend derrière le premier, refusé
  REQUIRE(preparer.credit());
  const auto done = sink.wait_done(2);
  REQUIRE(done.size() == 2);
  CHECK(done[0].id == 1);
  CHECK(done[1].id == 2);
}

TEST_CASE("Preparer : de bout en bout avec TrtexecBuilder et un faux trtexec", "[prepare][preparer]") {
  // Backend « TensorRT » sans GPU : il délègue build() à TrtexecBuilder.
  class TrtexecBackend final : public FakeBackend {
   public:
    explicit TrtexecBackend(TrtexecConfig config) : builder_(std::move(config)) {}
    std::expected<gpu::Artifact, ajp::Refusal> build(const gpu::ModelSpec& model, const gpu::ShapeSet& shapes,
                                                     gpu::Progress& progress) override {
      return builder_.build(model, shapes, progress);
    }

   private:
    TrtexecBuilder builder_;
  };

  const test::TempDir dir;
  const auto fake = dir / ("trtexec-ok" + test::helper_path().extension().string());
  test::fs::copy_file(test::helper_path(), fake);
  test::fs::permissions(fake, test::fs::perms::owner_all, test::fs::perm_options::add);
  const std::string onnx = test::onnx({});
  test::write_file(dir / "span.onnx", onnx);
  TrtexecBackend backend({.trtexec = fake,
                          .cache_dir = dir / "moteurs",
                          .runtime = {.major = 11, .minor = 3},
                          .device = {.name = "RTX de test"},
                          .timeout = 10s});
  Sink sink;
  Preparer preparer(sink);
  const gpu::Shape hd{{1, 3, 1080, 1920}};
  REQUIRE(
      preparer.submit({.id = 7,
                       .model = {.id = "span", .onnx_path = dir / "span.onnx", .sha256 = sha256_hex(onnx)},
                       .shapes = {.min = hd, .opt = hd, .max = hd},
                       .backend = &backend,
                       .cancel = {}}));
  const auto done = sink.wait_done(1);
  REQUIRE(done.size() == 1);
  REQUIRE(done[0].result.has_value());
  CHECK(test::fs::exists(done[0].result->path));
  const auto progress = sink.progress();
  REQUIRE_FALSE(progress.empty());
  CHECK(progress.back().fraction == 1.0F);
  CHECK(progress.back().stage == "Terminé");
}
