// L'écriture d'un export avec un encodeur factice qui garde N images avant de les rendre, comme
// NVENC avec son anticipation (constat E9 du second audit) : les tampons de la conversion lus en
// place ne reviennent pas tant que l'encodeur les tient, et une poignée de tampons (les kBuffers de
// FrameExport) s'épuise ; copiés dans les images de l'encodeur, ils reviennent dès la copie et
// l'export avance quel que soit N. Sans GPU : le mux écrit de vrais paquets factices dans un MKV.
#include "export/export_writer.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "clip_writer.h"
#include "export/av_handles.h"
#include "export/video_encoder.h"

namespace ajp::exporter {
namespace {

using namespace std::chrono_literals;

constexpr int kWidth = 64;
constexpr int kHeight = 48;
constexpr std::int64_t kFrameUs = 40'000;
// Les tampons de la conversion (FrameExportSetup::buffers par défaut) : lus en place, c'est tout ce que
// l'encodeur peut tenir avant que l'export ne se fige.
constexpr std::size_t kBuffers = 7;

// Un encodeur qui garde les `hold` dernières images reçues (une référence sur leur mémoire, comme
// NVENC qui lit en place) et rend un paquet factice par image ; vidé, il rend tout. Porte fermée
// (close_gate), il s'arrête au début de chaque image jusqu'à open_gate.
class HoldingEncoder : public FrameEncoder {
 public:
  explicit HoldingEncoder(std::size_t hold)
      : hold_(hold), ctx_(avcodec_alloc_context3(nullptr)), packet_(av_packet_alloc()) {
    ctx_->codec_type = AVMEDIA_TYPE_VIDEO;
    ctx_->codec_id = AV_CODEC_ID_FFV1;
    ctx_->width = kWidth;
    ctx_->height = kHeight;
    ctx_->pix_fmt = AV_PIX_FMT_YUV420P10LE;
    ctx_->time_base = kMicroseconds;
    ctx_->framerate = AVRational{25, 1};
  }
  ~HoldingEncoder() override { release_all(); }
  HoldingEncoder(const HoldingEncoder&) = delete;
  HoldingEncoder& operator=(const HoldingEncoder&) = delete;

  const AVCodecContext& context() const override { return *ctx_; }
  Layout layout() const override { return Layout::Yuv420p10; }
  Upload upload() const override { return Upload::Host; }
  AVBufferRef* frames_context() const override { return nullptr; }
  bool shared_device() const override { return false; }
  const std::string& label() const override { return label_; }

  // Sur le thread export : pas d'assertion Catch2 ici, un refus à la place.
  std::expected<void, Refusal> encode(AVFrame* frame, const PacketSink& sink) override {
    if (frame == nullptr) {
      release_all();
      return {};
    }
    entered_.fetch_add(1);
    {
      std::unique_lock lock(gate_mutex_);
      gate_.wait(lock, [this] { return !gate_closed_; });
    }
    AVFrame* kept = av_frame_clone(frame);
    if (kept == nullptr) return refuse(RefusalCode::EncodeFailed, "encodeur factice : mémoire");
    held_.push_back(kept);
    while (held_.size() > hold_) {
      av_frame_free(&held_.front());
      held_.pop_front();
    }
    pts_.push_back(frame->pts);
    seen_.fetch_add(1);
    if (av_new_packet(packet_.get(), 16) < 0) return refuse(RefusalCode::EncodeFailed, "paquet factice");
    packet_->pts = frame->pts;
    packet_->dts = frame->pts;
    packet_->flags = AV_PKT_FLAG_KEY;
    auto sunk = sink(*packet_);
    av_packet_unref(packet_.get());
    return sunk;
  }

  int seen() const { return seen_.load(); }
  int entered() const { return entered_.load(); }  // images commencées, porte passée ou non
  void close_gate() {
    const std::scoped_lock lock(gate_mutex_);
    gate_closed_ = true;
  }
  void open_gate() {
    {
      const std::scoped_lock lock(gate_mutex_);
      gate_closed_ = false;
    }
    gate_.notify_all();
  }
  // Les pts reçus, à lire après finish() (le thread est joint).
  const std::vector<std::int64_t>& pts() const { return pts_; }

 private:
  void release_all() {
    for (AVFrame*& frame : held_) av_frame_free(&frame);
    held_.clear();
  }

  std::size_t hold_;
  CodecContextPtr ctx_;
  PacketPtr packet_;
  std::string label_ = "factice";
  std::deque<AVFrame*> held_;
  std::vector<std::int64_t> pts_;
  std::atomic<int> seen_{0};
  std::atomic<int> entered_{0};
  std::mutex gate_mutex_;
  std::condition_variable gate_;
  bool gate_closed_ = false;
};

// Les tampons de la conversion : pris libres, rendus par Picture::release (depuis le thread export)
// ou dès la copie dans une image de l'encodeur.
struct Buffers {
  Buffers() {
    for (std::size_t i = 0; i < kBuffers; ++i) {
      memory[i].assign(bytes(), 0);
      free[i].store(true);
    }
  }
  static std::size_t bytes() { return luma_bytes() + 2 * chroma_bytes(); }
  static std::size_t luma_bytes() { return static_cast<std::size_t>(kWidth) * 2 * kHeight; }
  static std::size_t chroma_bytes() { return static_cast<std::size_t>(kWidth) * (kHeight / 2); }
  std::optional<std::size_t> take() {
    for (std::size_t i = 0; i < kBuffers; ++i) {
      if (free[i].exchange(false)) return i;
    }
    return std::nullopt;
  }
  std::size_t free_count() const {
    std::size_t n = 0;
    for (const auto& f : free)
      if (f.load()) ++n;
    return n;
  }
  // L'image lue en place par l'encodeur : le tampon revient quand il la rend.
  Picture in_place(std::size_t i, std::int64_t pts_us) {
    std::uint8_t* base = memory[i].data();
    return Picture{.planes = {base, base + luma_bytes(), base + luma_bytes() + chroma_bytes()},
                   .pitches = {kWidth * 2, kWidth, kWidth},
                   .bytes = bytes(),
                   .pts_us = pts_us,
                   .release = [this, i] { free[i].store(true); }};
  }

  std::array<std::vector<std::uint8_t>, kBuffers> memory;
  std::array<std::atomic<bool>, kBuffers> free;
};

std::filesystem::path source_clip() {
  static const auto path = media::test::write_clip(media::test::ClipSpec{
      .name = "export-writer-source.mkv", .width = kWidth, .height = kHeight, .frames = 10});
  REQUIRE(path);
  return *path;
}

WriterConfig config_for(const std::filesystem::path& output) {
  return WriterConfig{.source = source_clip(),
                      .output = output,
                      .width = kWidth,
                      .height = kHeight,
                      .fps_num = 25,
                      .fps_den = 1,
                      .codec = Codec::Lossless,
                      .quality = Quality::VeryHigh};
}

// Confie l'image en attendant une place dans la file ; échoue si l'export s'est arrêté.
void push(ExportWriter& writer, Picture&& picture) {
  for (int i = 0; i < 5000; ++i) {
    if (writer.try_push(std::move(picture))) return;
    const auto failure = writer.failure();
    if (failure) FAIL("export arrêté : " << failure->detail);
    std::this_thread::sleep_for(1ms);
  }
  FAIL("file de l'écriture jamais libre");
}

void wait_seen(const HoldingEncoder& encoder, int count) {
  for (int i = 0; i < 5000 && encoder.seen() < count; ++i) std::this_thread::sleep_for(1ms);
  REQUIRE(encoder.seen() == count);
}

std::filesystem::path output_path(const char* name) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove(path);
  return path;
}

}  // namespace

TEST_CASE("Écriture : lus en place, les tampons restent à l'encodeur qui garde N images (E9)",
          "[export][writer]") {
  const std::filesystem::path output = output_path("ajp-export-writer-in-place.mkv");
  auto encoder = std::make_unique<HoldingEncoder>(8);  // plus que de tampons
  const HoldingEncoder* fake = encoder.get();
  auto writer = ExportWriter::open(config_for(output), std::move(encoder));
  REQUIRE(writer);
  CHECK((*writer)->layout() == Layout::Yuv420p10);
  CHECK((*writer)->upload() == Upload::Host);
  CHECK((*writer)->cuda_frames() == nullptr);

  Buffers buffers;
  for (std::size_t n = 0; n < kBuffers; ++n) {
    const auto i = buffers.take();
    REQUIRE(i);
    push(**writer, buffers.in_place(*i, static_cast<std::int64_t>(n) * kFrameUs));
  }
  wait_seen(*fake, static_cast<int>(kBuffers));
  // L'export se fige : aucun tampon ne revient tant que l'encodeur tient les images.
  CHECK(buffers.free_count() == 0);
  std::this_thread::sleep_for(50ms);
  CHECK(buffers.free_count() == 0);
  CHECK_FALSE(buffers.take());

  // Vidé, l'encodeur rend tout.
  REQUIRE((*writer)->finish());
  CHECK(buffers.free_count() == kBuffers);
  CHECK((*writer)->progress().frames == static_cast<std::int64_t>(kBuffers));
  CHECK((*writer)->progress().bytes > 0);
  writer->reset();
  std::error_code error;
  CHECK(std::filesystem::file_size(output, error) > 0);
  std::filesystem::remove(output, error);
}

TEST_CASE("Écriture : copiés dans les images de l'encodeur, les tampons reviennent aussitôt",
          "[export][writer]") {
  const std::filesystem::path output = output_path("ajp-export-writer-pool.mkv");
  auto encoder = std::make_unique<HoldingEncoder>(8);
  const HoldingEncoder* fake = encoder.get();
  auto writer = ExportWriter::open(config_for(output), std::move(encoder));
  REQUIRE(writer);

  Buffers buffers;
  constexpr int kFrames = 30;
  for (int n = 0; n < kFrames; ++n) {
    const auto i = buffers.take();
    REQUIRE(i);  // un tampon est toujours libre : rendu dès la copie, pas à la fin de l'encodage
    AVFrame* pool = av_frame_alloc();
    REQUIRE(pool != nullptr);
    pool->format = AV_PIX_FMT_YUV420P10LE;
    pool->width = kWidth;
    pool->height = kHeight;
    REQUIRE(av_frame_get_buffer(pool, 0) == 0);
    buffers.free[*i].store(true);  // la copie faite, le tampon repart au rendu
    push(**writer, Picture{.pts_us = std::int64_t{n} * kFrameUs, .frame = pool});
  }
  REQUIRE((*writer)->finish());
  CHECK(fake->seen() == kFrames);
  CHECK(buffers.free_count() == kBuffers);
  CHECK((*writer)->progress().frames == kFrames);
  const std::vector<std::int64_t>& pts = fake->pts();
  REQUIRE(pts.size() == kFrames);
  for (std::size_t n = 1; n < pts.size(); ++n) CHECK(pts[n] > pts[n - 1]);
  CHECK(pts[1] - pts[0] == kFrameUs);
  writer->reset();
  std::error_code error;
  CHECK(std::filesystem::file_size(output, error) > 0);
  std::filesystem::remove(output, error);
}

TEST_CASE("Écriture : la file tient queue_depth images, pas une de plus", "[export][writer]") {
  const std::size_t depth = GENERATE(kDefaultQueueDepth, std::size_t{8});
  const std::filesystem::path output = output_path("ajp-export-writer-depth.mkv");
  auto encoder = std::make_unique<HoldingEncoder>(0);
  HoldingEncoder* fake = encoder.get();
  fake->close_gate();
  WriterConfig config = config_for(output);
  config.queue_depth = depth;
  auto writer = ExportWriter::open(config, std::move(encoder));
  REQUIRE(writer);

  std::vector<std::vector<std::uint8_t>> memory(depth + 2, std::vector<std::uint8_t>(Buffers::bytes()));
  const auto picture = [&](std::size_t n) {
    std::uint8_t* base = memory[n].data();
    return Picture{.planes = {base, base + Buffers::luma_bytes(),
                              base + Buffers::luma_bytes() + Buffers::chroma_bytes()},
                   .pitches = {kWidth * 2, kWidth, kWidth},
                   .bytes = Buffers::bytes(),
                   .pts_us = static_cast<std::int64_t>(n) * kFrameUs,
                   .release = [] {}};
  };
  // La première sort de la file et attend à la porte de l'encodeur : la file en reçoit depth.
  push(**writer, picture(0));
  for (int i = 0; i < 5000 && fake->entered() < 1; ++i) std::this_thread::sleep_for(1ms);
  REQUIRE(fake->entered() == 1);
  for (std::size_t n = 1; n <= depth; ++n) CHECK((*writer)->try_push(picture(n)));
  CHECK_FALSE((*writer)->try_push(picture(depth + 1)));

  fake->open_gate();
  REQUIRE((*writer)->finish());
  CHECK((*writer)->progress().frames == static_cast<std::int64_t>(depth + 1));
  writer->reset();
  std::error_code error;
  std::filesystem::remove(output, error);
}

TEST_CASE("Écriture : répétitions de la cadence fixe", "[export][writer]") {
  CHECK(repeats_between(0, 40'000, 25, 1) == 0);
  CHECK(repeats_between(0, 59'999, 25, 1) == 0);  // moins d'une période et demie
  CHECK(repeats_between(0, 60'000, 25, 1) == 1);  // arrondi au plus proche
  CHECK(repeats_between(0, 80'000, 25, 1) == 1);
  CHECK(repeats_between(0, 120'000, 25, 1) == 2);
  // ×2 sur 23,976 i/s : une paire laissée sans interpolation creuse un trou d'une image.
  CHECK(repeats_between(0, 41'708, 48'000, 1'001) == 1);
  CHECK(repeats_between(0, 20'854, 48'000, 1'001) == 0);
  CHECK(repeats_between(100, 100, 25, 1) == 0);
  CHECK(repeats_between(100, 50, 25, 1) == 0);
  CHECK(repeats_between(0, 1'000'000, 0, 1) == 0);
  // Une heure sans image : un saut d'horodatage, pas un dessin tenu.
  CHECK(repeats_between(0, std::int64_t{3'600} * 1'000'000, 25, 1) == 0);
  CHECK(repeats_between(0, (kMaxRepeatRun + 1) * 40'000, 25, 1) == kMaxRepeatRun);
}

// 25 i/s : les instants 80, 200 et 240 ms manquent (une coupe, un dessin tenu sur deux images).
constexpr std::array<std::int64_t, 5> kShownUs{0, 40'000, 120'000, 160'000, 280'000};

std::vector<std::int64_t> write_shown(const char* name, bool constant_rate, WriterProgress& progress) {
  const std::filesystem::path output = output_path(name);
  auto encoder = std::make_unique<HoldingEncoder>(2);
  const HoldingEncoder* fake = encoder.get();
  WriterConfig config = config_for(output);
  config.constant_rate = constant_rate;
  auto writer = ExportWriter::open(config, std::move(encoder));
  REQUIRE(writer);
  Buffers buffers;
  for (const std::int64_t pts : kShownUs) {
    const auto i = buffers.take();
    REQUIRE(i);
    push(**writer, buffers.in_place(*i, pts));
  }
  REQUIRE((*writer)->finish());
  // Les répétitions lisent la mémoire de l'image répétée : tout revient une fois l'encodeur vidé.
  CHECK(buffers.free_count() == kBuffers);
  progress = (*writer)->progress();
  std::vector<std::int64_t> pts = fake->pts();
  writer->reset();
  std::error_code error;
  CHECK(std::filesystem::file_size(output, error) > 0);
  std::filesystem::remove(output, error);
  return pts;
}

TEST_CASE("Écriture : à cadence fixe, l'image précédente remplit les instants manquants",
          "[export][writer]") {
  WriterProgress progress;
  const std::vector<std::int64_t> pts = write_shown("ajp-export-writer-constant.mkv", true, progress);
  REQUIRE(pts.size() == 8);
  for (std::size_t n = 1; n < pts.size(); ++n) CHECK(pts[n] - pts[n - 1] == kFrameUs);
  CHECK(progress.frames == 8);
  CHECK(progress.repeated == 3);
}

TEST_CASE("Écriture : à cadence variable, les instants manquants restent vides", "[export][writer]") {
  WriterProgress progress;
  const std::vector<std::int64_t> pts = write_shown("ajp-export-writer-variable.mkv", false, progress);
  REQUIRE(pts.size() == kShownUs.size());
  for (std::size_t n = 1; n < pts.size(); ++n) CHECK(pts[n] - pts[0] == kShownUs[n]);
  CHECK(progress.frames == static_cast<std::int64_t>(kShownUs.size()));
  CHECK(progress.repeated == 0);
}

TEST_CASE("Écriture : un refus de l'encodeur arrête l'export et le dit", "[export][writer]") {
  // Un encodeur qui refuse dès la première image : le thread s'arrête, try_push refuse ensuite.
  class RefusingEncoder final : public HoldingEncoder {
   public:
    RefusingEncoder() : HoldingEncoder(1) {}
    std::expected<void, Refusal> encode(AVFrame* frame, const PacketSink& /*sink*/) override {
      if (frame == nullptr) return {};
      return refuse(RefusalCode::EncodeFailed, "encodeur factice : image refusée");
    }
  };
  const std::filesystem::path output = output_path("ajp-export-writer-refused.mkv");
  auto writer = ExportWriter::open(config_for(output), std::make_unique<RefusingEncoder>());
  REQUIRE(writer);
  Buffers buffers;
  push(**writer, buffers.in_place(0, 0));
  for (int i = 0; i < 5000 && !(*writer)->failure(); ++i) std::this_thread::sleep_for(1ms);
  const auto failure = (*writer)->failure();
  REQUIRE(failure);
  CHECK(failure->code == RefusalCode::EncodeFailed);
  CHECK(failure->detail == "encodeur factice : image refusée");
  CHECK_FALSE((*writer)->try_push(buffers.in_place(1, kFrameUs)));
  CHECK_FALSE((*writer)->finish());
  writer->reset();  // abandonné : le fichier partiel est supprimé
  std::error_code error;
  CHECK_FALSE(std::filesystem::exists(output, error));
}

}  // namespace ajp::exporter
