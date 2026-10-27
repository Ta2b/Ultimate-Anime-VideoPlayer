#include "app/play/export_run.h"

#include <algorithm>
#include <deque>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_timer.h>

extern "C" {
#include <libavutil/frame.h>
}

#include "app/play/chain_setup.h"
#include "app/play/controls.h"
#include "app/play/export_depth.h"
#include "app/play/export_sink.h"
#include "app/play/first_run.h"
#include "app/play/live_settings.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_export.h"
#include "app/play/upscale_stage.h"
#include "core/clock.h"
#include "core/log.h"
#include "export/export_writer.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

// La progression est redessinée 4 fois par seconde : la présentation (FIFO) ne freine pas l'export.
constexpr std::int64_t kProgressEveryUs = 250'000;

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

std::string rate_text(media::Rational rate) {
  return std::format("{:.3f}", static_cast<double>(rate.num) / static_cast<double>(rate.den));
}

std::string seconds_text(double seconds) {
  return format_clock(static_cast<std::int64_t>(seconds * 1e6), 0);
}

// Une touche ou la fenêtre pendant l'export : annuler, quitter, plein écran, détails (Tab).
std::optional<ExportEnd> poll_export_events(PlaceboScreen& screen, bool& details) {
  std::optional<ExportEnd> end;
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
      end = ExportEnd::Quit;
    } else if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
      if (event.key.key == SDLK_ESCAPE || event.key.key == SDLK_Q) {
        if (!end) end = ExportEnd::Cancelled;
      } else if (event.key.key == SDLK_TAB) {
        details = !details;
      } else if (event.key.key == SDLK_F) {
        screen.set_fullscreen(!screen.fullscreen());
      }
    }
  }
  return end;
}

}  // namespace

ExportOutcome run_export(Player& p, const ExportJob& job) {
  const auto log = log_channel("app");
  const auto stopped = [&](const Refusal& refusal) {
    log->warn("export : {} {}", name(refusal.code), refusal.detail);
    return ExportOutcome{.end = ExportEnd::Failed,
                         .notice = std::format("Export arrêté : {}", refusal.detail)};
  };
  std::error_code unknown;
  if (job.output.empty() || std::filesystem::equivalent(job.output, p.options.file, unknown))
    return stopped(Refusal{RefusalCode::FileUnwritable, "le fichier exporté remplacerait la source"});

  std::vector<std::string> unavailable;
  const PlaySettings settings = within_engines(export_chain(job.chain), p.engines, unavailable);
  for (const std::string& line : unavailable) log->warn("{}", line);
  auto prepared = prepare_chain(p, settings);
  if (!prepared) return stopped(prepared.error());
  if (!*prepared) return {.end = ExportEnd::Quit};
  ChainEngines& c = **prepared;
  const media::TrackInfo& video = p.video;
  const ExportSize chain_size{.width = c.work.width * c.factor, .height = c.work.height * c.factor};
  const ExportSize size = export_size(chain_size, {.width = video.width, .height = video.height}, job.height);
  const Factor factor = c.interpolates() ? factor_of(settings) : Factor{};
  media::Rational rate = export_rate(video.frame_rate, factor);
  if (rate.num <= 0) rate = export_rate({.num = 24'000, .den = 1'001}, factor);  // fréquence inconnue
  p.screen.set_source_color(ColorInfo{});  // les images du ring sans RTX HDR, comme à la lecture
  // Le ring, la conversion et la file d'écriture : deux paires en vol si les voies de calcul le
  // permettent (export_depth.h).
  const ExportDepth depth = export_depth(c.one_lane ? 1 : c.backend->capabilities().lanes);
  show_loading(p.screen, p.ui, "Ouverture de l'export");

  // Ordre de déclaration = ordre inverse de destruction : l'écriture, qui rend ses images aux
  // tampons de la conversion, part avant elle ; les moteurs (prepared) partent les derniers.
  std::unique_ptr<FrameExport> frames;
  auto writer = exporter::ExportWriter::open(
      exporter::WriterConfig{.source = p.options.file,
                             .output = job.output,
                             .width = size.width,
                             .height = size.height,
                             .fps_num = rate.num,
                             .fps_den = rate.den,
                             .codec = job.codec,
                             .quality = job.quality,
                             .vendor = exporter::vendor_of(p.screen.device_facts().pci_vendor),
                             .cuda_ordinal = export_cuda_ordinal(p.screen.device()),
                             .vulkan_device = p.screen.vulkan_device(),
                             .queue_depth = depth.queue,
                             .caller_frames = depth.buffers,
                             .constant_rate = job.constant_rate});
  if (!writer) return stopped(writer.error());
  // Sous TensorRT avec NVENC à la taille de la chaîne, l'étage gpu écrit les images de l'encodeur
  // lui-même (export_sink.h) ; sinon le ring, rendu par libplacebo (placebo_export.h).
  std::shared_ptr<ExportSink> sink;
  const bool yuv_targets = c.backend->capabilities().yuv_targets;
  if (const auto blocker = sink_blocker(yuv_targets, (*writer)->cuda_frames() != nullptr, (*writer)->layout(),
                                        (*writer)->color().matrix, size, chain_size)) {
    if (yuv_targets) log->info("export : rendu par libplacebo, {}", *blocker);
  } else if (auto made =
                 ExportSink::open(sink_setup(pool_frames((*writer)->cuda_frames()), size, (*writer)->layout(),
                                             (*writer)->color(), depth.ring_slots))) {
    sink = std::move(*made);
    log->info("export : images écrites en YUV dans celles de l'encodeur, sans ring ni libplacebo");
  } else {
    log->warn("export : puits CUDA refusé ({}), rendu par libplacebo", made.error().detail);
  }
  if (!sink) {
    auto opened = FrameExport::open(p.screen, FrameExportSetup{.width = size.width,
                                                               .height = size.height,
                                                               .layout = (*writer)->layout(),
                                                               .color = (*writer)->color(),
                                                               .cuda_frames = (*writer)->cuda_frames(),
                                                               .buffers = depth.buffers});
    if (!opened) return stopped(opened.error());
    frames = std::move(*opened);
  }

  auto live = std::make_shared<LiveSettings>();
  live->set_rife_factor(factor);
  live->set_reduction(reduction_of(settings.downscale));
  live->set_keep_poses(settings.keep_poses);
  const PipelineConfig config{
      .file = p.options.file,
      .decoder = decoder_options(c.backend_name, p.screen.device(), p.screen.shared_devices()),
      .audio = false,
      .subtitles = false};
  auto pipeline = Pipeline::open(
      config, [&](VideoStageInit stage) -> std::expected<std::unique_ptr<VideoStage>, Refusal> {
        const int width = stage.track->width;
        const int height = stage.track->height;
        const WorkSize sized = work_size(width, height, settings.work_height);
        const ExportSize output{.width = sized.width * c.factor, .height = sized.height * c.factor};
        std::vector<RingSlotDesc> slots;
        if (sink) {
          if (output != size) {
            return std::unexpected(Refusal{RefusalCode::ShapeInvalid,
                                           std::format("export : sortie {}x{} au lieu de {}x{}", output.width,
                                                       output.height, size.width, size.height)});
          }
          slots = sink->slots();
        } else {
          auto ring = p.screen.create_ring(output.width, output.height, depth.ring_slots);
          if (!ring) return std::unexpected(std::move(ring.error()));
          slots = std::move(*ring);
        }
        // Sans horloge : aucune paire sautée pour retard, chaque image intermédiaire est faite.
        UpscaleSetup setup = c.upscale(settings, live, nullptr, width, height, std::move(slots));
        setup.yuv_slots = sink;
        return UpscaleStage::start(std::move(stage), std::move(setup));
      });
  if (!pipeline) return stopped(pipeline.error());

  const std::int64_t duration_us = (*pipeline)->session_config().duration_us;
  const std::string chain_text = std::format("{}{}", c.motion_label(settings), c.span_label());
  log->info("export : {} → {} ; {}x{} à {} i/s · {} ; {}", prepare::utf8_path(p.options.file),
            prepare::utf8_path(job.output), size.width, size.height, rate_text(rate), (*writer)->encoder(),
            chain_text);
  log->info("export : profil {} : {} slots, {} tampons de conversion, file d'écriture de {}",
            depth.throughput ? "débit" : "mémoire", depth.ring_slots, depth.buffers, depth.queue);

  std::deque<RingFrame> waiting;          // reçues du thread gpu, pas encore converties
  std::optional<exporter::Picture> held;  // puits : prise au slot, refusée par la file pleine
  bool ended = false;                     // StreamEnded vu
  std::optional<Refusal> failure;
  std::optional<ExportEnd> interrupted;
  std::int64_t converted = 0;
  std::int64_t last_pts = 0;
  bool details = true;
  const std::int64_t started = host_now_us();
  std::int64_t shown_at = 0;
  const std::function<bool(exporter::Picture&&)> push = [&](exporter::Picture&& picture) {
    return (*writer)->try_push(std::move(picture));
  };
  while (true) {
    interrupted = poll_export_events(p.screen, details);
    if (interrupted) break;
    if (auto refused = (*writer)->failure()) {
      failure = std::move(*refused);
      break;
    }
    bool moved = false;
    while (frames) {
      auto delivered = frames->deliver(push);
      if (!delivered) {
        failure = std::move(delivered.error());
        break;
      }
      if (!*delivered) break;
      moved = true;
    }
    if (failure) break;
    // Lu avant de vider la boîte : le thread gpu poste ses RingFrame avant de se dire au repos.
    const bool drained = ended && (*pipeline)->video_idle();
    (*pipeline)->pump();
    while (auto message = try_take((*pipeline)->inbox())) {
      std::visit(Overloaded{[&](RingFrame& frame) { waiting.push_back(frame); }, [](subs::SubtitleFrame&) {},
                            [&](const StreamEnded&) { ended = true; },
                            [&](const StageRefusal& refusal) {
                              if (refusal.fatal || refusal.stage == "gpu") {
                                if (!failure) failure = refusal.refusal;
                              } else {
                                log->warn("export : {} : {}", refusal.stage, refusal.refusal.detail);
                              }
                            }},
                 *message);
    }
    if (failure) break;
    // Puits : chaque image finie part à l'écriture dans l'ordre, son slot repart avec une image neuve.
    while (sink) {
      if (!held) {
        if (waiting.empty() || !sink->ready(waiting.front())) break;
        const RingFrame frame = waiting.front();
        auto taken = sink->take(frame.slot);
        if (!taken) {
          failure = std::move(taken.error());
          break;
        }
        waiting.pop_front();
        static_cast<void>((*pipeline)->stages().slot_free(release_of(frame)));
        held = exporter::Picture{.pts_us = frame.pts_us, .frame = *taken};
        last_pts = frame.pts_us;
        ++converted;
        moved = true;
      }
      if (!(*writer)->try_push(std::move(*held))) break;  // refusée : l'image reste à held
      held.reset();
    }
    while (frames && !waiting.empty() && frames->has_room()) {
      const RingFrame frame = waiting.front();
      waiting.pop_front();
      FrameExport::Converted done = frames->convert(frame);
      static_cast<void>((*pipeline)->stages().slot_free(done.freed));
      if (done.failure) {
        failure = std::move(done.failure);
        break;
      }
      last_pts = frame.pts_us;
      ++converted;
      moved = true;
    }
    if (failure) break;
    if (drained && waiting.empty() && (frames ? frames->idle() : !held)) break;

    const std::int64_t now = host_now_us();
    if (now - shown_at >= kProgressEveryUs) {
      shown_at = now;
      const double elapsed = static_cast<double>(now - started) / 1e6;
      const float fraction =
          duration_us > 0
              ? std::clamp(static_cast<float>(last_pts) / static_cast<float>(duration_us), 0.0F, 1.0F)
              : 0.0F;
      const std::string left =
          fraction > 0.01F ? seconds_text(elapsed * (1.0 - fraction) / fraction) : std::string("…");
      const PrepareView view{
          .label = "Export en cours · Échap pour annuler",
          .fraction = duration_us > 0 ? std::optional(fraction) : std::nullopt,
          .details = {std::format("{} images · {:.1f} i/s · écoulé {} · reste ~{}", converted,
                                  elapsed > 0.0 ? static_cast<double>(converted) / elapsed : 0.0,
                                  seconds_text(elapsed), left),
                      std::format("{}x{} à {} i/s · {}", size.width, size.height, rate_text(rate),
                                  (*writer)->encoder()),
                      chain_text,
                      std::format("{} · {:.0f} Mo", prepare::utf8_path(job.output),
                                  static_cast<double>((*writer)->progress().bytes) / 1e6)},
          .show_details = details};
      if (auto shown = p.screen.show(ShowRequest{.ui = p.ui.preparation(now, view)}); !shown)
        log->warn("export : {}", shown.error().detail);
    }
    if (!moved) SDL_Delay(1);
  }

  std::string truncated;  // la source n'a pas pu être relue jusqu'au bout pour les pistes copiées
  if (!interrupted && !failure) {
    show_loading(p.screen, p.ui, "Finalisation de l'export");
    if (auto finished = (*writer)->finish(); !finished) failure = std::move(finished.error());
    else
      truncated = (*writer)->warning();
  }
  const double elapsed = static_cast<double>(host_now_us() - started) / 1e6;
  const exporter::WriterProgress written = (*writer)->progress();
  // Ordre d'arrêt : les slots rendus, les threads arrêtés, l'écriture (abandonnée si elle n'est pas
  // finie : le fichier partiel est supprimé), la conversion, puis les moteurs avant leur backend.
  for (const RingFrame& frame : waiting)
    static_cast<void>((*pipeline)->stages().slot_free(sink ? release_of(frame) : p.screen.retire(frame)));
  waiting.clear();
  if (held) av_frame_free(&held->frame);
  (*pipeline)->stages().quit();
  pipeline->reset();
  writer->reset();
  frames.reset();
  sink.reset();  // attend les écritures YUV en vol : le backend vit encore
  prepared->reset();

  if (interrupted == ExportEnd::Quit) return {.end = ExportEnd::Quit};
  if (interrupted == ExportEnd::Cancelled) {
    log->info("export annulé après {} images", converted);
    return {.end = ExportEnd::Cancelled, .notice = "Export annulé"};
  }
  if (failure) return stopped(*failure);
  log->info("export terminé : {} images, {} octets, {:.1f} s", written.frames, written.bytes, elapsed);
  std::string notice =
      std::format("Export terminé : {} ({} images, {:.0f} Mo, {})", prepare::utf8_path(job.output.filename()),
                  written.frames, static_cast<double>(written.bytes) / 1e6, seconds_text(elapsed));
  if (!truncated.empty())
    notice += std::format(
        " — attention : l'audio et les sous-titres s'arrêtent avant la fin, la source "
        "n'a pas pu être relue ({})",
        truncated);
  return {.end = ExportEnd::Done, .notice = std::move(notice)};
}

}  // namespace ajp::app::play
