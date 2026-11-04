#include "app/play/placebo_wayland.h"

#include <format>

#ifdef AJP_WITH_WAYLAND_PRESENTATION
#include <cstring>
#include <poll.h>
#include <string_view>
#include <time.h>
#include <wayland-client.h>

#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>

#include "presentation-time-client.h"
#endif

namespace ajp::app::play {

ScreenStamp stamp_of(std::uint64_t seq, std::uint32_t sec_hi, std::uint32_t sec_lo, std::uint32_t nsec,
                     std::uint32_t refresh_ns, std::uint32_t seq_hi, std::uint32_t seq_lo,
                     std::uint32_t flags) {
  const auto sec = static_cast<std::int64_t>((std::uint64_t{sec_hi} << 32U) | sec_lo);
  return ScreenStamp{.seq = seq,
                     .presented_us = sec * 1'000'000 + std::int64_t{nsec} / 1000,
                     .refresh_ns = refresh_ns,
                     .msc = (std::uint64_t{seq_hi} << 32U) | seq_lo,
                     .flags = flags};
}

std::string_view screen_stamp_header() {
  return "seq\tpresented_us\trefresh_ns\tmsc\tflags\n";
}

std::string screen_stamp_line(const ScreenStamp& s) {
  return std::format("{}\t{}\t{}\t{}\t{:#x}\n", s.seq, s.presented_us, s.refresh_ns, s.msc, s.flags);
}

#ifdef AJP_WITH_WAYLAND_PRESENTATION

struct WaylandPresentation::State {
  struct Pending {
    State* owner = nullptr;
    std::uint64_t id = 0;
    struct wp_presentation_feedback* feedback = nullptr;
  };
  wl_display* display = nullptr;
  wl_surface* surface = nullptr;
  wl_event_queue* queue = nullptr;
  wl_display* wrapper = nullptr;  // la connexion vue de notre file
  wl_registry* registry = nullptr;
  wp_presentation* presentation = nullptr;
  std::uint32_t clock_id = UINT32_MAX;
  std::vector<WaylandDone>* out = nullptr;  // pendant read()
  std::vector<WaylandDone> early;           // arrivés hors read()
  std::vector<Pending*> pending;

  void finish(Pending* p, const ScreenStamp& stamp) {
    (out != nullptr ? *out : early)
        .push_back(WaylandDone{.id = p->id,
                               .visible_us = stamp.presented_us,
                               .refresh_ns = stamp.refresh_ns,
                               .msc = stamp.msc,
                               .flags = stamp.flags});
    wp_presentation_feedback_destroy(p->feedback);
    std::erase(pending, p);
    delete p;
  }

  ~State() {
    for (Pending* p : pending) {
      wp_presentation_feedback_destroy(p->feedback);
      delete p;
    }
    if (presentation != nullptr) wp_presentation_destroy(presentation);
    if (registry != nullptr) wl_registry_destroy(registry);
    if (wrapper != nullptr) wl_proxy_wrapper_destroy(wrapper);
    if (queue != nullptr) wl_event_queue_destroy(queue);
  }
};

namespace {

using State = WaylandPresentation::State;

void on_global(void* data, wl_registry* registry, std::uint32_t name, const char* interface,
               std::uint32_t version) {
  auto* s = static_cast<State*>(data);
  if (s->presentation == nullptr && std::string_view(interface) == wp_presentation_interface.name) {
    (void)version;
    s->presentation =
        static_cast<wp_presentation*>(wl_registry_bind(registry, name, &wp_presentation_interface, 1));
  }
}

void on_global_remove(void*, wl_registry*, std::uint32_t) {}

constexpr wl_registry_listener kRegistry{.global = on_global, .global_remove = on_global_remove};

void on_clock(void* data, wp_presentation*, std::uint32_t clock_id) {
  static_cast<State*>(data)->clock_id = clock_id;
}

constexpr wp_presentation_listener kPresentation{.clock_id = on_clock};

void on_sync_output(void*, struct wp_presentation_feedback*, wl_output*) {}

void on_presented(void* data, struct wp_presentation_feedback*, std::uint32_t sec_hi, std::uint32_t sec_lo,
                  std::uint32_t nsec, std::uint32_t refresh, std::uint32_t seq_hi, std::uint32_t seq_lo,
                  std::uint32_t flags) {
  auto* p = static_cast<State::Pending*>(data);
  p->owner->finish(p, stamp_of(p->id, sec_hi, sec_lo, nsec, refresh, seq_hi, seq_lo, flags));
}

void on_discarded(void* data, struct wp_presentation_feedback*) {
  auto* p = static_cast<State::Pending*>(data);
  p->owner->finish(p, ScreenStamp{.seq = p->id});
}

constexpr wp_presentation_feedback_listener kFeedback{
    .sync_output = on_sync_output, .presented = on_presented, .discarded = on_discarded};

}  // namespace

std::unique_ptr<WaylandPresentation> WaylandPresentation::open(
    SDL_Window* window, const std::shared_ptr<spdlog::logger>& channel) {
  const char* driver = SDL_GetCurrentVideoDriver();
  if (window == nullptr || driver == nullptr || std::strcmp(driver, "wayland") != 0) return nullptr;
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  auto state = std::make_unique<State>();
  state->display = static_cast<wl_display*>(
      SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr));
  state->surface = static_cast<wl_surface*>(
      SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr));
  if (state->display == nullptr || state->surface == nullptr) return nullptr;
  state->queue = wl_display_create_queue(state->display);
  state->wrapper = static_cast<wl_display*>(wl_proxy_create_wrapper(state->display));
  if (state->queue == nullptr || state->wrapper == nullptr) return nullptr;
  wl_proxy_set_queue(reinterpret_cast<wl_proxy*>(state->wrapper), state->queue);
  state->registry = wl_display_get_registry(state->wrapper);
  wl_registry_add_listener(state->registry, &kRegistry, state.get());
  // À l'ouverture seulement : deux allers-retours sur notre file (les globaux, puis l'horloge).
  if (wl_display_roundtrip_queue(state->display, state->queue) < 0) return nullptr;
  if (state->presentation == nullptr) {
    channel->info("présentation : le compositeur n'offre pas wp_presentation");
    return nullptr;
  }
  wp_presentation_add_listener(state->presentation, &kPresentation, state.get());
  if (wl_display_roundtrip_queue(state->display, state->queue) < 0) return nullptr;
  if (state->clock_id != CLOCK_MONOTONIC) {
    channel->info("présentation : horloge wp_presentation {} et non CLOCK_MONOTONIC, niveau wayland écarté",
                  state->clock_id);
    return nullptr;
  }
  return std::unique_ptr<WaylandPresentation>(new WaylandPresentation(std::move(state)));
}

void WaylandPresentation::request(std::uint64_t id) {
  State& s = *state_;
  auto* p = new State::Pending{.owner = &s, .id = id, .feedback = nullptr};
  // Le retour naît sur la file de wp_presentation, la nôtre.
  p->feedback = wp_presentation_feedback(s.presentation, s.surface);
  if (p->feedback == nullptr) {
    delete p;
    return;
  }
  wp_presentation_feedback_add_listener(p->feedback, &kFeedback, p);
  s.pending.push_back(p);
}

void WaylandPresentation::read(std::vector<WaylandDone>& out) {
  State& s = *state_;
  out.insert(out.end(), s.early.begin(), s.early.end());
  s.early.clear();
  s.out = &out;
  // Lecture sans attente (wayland-client.h, « multi-threaded event reading ») : préparer, sonder le fd à
  // délai nul, lire ou renoncer, puis répartir ce qui revient à notre file.
  while (wl_display_prepare_read_queue(s.display, s.queue) != 0)
    wl_display_dispatch_queue_pending(s.display, s.queue);
  wl_display_flush(s.display);
  pollfd fd{.fd = wl_display_get_fd(s.display), .events = POLLIN, .revents = 0};
  if (poll(&fd, 1, 0) > 0 && (fd.revents & POLLIN) != 0) {
    wl_display_read_events(s.display);
  } else {
    wl_display_cancel_read(s.display);
  }
  wl_display_dispatch_queue_pending(s.display, s.queue);
  s.out = nullptr;
}

#else

struct WaylandPresentation::State {};

std::unique_ptr<WaylandPresentation> WaylandPresentation::open(SDL_Window*,
                                                               const std::shared_ptr<spdlog::logger>&) {
  return nullptr;
}

void WaylandPresentation::request(std::uint64_t) {}
void WaylandPresentation::read(std::vector<WaylandDone>&) {}

#endif

WaylandPresentation::WaylandPresentation(std::unique_ptr<State> state) : state_(std::move(state)) {}
WaylandPresentation::~WaylandPresentation() = default;

}  // namespace ajp::app::play
