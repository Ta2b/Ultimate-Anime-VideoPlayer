#include "app/play/modal_redraw.h"

#include <utility>

#include <SDL3/SDL_video.h>

#include "app/play/pipeline.h"
#include "app/play/session.h"
#include "core/clock.h"
#include "core/inbox.h"
#include "core/log.h"

namespace ajp::app::play {

void modal_turn(Pipeline& pipeline, Session& session) {
  pipeline.pump();
  while (auto message = try_take(pipeline.inbox())) session.handle(std::move(*message));
  if (const auto shown = session.tick(host_now_us(), UiFrame{}); !shown)
    log_channel("app")->debug("boucle modale : {}", shown.error().detail);  // i18n-ok : journal
}

int ModalRedraw::suspended = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

void wait_events_no_modal_turn(Sint32 timeout_ms) {
  ++ModalRedraw::suspended;
  (void)SDL_WaitEventTimeout(nullptr, timeout_ms);
  --ModalRedraw::suspended;
}

ModalRedraw::ModalRedraw(SDL_Window* window, std::function<void()> turn)
    : window_(window != nullptr ? SDL_GetWindowID(window) : 0), turn_(std::move(turn)) {
  (void)SDL_AddEventWatch(&on_event, this);
}

ModalRedraw::~ModalRedraw() {
  SDL_RemoveEventWatch(&on_event, this);
}

bool ModalRedraw::on_event(void* context, SDL_Event* event) {
  auto* self = static_cast<ModalRedraw*>(context);
  // SDL garantit le thread principal pour cette exposition ; un autre thread qui pousserait un événement du
  // même type ne fait pas tourner la lecture.
  if (event->type != SDL_EVENT_WINDOW_EXPOSED || std::this_thread::get_id() != self->owner_) return true;
  if (self->inside_ || suspended > 0) return true;
  const std::int64_t now = host_now_us();
  if (!modal_turn_due(event->type, event->window.data1, event->window.windowID == self->window_, now,
                      self->last_us_))
    return true;
  self->inside_ = true;
  self->turn_();
  self->inside_ = false;
  self->last_us_ = host_now_us();
  ++self->turns_;
  return true;
}

}  // namespace ajp::app::play
