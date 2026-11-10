// La place des sous-titres dans la sortie : la cible des demandes à subs, et la demande refaite pour l'image
// à l'écran quand cette place change.
#include "app/play/controls.h"
#include "app/play/session.h"

namespace ajp::app::play {

subs::RenderTarget Session::subtitle_target(OutputSize size) {
  placed_size_ = size;
  return {.width = size.width,
          .height = size.height,
          .video_width = config_.video_width,
          .video_height = config_.video_height,
          .sample_aspect = config_.video_sar,
          .video_matrix = config_.video_matrix,
          .font_scale = config_.subtitle_scale,
          .line_lift = line_lift_};
}

// Les réponses gardées sont à l'ancienne place : l'image à l'écran redemande les siennes, en pause aussi
// (sans cela une réplique reste étirée à l'ancienne taille, ou sous la barre, jusqu'à la reprise) ; les
// images suivantes sont demandées à la nouvelle.
void Session::place_subtitles(const UiFrame& ui) {
  const OutputSize size = screen_.size();
  if (!config_.subtitles || size.width <= 0 || size.height <= 0) return;
  const float lift = line_lift(ui.draw != nullptr ? ui.bottom : 0, size.height);
  if (lift == line_lift_ && (size == placed_size_ || placed_size_ == OutputSize{})) return;
  line_lift_ = lift;
  placed_size_ = size;  // une seule demande par changement, même si elle doit attendre une réponse
  if (current_) request_subtitles(*current_);
}

}  // namespace ajp::app::play
