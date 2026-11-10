// La lecture continue pendant la boucle modale du système (A10-05). Sous Windows, tant que l'utilisateur
// tient la barre de titre ou un bord de la fenêtre, DefWindowProc garde le thread : SDL_PollEvent ne rend pas
// la main et la boucle du lecteur ne tourne plus (image figée, son qui continue, ring qui se remplit, images
// jetées au relâchement). macOS fait de même pendant un redimensionnement. SDL3 envoie alors des
// SDL_EVENT_WINDOW_EXPOSED marqués « redimensionnement en direct » (data1 = 1) aux observateurs
// d'événements, sur le thread principal, pour qu'on dessine de là : ce garde y fait un tour de lecture.
#pragma once

#include <cstdint>
#include <functional>
#include <thread>

#include <SDL3/SDL_events.h>

namespace ajp::app::play {

class Pipeline;
class Session;

// Le tour de lecture fait depuis la boucle modale : les messages des étages vont à la session (le ring se
// vide, les étages ne s'arrêtent pas), puis elle choisit et présente l'image due. Sans interface : la barre
// et le menu ne se redessinent pas tant que la fenêtre est tenue, la vidéo et les sous-titres si. Une
// erreur de présentation n'est pas traitée ici : le tour suivant de la boucle la retrouve.
void modal_turn(Pipeline& pipeline, Session& session);

// Faut-il un tour de lecture pour cet événement ? Seulement une exposition en direct de notre fenêtre, et
// pas plus souvent que min_gap_us : le système en envoie à son rythme, pas à celui de l'écran.
inline constexpr std::int64_t kModalTurnGapUs = 2'000;
constexpr bool modal_turn_due(Uint32 type, Sint32 live, bool our_window, std::int64_t now_us,
                              std::int64_t last_us) {
  return type == SDL_EVENT_WINDOW_EXPOSED && live == 1 && our_window && now_us - last_us >= kModalTurnGapUs;
}

// Une attente d'événements faite à l'intérieur d'un tour de lecture (l'écran, fenêtre cachée) : la boucle
// modale peut commencer là aussi, et un tour dans le tour serait une réentrance de Session::tick. Les tours
// de la boucle modale sont suspendus le temps de cette attente.
void wait_events_no_modal_turn(Sint32 timeout_ms);

class ModalRedraw {
 public:
  // turn : un tour de lecture (messages des étages, puis Session::tick), appelé sur le thread qui a créé le
  // garde, depuis SDL_PollEvent. Il ne doit pas vider la file des événements.
  ModalRedraw(SDL_Window* window, std::function<void()> turn);
  ModalRedraw(const ModalRedraw&) = delete;
  ModalRedraw& operator=(const ModalRedraw&) = delete;
  ~ModalRedraw();

  std::uint64_t turns() const { return turns_; }  // tours faits depuis la boucle modale

 private:
  static bool on_event(void* context, SDL_Event* event);

  SDL_WindowID window_ = 0;
  std::function<void()> turn_;
  std::thread::id owner_ = std::this_thread::get_id();
  std::int64_t last_us_ = 0;
  std::uint64_t turns_ = 0;
  bool inside_ = false;  // un tour en cours : pas de tour dans le tour
  friend void wait_events_no_modal_turn(Sint32 timeout_ms);
  static int suspended;  // attentes en cours à l'intérieur d'un tour (thread principal seulement)
};

}  // namespace ajp::app::play
