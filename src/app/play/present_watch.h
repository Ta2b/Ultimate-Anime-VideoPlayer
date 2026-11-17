// Le garde de la présentation. Sur certains pilotes, le retour de présentation (present_level.h :
// present_id2, present_wait2, present timing) fige le thread de la fenêtre dans une acquisition ou une
// présentation : RTX 4070 Laptop, pilote NVIDIA 621.01 sous Windows, « Ne répond pas » dès l'accueil, plus
// rien au journal, alors que le même poste répond avec --debug --no-present. Un thread à part voit l'appel
// qui ne revient pas, le dit au journal, écrit present-stall.txt (la carte et le pilote), prévient et sort.
// Le lancement suivant présente sans retour sur cette carte et ce pilote, comme --no-present ; un autre
// pilote ou une autre carte le retente.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

namespace ajp::app::play {

// Au-delà, un appel de la swapchain est figé : une image prend 4 à 42 ms, une swapchain refaite ou un
// changement de mode d'écran moins d'une seconde.
inline constexpr std::chrono::milliseconds kPresentStallLimit{4000};

// La carte et le pilote, comme le fichier les garde : « NVIDIA GeForce RTX 4070 Laptop GPU · 9b404000 »
// (driverVersion de Vulkan, brut). Pure.
std::string present_stall_key(std::string_view device, std::uint32_t driver_version);
// La première ligne du fichier ; vide s'il manque ou ne se lit pas.
std::string read_present_stall(const std::filesystem::path& file);

class PresentWatch {
 public:
  // file : present-stall.txt ; key : present_stall_key de l'appareil ; tell : une boîte avant la sortie
  // (pas pour un essai sans fenêtre visible).
  PresentWatch(std::filesystem::path file, std::string key, bool tell);
  ~PresentWatch();
  PresentWatch(const PresentWatch&) = delete;
  PresentWatch& operator=(const PresentWatch&) = delete;
  PresentWatch(PresentWatch&&) = delete;
  PresentWatch& operator=(PresentWatch&&) = delete;

  // Un appel du pilote qui peut se figer, gardé tant que le Scope vit ; watch nul : rien.
  class Scope {
   public:
    Scope(PresentWatch* watch, const char* stage);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope(Scope&&) = delete;
    Scope& operator=(Scope&&) = delete;

   private:
    PresentWatch* watch_;
  };

 private:
  void run(const std::stop_token& stop);
  [[noreturn]] void stalled(const char* stage, std::int64_t for_us);

  std::filesystem::path file_;
  std::string key_;
  bool tell_ = false;
  std::atomic<std::int64_t> since_us_{0};  // l'entrée dans l'appel gardé ; 0 : aucun
  std::atomic<const char*> stage_{nullptr};
  std::mutex mutex_;
  std::condition_variable_any wake_;
  std::jthread thread_;  // dernier : arrêté et attendu avant le reste
};

}  // namespace ajp::app::play
