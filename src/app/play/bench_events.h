// Ce que l'enfant d'un banc dit à son parent (P2.11) : une ligne JSON par événement sur sa sortie standard,
//   ajp-bench {"event":"build"}
//   ajp-bench {"event":"refusal","code":"DeviceLost","detail":"…"}
// que le parent lit champ par champ, au lieu de chercher des mots dans le journal de l'enfant.
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace ajp::app::play {

struct ChildBenchEvent {
  std::string
      event;  // « build » : un moteur se construit ; « refusal » : la lecture s'est arrêtée sur un refus
  std::string code;  // name(RefusalCode) d'un refus
  std::string detail;
  friend bool operator==(const ChildBenchEvent&, const ChildBenchEvent&) = default;
};

inline constexpr std::string_view kBenchEventTag = "ajp-bench ";

std::string bench_event_line(const ChildBenchEvent& event);  // sans fin de ligne
// L'événement d'une ligne de l'enfant ; absent pour toute autre ligne, ou un JSON mal formé.
std::optional<ChildBenchEvent> parse_bench_event(std::string_view line);

// Dans l'enfant (--bench-out) : écrit l'événement sur la sortie standard, vidée aussitôt. Sans effet tant que
// set_bench_child n'a pas été appelé. Sûr depuis tout fil.
void set_bench_child(bool child);
void tell_bench_parent(const ChildBenchEvent& event);

}  // namespace ajp::app::play
