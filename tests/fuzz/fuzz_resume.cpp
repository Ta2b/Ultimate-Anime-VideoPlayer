// Mémoire de reprise (resume.tsv) : un fichier quelconque à la place du sien, où « @ » devient le
// chemin d'une vraie vidéo (le fuzz trouve ainsi des lignes qui la nomment). La lecture et la
// réécriture ne plantent pas ; une position retenue se relit, quoi que le fichier contînt.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <unistd.h>

#include "app/play/resume.h"
#include "fuzz_check.h"
#include "fuzz_targets.h"

namespace {

struct Place {
  std::filesystem::path dir;
  std::filesystem::path store;
  std::filesystem::path video;
};

const Place& place() {
  static const Place p = [] {
    Place made;
    made.dir = std::filesystem::temp_directory_path() / ("ajp-fuzz-resume-" + std::to_string(getpid()));
    std::filesystem::create_directories(made.dir);
    made.store = made.dir / "resume.tsv";
    made.video = made.dir / "vidéo.mkv";
    std::ofstream(made.video, std::ios::binary) << "1234567";
    return made;
  }();
  return p;
}

}  // namespace

int ajp::fuzz::resume(const std::uint8_t* data, std::size_t size) {
  namespace play = ajp::app::play;
  const Place& p = place();
  const std::string video = std::filesystem::weakly_canonical(p.video).string();
  std::string text;
  for (const char c : std::string_view(reinterpret_cast<const char*>(data), size)) {
    if (c == '@') {
      text += video;
    } else {
      text += c;
    }
  }
  std::ofstream(p.store, std::ios::binary | std::ios::trunc) << text;
  static_cast<void>(play::resume_position(p.store, p.video));
  constexpr std::int64_t kPosition = play::kResumeMinUs + 1'000'000;
  ajp::fuzz::require(play::remember_position(p.store, p.video, kPosition, 0).has_value(),
                     "la position s'écrit");
  ajp::fuzz::require(play::resume_position(p.store, p.video) == kPosition, "la position retenue se relit");
  return 0;
}
