// Chapitres Matroska (B35-03) : un fichier quelconque à la place d'un MKV, lu par read_chapter_layout. Le
// lecteur EBML maison lit chaque MKV ouvert ; il ne plante pas, ne lit pas au-delà de son fichier, et les
// parties qu'il rend se tiennent (début avant fin, jamais négatif).
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "fuzz_check.h"
#include "fuzz_targets.h"
#include "media/mkv_chapters.h"

namespace {

const std::filesystem::path& place() {
  static const std::filesystem::path path = [] {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("ajp-fuzz-mkv-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    return dir / "entrée.mkv";
  }();
  return path;
}

}  // namespace

int ajp::fuzz::mkv_chapters(const std::uint8_t* data, std::size_t size) {
  const std::filesystem::path& path = place();
  std::ofstream(path, std::ios::binary | std::ios::trunc)
      .write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
  const ajp::media::ChapterLayout layout = ajp::media::read_chapter_layout(path);
  for (const ajp::media::ChapterPart& part : layout.parts) {
    ajp::fuzz::require(part.start_us >= 0, "le début d'une partie n'est pas négatif");
    ajp::fuzz::require(part.end_us >= part.start_us, "une partie finit après son début");
  }
  ajp::fuzz::require(layout.parts.empty() || layout.ordered,
                     "des parties n'existent que pour une édition ordonnée");
  return 0;
}
