// Chapitres Matroska (B35-03) : un fichier quelconque à la place d'un MKV, lu par read_chapter_layout. Le
// lecteur EBML maison lit chaque MKV ouvert ; il ne plante pas, ne lit pas au-delà de son fichier, et les
// parties qu'il rend se tiennent (début avant fin, jamais négatif).
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>

#include "fuzz_check.h"
#include "fuzz_targets.h"
#include "media/mkv_chapters.h"

namespace {

const std::filesystem::path& place() {
  static const std::filesystem::path path = [] {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("kirari-fuzz-mkv-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    // Le dossier ne reste pas : une campagne longue en laissait un par processus.
    std::atexit(+[] {
      std::error_code ignored;
      std::filesystem::remove_all(
          std::filesystem::temp_directory_path() / ("kirari-fuzz-mkv-" + std::to_string(getpid())), ignored);
    });
    return dir / "entrée.mkv";
  }();
  return path;
}

}  // namespace

int kirari::fuzz::mkv_chapters(const std::uint8_t* data, std::size_t size) {
  const std::filesystem::path& path = place();
  std::ofstream(path, std::ios::binary | std::ios::trunc)
      .write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
  const kirari::media::ChapterLayout layout = kirari::media::read_chapter_layout(path);
  for (const kirari::media::ChapterPart& part : layout.parts) {
    kirari::fuzz::require(part.start_us >= 0, "le début d'une partie n'est pas négatif");
    kirari::fuzz::require(part.end_us >= part.start_us, "une partie finit après son début");
  }
  kirari::fuzz::require(layout.parts.empty() || layout.ordered,
                        "des parties n'existent que pour une édition ordonnée");
  return 0;
}
