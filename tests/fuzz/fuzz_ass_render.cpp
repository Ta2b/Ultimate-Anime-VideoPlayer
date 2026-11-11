// Scripts ASS venus de l'extérieur (un .ass à côté de la vidéo, une piste ASS d'un MKV) : lus par
// SubtitleTrack::from_ass, puis rendus par le libass corrigé (rendu sur plusieurs fils,
// ports/libass/rendu-sur-plusieurs-fils.patch) à plusieurs instants et à deux résolutions. Ce code-là n'avait
// aucune cible : ni OSS-Fuzz (le correctif vient d'un fork), ni TSan (libass n'est pas instrumenté, sauf dans
// la tâche TSan de ci.yml). Un script forgé (\t, \clip de dessins, \blur et \be extrêmes, des milliers
// d'événements simultanés) ne doit ni lire ni écrire hors d'un tampon, ni courir.
//
// Le moteur de rendu est gardé d'une entrée à l'autre (sa création, avec les polices du système, est lente) ;
// il oublie son dernier contenu entre deux entrées, et la piste, elle, est neuve à chaque entrée.
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "fuzz_check.h"
#include "fuzz_targets.h"
#include "subs/ass_library.h"
#include "subs/atlas_pool.h"
#include "subs/subtitle_renderer.h"
#include "subs/subtitle_track.h"

namespace {

// Quatre fils : ce que le lecteur demande (src/app/play/pipeline.h), donc le code multi-fils du correctif.
constexpr int kRenderThreads = 4;

struct Engine {
  ajp::subs::AssLibrary library;
  ajp::subs::SubtitleRenderer renderer;
};

Engine& engine() {
  static Engine instance = [] {
    auto library = ajp::subs::AssLibrary::create();
    ajp::fuzz::require(library.has_value(), "la bibliothèque libass se crée");
    auto renderer = ajp::subs::SubtitleRenderer::create(*library, {.render_threads = kRenderThreads});
    ajp::fuzz::require(renderer.has_value(), "le moteur de rendu se crée");
    return Engine{std::move(*library), std::move(*renderer)};
  }();
  return instance;
}

}  // namespace

int ajp::fuzz::ass_render(const std::uint8_t* data, std::size_t size) {
  Engine& e = engine();
  const std::string_view script(reinterpret_cast<const char*>(data), size);
  auto track = ajp::subs::SubtitleTrack::from_ass(e.library, script);
  if (!track) return 0;

  // Les atlas relâchés attendent ici, comme dans la boîte du thread subs ; déclarés avant la réserve qui les
  // rend.
  std::vector<std::unique_ptr<ajp::subs::SubtitleAtlas>> returned;
  ajp::subs::AtlasPool pool{
      [&returned](std::unique_ptr<ajp::subs::SubtitleAtlas> atlas) { returned.push_back(std::move(atlas)); }};
  constexpr std::array<std::int64_t, 5> kInstantsUs{0, 750'000, 2'500'000, 10'000'000, 600'000'000};
  constexpr std::array<std::array<int, 2>, 2> kSizes{{{640, 360}, {1920, 1080}}};
  for (const auto& [width, height] : kSizes) {
    for (const std::int64_t pts : kInstantsUs) {
      for (auto& atlas : returned) pool.give_back(std::move(atlas));
      returned.clear();
      const ajp::subs::RenderRequest request{
          .pts_us = pts, .epoch = 0, .target = {.width = width, .height = height}};
      // Un refus (atlas plein, cible hors plage) est une réponse permise ; seul un arrêt du processus est un
      // défaut.
      (void)e.renderer.render(*track, request, pool);
    }
  }
  // L'atlas que le moteur tient encore rentre dans returned avant que la réserve et la piste disparaissent.
  e.renderer.forget();
  return 0;
}
