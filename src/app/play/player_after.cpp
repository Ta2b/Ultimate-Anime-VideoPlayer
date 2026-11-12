#include "app/play/player_after.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>

#include "app/play/pumped_task.h"
#include "core/clock.h"
#include "core/log.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

// Les E/S du dossier de la vidéo (fichier qui grandit, épisode suivant) : au-delà, laissées à leur sort, la
// lecture continue comme si elles n'avaient rien trouvé.
constexpr std::chrono::milliseconds kFolderIoLimit{5000};
// Une source qui change de taille 4 fois en une minute est refusée : la chaîne se rouvrirait sans fin.
constexpr std::int64_t kResizeWindowUs = 60'000'000;
constexpr std::size_t kResizesRefused = 4;

}  // namespace

void plan_follow_up(Player& p, Chain& chain, const Session& session, const GrowingFile& opened_file,
                    std::optional<Refusal>& failure) {
  const auto log = log_channel("app");
  // La source a changé de taille en cours de lecture : la chaîne se rouvre à la nouvelle taille, au même
  // endroit ; au-delà de 4 réouvertures par minute, c'est un refus.
  if (session.resize_asked() && !chain.reopen && !failure) {
    std::deque<std::int64_t>& resizes = p.resizes;
    const std::int64_t now = host_now_us();
    while (!resizes.empty() && now - resizes.front() > kResizeWindowUs) resizes.pop_front();
    if (resizes.size() >= kResizesRefused) {
      failure = Refusal{RefusalCode::SourceResized, "la source change de taille trop souvent"};
    } else {
      resizes.push_back(now);
      log->info("la source a changé de taille : réouverture de la chaîne");
      chain.reopen = chain.running;
    }
  }
  // B13-10, B29-08, B31-12 : l'attente du fichier qui grandit et le parcours du dossier se font hors du
  // thread de la fenêtre, pompée, et bornés (un NAS lent ou muet ne la gèle plus).
  if (!chain.reopen && !failure && !session.quit_asked() &&
      run_pumped_for<bool>("growing-file", kFolderIoLimit, true, [file = p.options.file, opened_file] {
        return opened_file.grew(file);
      }).value_or(false)) {
    log->info("le fichier grandit encore : relu au même endroit");
    chain.reopen = chain.running;  // A29-03 : comme une réouverture de la chaîne, pistes gardées
  }
  // Fin de la vidéo, sans demande de quitter : l'épisode suivant du dossier, si le réglage le veut.
  if (!chain.reopen && !failure && !session.quit_asked() && !p.options.bench_out &&
      chain.running.autoplay_next && !p.options.file.empty()) {
    if (const auto next = run_pumped_for<std::optional<std::filesystem::path>>(
                              "next-episode", kFolderIoLimit, true,
                              [file = p.options.file] { return sibling_episode(file, 1); })
                              .value_or(std::nullopt)) {
      log->info("épisode suivant : {}", prepare::utf8_path(next->filename()));
      chain.next_file = *next;
      chain.reopen = chain.running;
    }
  }
}

}  // namespace ajp::app::play
