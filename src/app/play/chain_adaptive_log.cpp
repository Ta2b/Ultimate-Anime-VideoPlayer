#include "app/play/chain_adaptive_log.h"

#include <algorithm>
#include <ctime>
#include <string>
#include <string_view>
#include <system_error>

#include "app/play/chain.h"

namespace ajp::app::play {

namespace {

// Au-delà de 32 Mo (une soixantaine d'heures), le journal passe en .old et repart de zéro.
constexpr std::uintmax_t kMaxBytes = 32u << 20u;

constexpr std::string_view kHeader =
    "unix_s\tfichier\tposition_s\tmoyenne\tp05\tp50\tp95\tp995\tpart_noir\tpart_clair\tnoirs\tcouleurs\t"
    "eclats\taplats\tetendues\tlimiteur\tgris_moyen\tcontraste\tsaturation\tadaptatif\tt\ttv\tserie\t"
    "luma_serie\tsource\tsans_geste_s";

void retire(const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::path old = path;
  old += ".old";
  std::filesystem::rename(path, old, error);
}

// Ouvre le journal en ajout. Un fichier trop gros, ou d'un ancien format (autre en-tête), passe en .old.
std::shared_ptr<std::ofstream> open_log(const std::filesystem::path& path) {
  std::error_code error;
  if (const std::uintmax_t size = std::filesystem::file_size(path, error); !error && size > kMaxBytes)
    retire(path);
  if (std::filesystem::exists(path, error)) {
    std::ifstream in(path);
    std::string first;
    std::getline(in, first);
    in.close();
    if (first != kHeader) retire(path);
  }
  const bool fresh = !std::filesystem::exists(path, error);
  auto out = std::make_shared<std::ofstream>(path, std::ios::app);
  if (!*out) return nullptr;
  if (fresh) *out << kHeader << '\n';
  return out;
}

}  // namespace

void adaptive_record(Chain& to, const gpu::KernelOps::SceneStats& stats, std::int64_t now_us) {
  AdaptiveLog& log = to.adaptive_log;
  if (log.path.empty() || to.session.view().paused || now_us - log.last_us < 1'000'000) return;
  log.last_us = now_us;
  // Environ 140 octets par ligne : la taille se revoit toutes les 4096 lignes (une heure de lecture).
  if (log.out && ++log.lines % 4096 == 0) {
    log.out->flush();
    std::error_code error;
    if (const std::uintmax_t size = std::filesystem::file_size(log.path, error); !error && size > kMaxBytes)
      log.out.reset();
  }
  if (!log.out) log.out = open_log(log.path);
  if (!log.out) {
    log.path.clear();  // dossier en lecture seule : plus d'essai pour cette chaîne
    return;
  }
  const PlaySettings& s = to.running;
  const render::GradeSettings& g = s.grade_values;
  const std::int64_t quiet_us =
      to.learn.still_us > 0 ? std::max<std::int64_t>(now_us - to.learn.still_us, 0) : -1;
  *log.out << std::time(nullptr) << '\t' << to.file.filename().string() << '\t'
           << static_cast<double>(std::max<std::int64_t>(to.session.view().position_us, 0)) / 1e6 << '\t'
           << stats.mean << '\t' << stats.p05 << '\t' << stats.p50 << '\t' << stats.p95 << '\t' << stats.p995
           << '\t' << stats.dark << '\t' << stats.bright << '\t' << stats.blue << '\t' << stats.chroma << '\t'
           << g.shadows << '\t' << g.vibrance_rtx << '\t' << g.highlights_rtx << '\t' << g.area_rtx << '\t'
           << g.expand << '\t' << g.guard << '\t' << s.true_hdr_gray << '\t' << s.true_hdr_contrast << '\t'
           << s.true_hdr_saturation << '\t' << (g.adaptive ? 1 : 0) << '\t' << to.adaptive_t << '\t'
           << to.adaptive_tv << '\t' << to.adaptive_bs << '\t' << to.series_luma << '\t' << "plan" << '\t'
           << (quiet_us < 0 ? -1.0 : static_cast<double>(quiet_us) / 1e6) << '\n';
}

}  // namespace ajp::app::play
