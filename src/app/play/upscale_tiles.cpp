// Les tuiles qui bougent (upscale_tiles.h).
#include "app/play/upscale_tiles.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <utility>

namespace kirari::app::play {
namespace {

// Les tuiles de la grille que touche un rectangle de pixels de travail [x0, x1) × [y0, y1), par leur indice.
template <class F>
void for_tiles(const kernels::SpanTileGrid& g, int x0, int y0, int x1, int y1, F&& f) {
  if (x1 <= 0 || y1 <= 0) return;
  const int c0 = std::max(0, x0 / g.tile_w), c1 = std::min(g.cols - 1, (x1 - 1) / g.tile_w);
  const int r0 = std::max(0, y0 / g.tile_h), r1 = std::min(g.rows - 1, (y1 - 1) / g.tile_h);
  for (int r = r0; r <= r1; ++r)
    for (int c = c0; c <= c1; ++c)
      f(static_cast<std::size_t>(r) * static_cast<std::size_t>(g.cols) + static_cast<std::size_t>(c));
}

// La somme d'un bloc et de l'écart d'une paire, bornée. La mesure tronque l'écart au 1/256 de niveau
// (pair_diff) : un de plus le majore, même pour une dérive lente que chaque paire lirait nulle.
std::uint32_t add_pair(std::uint32_t sum, std::uint32_t deviation) {
  const std::uint64_t total = std::uint64_t{sum} + std::uint64_t{deviation} + 1U;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(total, std::numeric_limits<std::uint32_t>::max()));
}

// La mesure a une géométrie : des blocs, un côté, des lignes entières.
bool measured(const PairBlocks& blocks) {
  return blocks.cols > 0 && blocks.side_px > 0.0F && !blocks.deviation.empty() &&
         blocks.deviation.size() % blocks.cols == 0;
}

// Les tuiles dont tout le voisinage (cols, rows) est dans mask : l'érosion, complément de la dilatation du
// complément ; hors de la grille compte comme dedans.
std::vector<std::uint8_t> eroded(const std::vector<std::uint8_t>& mask, const kernels::SpanTileGrid& g,
                                 int cols, int rows) {
  std::vector<std::uint8_t> out(mask.size());
  for (std::size_t i = 0; i < mask.size(); ++i) out[i] = mask[i] != 0 ? 0 : 1;
  out = kernels::dilate_span_tiles(out, g.cols, g.rows, cols, rows);
  for (std::uint8_t& v : out) v = v != 0 ? 0 : 1;
  return out;
}

}  // namespace

void SpanTiles::set_grid(const kernels::SpanTileGrid& grid) {
  grid_ = grid;
  span_ = true;
  age_.assign(grid.usable() ? static_cast<std::size_t>(grid.count()) : 0U, 0);
  sums_.clear();
  base_.clear();
  block_cols_ = 0;
  block_px_ = 0.0F;
  since_full_ = 0;
  known_ = false;
  unmeasured_ = false;
}

template <class Visit>
void SpanTiles::tiles_under(std::size_t b, Visit&& visit) const {
  const auto col = static_cast<float>(b % block_cols_), row = static_cast<float>(b / block_cols_);
  // Arrondi vers l'extérieur : un bloc à cheval touche les deux tuiles.
  for_tiles(grid_, static_cast<int>(std::floor(col * block_px_)),
            static_cast<int>(std::floor(row * block_px_)),
            static_cast<int>(std::ceil((col + 1.0F) * block_px_)),
            static_cast<int>(std::ceil((row + 1.0F) * block_px_)), std::forward<Visit>(visit));
}

TilePlan SpanTiles::full(const PairBlocks& blocks) {
  std::ranges::fill(age_, 0);
  // Après une image entière, chaque somme est nulle à toute géométrie : celle de la mesure, ou sans mesure
  // celle d'avant (une paire non relue ne coûte que son image).
  if (measured(blocks)) {
    block_cols_ = blocks.cols;
    block_px_ = blocks.side_px;
    sums_.assign(blocks.deviation.size(), 0U);
  } else {
    std::ranges::fill(sums_, 0U);
  }
  base_.assign(sums_.size(), 0U);
  since_full_ = 0;
  known_ = true;
  unmeasured_ = false;
  return TilePlan{};
}

bool SpanTiles::fits(const PairBlocks& blocks) const {
  return measured(blocks) && blocks.cols == block_cols_ && blocks.side_px == block_px_ &&
         blocks.deviation.size() == sums_.size();
}

bool SpanTiles::adopt(const PairBlocks& blocks) {
  if (!measured(blocks) || !std::ranges::all_of(sums_, [](std::uint32_t s) { return s == 0; })) return false;
  block_cols_ = blocks.cols;
  block_px_ = blocks.side_px;
  sums_.assign(blocks.deviation.size(), 0U);
  base_.assign(sums_.size(), 0U);
  return true;
}

std::uint32_t SpanTiles::bound(std::size_t b, const PairBlocks& blocks, const PairBlocks& gap) const {
  const std::uint32_t summed = add_pair(sums_[b], blocks.deviation[b]);
  if (!fits(gap) || base_.size() != sums_.size()) return summed;
  return std::min(summed, add_pair(base_[b], gap.deviation[b]));
}

std::uint64_t* SpanTiles::whole_reason(const TileFacts& facts, bool new_recipe) {
  if (facts.stale != TileStale::None) return &stale_[static_cast<std::size_t>(facts.stale)];
  if (!known_) return unmeasured_ ? &no_measure_ : &stale_[static_cast<std::size_t>(TileStale::Restart)];
  if (new_recipe) return &recipe_;
  if (!fits(facts.blocks) && !adopt(facts.blocks)) return &no_measure_;
  if (since_full_ >= kFullEvery) return &refresh_;
  return nullptr;
}

void SpanTiles::held(const PairBlocks& blocks, const PairBlocks& gap) {
  if (!active() || !known_) return;
  ++since_full_;
  for (int& age : age_) ++age;
  if (!fits(blocks) && !adopt(blocks)) {
    known_ = false;  // un écart non mesuré : la prochaine image calculée est entière, « sans mesure »
    unmeasured_ = true;
    return;
  }
  for (std::size_t b = 0; b < sums_.size(); ++b) sums_[b] = bound(b, blocks, gap);
}

std::pair<int, int> SpanTiles::reach(int spread_px) const {
  const auto along = [spread_px](int network, int tile) {
    return std::min(network, kernels::span_reach_tiles(kCoreReachPx, tile)) +
           kernels::span_reach_tiles(spread_px, tile);
  };
  return {along(grid_.reach_cols, grid_.tile_w), along(grid_.reach_rows, grid_.tile_h)};
}

TilePlan SpanTiles::plan(const TileFacts& facts) {
  if (!active()) return TilePlan{};
  ++planned_;
  ++since_full_;
  const bool new_recipe = known_ && facts.input_key != input_key_;
  input_key_ = facts.input_key;
  if (std::uint64_t* reason = whole_reason(facts, new_recipe); reason != nullptr) {
    ++*reason;
    return full(facts.blocks);
  }
  // Chaque bloc reçoit l'écart de la paire ; à recalculer : les tuiles sous un bloc au-delà du seuil, et
  // celles qui ont trop vieilli.
  std::vector<Cause> why(age_.size(), Cause::None);
  for (std::size_t i = 0; i < age_.size(); ++i)
    if (age_[i] + 1 > kMaxAge) why[i] = Cause::Age;
  for (std::size_t b = 0; b < sums_.size(); ++b) {
    sums_[b] = bound(b, facts.blocks, facts.gap);
    if (kernels::is_held(sums_[b], kThreshold)) continue;
    tiles_under(b, [&why](std::size_t i) {
      if (why[i] == Cause::None) why[i] = Cause::Block;
    });
  }
  std::vector<std::uint8_t> raw(age_.size(), 0);
  std::array<std::uint64_t, kCauseCount> causes{};
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = why[i] != Cause::None ? 1 : 0;
    ++causes[static_cast<std::size_t>(why[i])];
  }
  // Le cœur : ce qui a changé, et toute sortie qui le voit ; la région : le cœur, et tout ce qu'il lit.
  const auto [reach_cols, reach_rows] = reach(facts.spread_px);
  std::vector<std::uint8_t> core =
      kernels::dilate_span_tiles(raw, grid_.cols, grid_.rows, reach_cols, reach_rows);
  const std::vector<std::uint8_t> wide =
      kernels::dilate_span_tiles(core, grid_.cols, grid_.rows, grid_.reach_cols, grid_.reach_rows);
  const int wide_count = kernels::count_span_tiles(wide);
  if (static_cast<double>(wide_count) > kMaxShare * static_cast<double>(age_.size())) {
    ++too_wide_;
    for (std::size_t k = 0; k < kCauseCount; ++k) wide_causes_[k] += causes[k];
    wide_region_ += static_cast<std::uint64_t>(wide_count);
    wide_seen_ += age_.size();
    return full(facts.blocks);
  }
  // Toute sortie dont le champ du réseau tient dans la région calculée est exacte : elle s'écrit aussi, sans
  // rien calculer de plus (le moteur ne dilate pas au-delà de wide), et repart de cette image.
  core = eroded(wide, grid_, grid_.reach_cols, grid_.reach_rows);
  for (std::size_t i = 0; i < age_.size(); ++i) age_[i] = core[i] != 0 ? 0 : age_[i] + 1;
  // Un bloc repart de zéro quand toute sortie qui le voit est recalculée : ses tuiles et tout ce qui est à
  // leur portée. Recalculer ses seules tuiles ne suffit pas : une voisine reprise a vu son entrée d'avant.
  const std::vector<std::uint8_t> renewed = eroded(core, grid_, reach_cols, reach_rows);
  for (std::size_t b = 0; b < sums_.size(); ++b) {
    bool all = true;
    tiles_under(b, [&all, &renewed](std::size_t i) { all = all && renewed[i] != 0; });
    if (all) sums_[b] = 0;
  }
  base_ = sums_;
  last_ = Counted{.core = static_cast<std::uint64_t>(kernels::count_span_tiles(core)),
                  .wide = static_cast<std::uint64_t>(kernels::count_span_tiles(kernels::dilate_span_tiles(
                      core, grid_.cols, grid_.rows, grid_.reach_cols, grid_.reach_rows)))};
  ++tiled_;
  tiles_done_ += last_.core;
  tiles_wide_ += last_.wide;
  tiles_seen_ += age_.size();
  return TilePlan{.tiled = true, .core = std::move(core)};
}

void SpanTiles::refused() {
  ++refused_;
  // L'image se calcule entière : elle ne compte plus parmi celles par tuiles.
  if (tiled_ > 0) {
    --tiled_;
    tiles_done_ -= last_.core;
    tiles_wide_ -= last_.wide;
    tiles_seen_ -= age_.size();
  }
  last_ = Counted{};
  // La géométrie de la mesure reste : l'état repart de cette image entière.
  std::ranges::fill(age_, 0);
  std::ranges::fill(sums_, 0U);
  base_.assign(sums_.size(), 0U);
  since_full_ = 0;
  known_ = true;
  unmeasured_ = false;
}

void SpanTiles::checked(std::uint32_t worst_block, double mean_levels) {
  ++checks_;
  check_worst_ = std::max(check_worst_, worst_block);
  check_mean_ = std::max(check_mean_, mean_levels);
}

namespace {

double share(std::uint64_t part, std::uint64_t whole) {
  return whole > 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
}

}  // namespace

std::string SpanTiles::summary() const {
  if (!span_ || !enabled) return {};
  if (!grid_.usable()) return "SPAN par tuiles : aucune, ce moteur ne calcule que l'image entière";
  std::string line = std::format("SPAN par tuiles : {} images sur {} calculées ({:.1f} %)", tiled_, planned_,
                                 share(tiled_, planned_));
  if (tiled_ > 0)
    line += std::format(", {:.1f} % des tuiles recalculées ({:.1f} % avec le halo)",
                        share(tiles_done_, tiles_seen_), share(tiles_wide_, tiles_seen_));
  std::string wide;
  if (too_wide_ > 0) {
    const std::uint64_t raw = wide_causes_[1] + wide_causes_[2];
    wide = std::format(
        "région trop large ({:.1f} % des tuiles en cause, {:.1f} % de région, blocs {:.0f} %, "
        "âge {:.0f} %)",
        share(raw, wide_seen_), share(wide_region_, wide_seen_), share(wide_causes_[1], raw),
        share(wide_causes_[2], raw));
  }
  const auto stale = [this](TileStale why) { return stale_[static_cast<std::size_t>(why)]; };
  const std::array<std::pair<std::uint64_t, const char*>, 9> reasons{{
      {no_measure_, "sans mesure des blocs"},
      {too_wide_, wide.c_str()},
      {refused_, "refus du moteur"},
      {stale(TileStale::Graded), "sortie d'avant gradée ou plaquée sur place"},
      {stale(TileStale::Overwritten), "sortie d'avant écrasée (image créée, source sautée)"},
      {stale(TileStale::Chain), "chaîne sans tuiles (export, YUV, TemporalFix, Smooth)"},
      {stale(TileStale::Restart), "début, saut ou image abîmée"},
      {recipe_, "entrée de SPAN changée"},
      {refresh_, "rafraîchissement"},
  }};
  std::string whole;
  for (const auto& [count, why] : reasons) {
    if (count == 0) continue;
    whole += std::format("{}{} {}", whole.empty() ? "" : ", ", count, why);
  }
  line += " ; images entières : " + (whole.empty() ? std::string("aucune") : whole);
  if (checks_ > 0)
    line += std::format(
        " ; contrôle sur {} images contre un calcul complet : pire bloc {:.2f} niveau, écart moyen {:.3f} "
        "niveau au plus",
        checks_, static_cast<double>(check_worst_) / 256.0, check_mean_);
  return line;
}

}  // namespace kirari::app::play
