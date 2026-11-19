// Le choix d'un mélange de morceaux natifs (RIFE natif, docs/rife-natif.md ; GMFSS natif,
// docs/gmfss-natif.md). Chaque morceau d'une chaîne prend une option parmi quelques-unes (un réseau natif à
// telles sommes, ou TensorRT pour RIFE), et chaque option a son temps seul. L'effet d'une option est mesuré
// sur la chaîne, ce morceau seul hors de la base (TensorRT pour RIFE, le natif f32 pour GMFSS) : l'erreur
// quadratique de la sortie contre celle de la base, celle que les morceaux suivants font de l'erreur du
// morceau. L'erreur prévue d'un mélange, groupe par groupe, est celle de la base contre la référence, plus
// ses effets : de leur somme (erreurs indépendantes) au carré de la somme de leurs racines (erreurs de même
// signe aux mêmes pixels), selon une corrélation qui part de 0. Le mélange retenu est le plus rapide (la
// somme des temps seuls) dont l'erreur tient le seuil, vérifié sur la chaîne ; un mélange déjà mesuré (la
// base, un morceau seul hors d'elle, une graine) n'a rien à prévoir. Une vérification manquée recale la
// corrélation (sans jamais descendre) et relève de kSlack la barre des mélanges prévus ; au dernier essai,
// seul un mélange déjà mesuré peut être pris. En-tête sans CUDA : les tests sur hôte le lisent.
#pragma once

#include <cstddef>
#include <expected>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace kirari::gpu::cuda::native_mix {

// Les éléments [first, end) de chaque groupe de la sortie jugé à part (l'image ; le flot et le masque ;
// chaque sortie de la paire de GMFSS).
using Spans = std::vector<std::pair<std::size_t, std::size_t>>;
// Une valeur par morceau et par option : [morceau][option].
using Grid = std::vector<std::vector<double>>;
// L'option de chaque morceau.
using Options = std::vector<int>;
// L'erreur quadratique de chaque groupe, par morceau et par option : [morceau][option][groupe].
using Effects = std::vector<std::vector<std::vector<double>>>;

// Le pic de chaque groupe de want : max(1, |want| le plus grand du groupe) ; l'image va de 0 à 1, un flot en
// pixels peut dépasser 1.
std::vector<double> group_peak(const std::vector<float>& want, const Spans& spans);
// L'erreur quadratique moyenne de got contre want, groupe par groupe ; infinie sur une valeur non finie d'un
// côté ou de l'autre (une référence fautive ne laisse rien passer).
std::vector<double> group_mse(const std::vector<float>& want, const std::vector<float>& got,
                              const Spans& spans);
// Le PSNR d'un groupe : 200 dB sans erreur, moins l'infini sur une erreur non finie.
double db_of(double peak, double mse);
// Le PSNR du groupe le plus bas.
double worst_db(const std::vector<double>& peak, const std::vector<double>& mse);

// Ce que rend la chaîne pour un mélange : l'erreur de chaque groupe contre la référence, puis contre la
// sortie de la base.
struct Eval {
  std::vector<double> want;
  std::vector<double> base;
};

struct Problem {
  Grid times;  // le temps seul de chaque option (une option hors de cands n'est pas lue)
  std::vector<std::vector<int>> cands;  // les options que chaque morceau peut prendre, sa base comprise
  Options base;                         // l'option de chaque morceau pendant la mesure des effets
  std::vector<Options> seeds;           // des mélanges mesurés d'avance tels quels
  std::vector<double> peak;             // le pic de chaque groupe de la référence
  std::vector<double> floor;            // l'erreur de la base contre la référence (nulle : la base l'est)
  double min_db = 0.0;
};

// Un mélange passé sur la chaîne et son erreur contre la référence.
struct Measured {
  Options options;
  std::vector<double> mse;
};

// Ce que doit tenir un mélange : min_db s'il est mesuré, multi_db s'il est prévu ; rho, la corrélation de
// chaque groupe ; les mélanges mesurés.
struct Bar {
  double min_db = 0.0;
  double multi_db = 0.0;
  std::vector<double> rho;
  std::vector<Measured> measured;
};

// Les effets d'un mélange, groupe par groupe : leur somme et le carré de la somme de leurs racines.
struct Spread {
  std::vector<double> power;
  std::vector<double> amplitude;
};
Spread spread_of(const Options& options, const Effects& effects, std::size_t groups);
// L'erreur prévue d'un mélange de plusieurs morceaux hors de la base.
std::vector<double> predicted(const Spread& s, const std::vector<double>& floor,
                              const std::vector<double>& rho);

// Le mélange le plus rapide parmi cands dont l'erreur (mesurée, sinon prévue) tient bar ; à temps égal, le
// plus juste. Rien si aucun ne tient.
struct Pick {
  Options options;
  std::vector<double> mse;
};
std::optional<Pick> fastest_within(const Problem& problem, const Effects& effects, const Bar& bar);
// Après une vérification manquée (measured : l'erreur mesurée) : rho recalé, la barre des mélanges prévus à
// min_db plus slack, le mélange gardé comme mesuré.
void miss(Bar& bar, const Problem& problem, const Effects& effects, Pick pick, std::vector<double> measured,
          double slack);

// Le mélange retenu. held : vérifié au seuil ; sinon options est la base et rien ne tient.
struct Mix {
  Options options;
  double db = 0.0;         // contre la référence, mesurés
  double predicted = 0.0;  // ce que la prévision en attendait
  std::vector<double> base_mse;
  bool held = false;
  Grid effect_db;               // l'effet seul de chaque option, en dB contre la base ; NaN : pas mesuré
  std::vector<double> seed_db;  // chaque graine contre la référence
};

namespace detail {
template <class Chain>
using Error = typename std::invoke_result_t<Chain&, const Options&>::error_type;
}  // namespace detail

// Mesure l'effet de chaque option et les graines (chain : l'Eval d'un mélange), puis prend le mélange le plus
// rapide qui tient le seuil, vérifié sur la chaîne. Le dernier de kAttempts essais ne prend qu'un mélange
// déjà mesuré ; s'il n'y en a pas, rien ne tient.
template <class Chain>
std::expected<Mix, detail::Error<Chain>> choose(const Problem& problem, Chain&& chain) {
  constexpr int kAttempts = 4;
  constexpr double kSlack = 0.5;
  const std::size_t pieces = problem.times.size();
  const std::size_t groups = problem.peak.size();
  Effects effects(pieces);
  Mix mix{.options = problem.base, .effect_db = Grid(pieces)};
  Bar bar{.min_db = problem.min_db, .multi_db = problem.min_db, .rho = std::vector<double>(groups, 0.0)};
  bar.measured.push_back({.options = problem.base, .mse = problem.floor});
  for (std::size_t p = 0; p < pieces; ++p) {
    const auto base = static_cast<std::size_t>(problem.base[p]);
    effects[p].resize(problem.times[p].size());
    mix.effect_db[p].assign(problem.times[p].size(), std::numeric_limits<double>::quiet_NaN());
    effects[p][base].assign(groups, 0.0);
    for (const int o : problem.cands[p]) {
      if (static_cast<std::size_t>(o) == base) continue;
      Options alone = problem.base;
      alone[p] = o;
      auto got = chain(alone);
      if (!got) return std::unexpected(std::move(got.error()));
      effects[p][static_cast<std::size_t>(o)] = got->base;
      mix.effect_db[p][static_cast<std::size_t>(o)] = worst_db(problem.peak, got->base);
      bar.measured.push_back({.options = std::move(alone), .mse = std::move(got->want)});
    }
  }
  for (const Options& seed : problem.seeds) {
    auto got = chain(seed);
    if (!got) return std::unexpected(std::move(got.error()));
    mix.seed_db.push_back(worst_db(problem.peak, got->want));
    bar.measured.push_back({.options = seed, .mse = std::move(got->want)});
  }
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    if (attempt + 1 == kAttempts) bar.multi_db = std::numeric_limits<double>::infinity();
    std::optional<Pick> pick = fastest_within(problem, effects, bar);
    if (!pick) break;
    mix.predicted = worst_db(problem.peak, pick->mse);
    if (pick->options == problem.base) {
      mix.db = mix.predicted;
      mix.base_mse.assign(groups, 0.0);
      mix.held = true;
      return mix;
    }
    auto got = chain(pick->options);
    if (!got) return std::unexpected(std::move(got.error()));
    if (worst_db(problem.peak, got->want) >= problem.min_db) {
      mix.options = std::move(pick->options);
      mix.db = worst_db(problem.peak, got->want);
      mix.base_mse = std::move(got->base);
      mix.held = true;
      return mix;
    }
    miss(bar, problem, effects, std::move(*pick), std::move(got->want), kSlack);
  }
  mix.predicted = 0.0;
  return mix;
}

}  // namespace kirari::gpu::cuda::native_mix
