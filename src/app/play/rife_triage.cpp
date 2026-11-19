#include "app/play/rife_triage.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "core/log.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/motion_camera.h"
#include "kernels/spec/motion_doubt.h"

namespace kirari::app::play {

namespace {

std::size_t cell(int col, int row, int cols) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col);
}

// Le plus fort, puis la moyenne, des voisins 3 × 3 de chaque case (bords tronqués).
std::vector<float> neighbours(std::span<const float> in, int cols, int rows, bool average) {
  std::vector<float> out(in.size(), 0.0F);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float best = 0.0F;
      float sum = 0.0F;
      int seen = 0;
      for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
          const int rr = r + dr;
          const int cc = c + dc;
          if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue;
          const float v = in[cell(cc, rr, cols)];
          best = std::max(best, v);
          sum += v;
          ++seen;
        }
      }
      out[cell(c, r, cols)] = average ? sum / static_cast<float>(seen) : best;
    }
  }
  return out;
}

// Une seule case pleine : toute la paire au fondu.
DoubtGrid full_fade() {
  return {.cols = 1, .rows = 1, .weights = {kernels::kDoubtFull}, .share = 1.0};
}

}  // namespace

bool flow_held(const kernels::SceneMotion& scene) {
  if (!scene.drawing_held || !scene.background_held || scene.fade || scene.cut_likely || scene.tiles.empty())
    return false;
  const auto changed =
      std::ranges::count_if(scene.tiles, [](const kernels::SceneTile& t) { return t.changed; });
  return static_cast<double>(changed) <= triage::kHeldChanged * static_cast<double>(scene.tiles.size());
}

bool flow_lost(const kernels::SceneMotion& scene) {
  return scene.cut_likely || (!scene.camera_fit && scene.unmatched >= triage::kImpossibleUnmatched);
}

bool flow_pan(const kernels::SceneMotion& scene) {
  if (!scene.camera_fit || !scene.drawing_held || scene.background_held || scene.fade || scene.cut_likely)
    return false;
  if (scene.residual > triage::kPanResidual || scene.unmatched > triage::kPanUnmatched) return false;
  // Pas une seule tuile de dessin (la carte n'en classe qu'une tuile sûre, assez texturée et qui correspond,
  // qui bouge hors du modèle de la caméra) ni d'effet. L'erreur coûte inégalement : un petit personnage qui
  // marche pendant le panoramique, décalé avec la caméra, saccade aussitôt ; un vrai panoramique laissé à la
  // fusion ne coûte que du calcul.
  return std::ranges::none_of(scene.tiles, [](const kernels::SceneTile& t) {
    return t.kind == kernels::SceneClass::Drawing || t.kind == kernels::SceneClass::Effect;
  });
}

std::vector<float> tile_doubt(const kernels::SceneMotion& scene) {
  // Un effet sûr qui change : vu deux paires de suite par la carte (à la première, elle garde la classe
  // d'avant ; à la fin, la confiance de l'effet tenu tombe sous kEffectSure).
  const auto sure = [&](int col, int row) {
    if (col < 0 || col >= scene.cols || row < 0 || row >= scene.rows) return false;
    const kernels::SceneTile& t = scene.at(col, row);
    return t.changed && t.kind == kernels::SceneClass::Effect && t.confidence >= triage::kEffectSure;
  };
  std::vector<float> out(scene.tiles.size(), 0.0F);
  if (scene.tiles.size() != cell(0, scene.rows, scene.cols)) return out;
  for (int r = 0; r < scene.rows; ++r) {
    for (int c = 0; c < scene.cols; ++c) {
      if (!sure(c, r)) continue;
      // Et pas seul : une tuile isolée est une étincelle, un bout de trait trop rapide pour NVOF, dont un
      // fantôme de RIFE ne se verrait guère.
      bool joined = false;
      for (int dr = -1; dr <= 1 && !joined; ++dr) {
        for (int dc = -1; dc <= 1 && !joined; ++dc) joined = (dr != 0 || dc != 0) && sure(c + dc, r + dr);
      }
      if (joined) out[cell(c, r, scene.cols)] = 1.0F;
    }
  }
  return out;
}

DoubtGrid doubt_grid(std::span<const float> doubt, int cols, int rows) {
  DoubtGrid grid;
  if (cols < 1 || rows < 1 || doubt.size() != cell(0, rows, cols)) return grid;
  // La part de l'image qui doute : celle des tuiles mêmes. Dilatée puis adoucie, une tuile sur trois qui
  // doute en couvrait presque toute l'image.
  double sum = 0.0;
  for (const float v : doubt) sum += std::clamp(v, 0.0F, 1.0F);
  grid.share = sum / static_cast<double>(doubt.size());
  // Dilatée d'une tuile (le fantôme déborde de la zone sans correspondance), puis adoucie : le bord du fondu
  // ne se voit pas.
  const std::vector<float> soft = neighbours(neighbours(doubt, cols, rows, false), cols, rows, true);
  // Regroupées par le plus fort doute, au plus kDoubtMaxCells de côté.
  const int step_c = (cols + kernels::kDoubtMaxCells - 1) / kernels::kDoubtMaxCells;
  const int step_r = (rows + kernels::kDoubtMaxCells - 1) / kernels::kDoubtMaxCells;
  grid.cols = (cols + step_c - 1) / step_c;
  grid.rows = (rows + step_r - 1) / step_r;
  grid.weights.assign(cell(0, grid.rows, grid.cols), 0);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float v = soft[cell(c, r, cols)];
      if (v < triage::kDoubtFloor) v = 0.0F;
      const auto w = static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0F, 1.0F) * kernels::kDoubtFull));
      std::uint8_t& out = grid.weights[cell(c / step_c, r / step_r, grid.cols)];
      out = std::max(out, w);
    }
  }
  return grid;
}

void RifeTriage::reset() {
  now_ = PairSort::Today;
  grid_ = {};
  kept_.clear();
  kept_cols_ = 0;
  kept_rows_ = 0;
  unrun_ = 0;
}

PairSort RifeTriage::sort_unrun(const PairFacts& facts, const kernels::SceneMotion& scene) {
  // Une autre scène, ou une tenue plus longue qu'un dessin en quatre : le doute d'avant ne vaut plus.
  if ((facts.scores && pair_cut(facts)) || ++unrun_ > triage::kKeepAcross) reset();
  // Tenue par le flux seul : la mesure des blocs aurait laissé tourner RIFE.
  const bool by_flow = facts.scores && facts.same_epoch && facts.factor > 1 && !pair_cut(facts) &&
                       !kernels::is_held(facts.scores->worst_block) && flow_held(scene);
  if (by_flow) ++counts_.held;
  return by_flow ? PairSort::Held : PairSort::Today;
}

void RifeTriage::keep_doubt(std::vector<float>& doubt, const kernels::SceneMotion& scene) {
  // Le doute propre de la paire passe à la suivante, pas celui qu'elle a reçu : gardé de paire en paire, un
  // effet fondait encore quatre paires après sa fin.
  std::vector<float> own = doubt;
  if (kept_cols_ == scene.cols && kept_rows_ == scene.rows && kept_.size() == doubt.size()) {
    for (std::size_t i = 0; i < doubt.size(); ++i)
      doubt[i] = std::max(doubt[i], triage::kDoubtKeep * kept_[i]);
  }
  kept_ = std::move(own);
  kept_cols_ = scene.cols;
  kept_rows_ = scene.rows;
  unrun_ = 0;
}

PairSort RifeTriage::sort(const PairFacts& facts, bool runs) {
  now_ = PairSort::Today;
  grid_ = {};
  asked_ = asked_ || facts.flow_triage;
  if (!facts.flow_triage || !facts.scene) return now_;
  const kernels::SceneMotion& scene = *facts.scene;
  if (!runs) return sort_unrun(facts, scene);
  ++counts_.read;
  std::vector<float> doubt = tile_doubt(scene);
  keep_doubt(doubt, scene);
  if (gmfss_only_) return sort_gmfss(scene, doubt);
  if (refused_) return now_;
  // Coupe probable que la mesure de RIFE n'a pas vue, ou presque rien qui corresponde, sans même un décor que
  // la caméra explique : fondu entier.
  if (flow_lost(scene)) {
    grid_ = full_fade();
    ++counts_.lost;
  } else {
    grid_ = doubt_grid(doubt, scene.cols, scene.rows);
    if (grid_.share >= triage::kFadeShare) grid_ = full_fade();
  }
  if (std::ranges::none_of(grid_.weights, [](std::uint8_t w) { return w > 0; })) {
    grid_ = {};
    return now_;
  }
  now_ = grid_.share >= 1.0 ? PairSort::Fade : PairSort::Masked;
  ++(now_ == PairSort::Fade ? counts_.faded : counts_.masked);
  if (now_ == PairSort::Masked) counts_.masked_share += grid_.share;
  return now_;
}

PairSort RifeTriage::sort_gmfss(const kernels::SceneMotion& scene, const std::vector<float>& doubt) {
  // La paire impossible : la même règle que pour RIFE, mais toute la paire au fondu.
  const bool lost = flow_lost(scene);
  if (lost || doubt_grid(doubt, scene.cols, scene.rows).share >= triage::kFadeShare) {
    camera_ = kernels::kCameraStill;
    now_ = PairSort::Fade;
    ++counts_.faded;
    counts_.lost += lost ? 1 : 0;
  } else if (flow_pan(scene)) {
    for (std::size_t i = 0; i < camera_.size(); ++i) camera_[i] = static_cast<float>(scene.camera[i]);
    now_ = PairSort::Pan;
    ++counts_.panned;
  }
  return now_;
}

bool RifeTriage::settle(const PairFacts& facts, bool on_grid, bool runs, bool smooth_first) {
  sort(facts, runs && !smooth_first);
  return on_grid && !runs && facts.factor > 1 && facts.scores && (pair_cut(facts) || pair_held(facts));
}

void RifeTriage::log() const {
  if (!asked_) return;
  // i18n-ok : la fin de la ligne de journal plus bas.
  const char* why = counts_.refused > 0 ? " (fondu refusé par le backend : RIFE tel quel)"  // i18n-ok
                    : counts_.read == 0 && counts_.held == 0
                        ? " (aucune carte : --scene-map, NVOF, RIFE mouvement ou GMFSS ?)"  // i18n-ok
                        : "";
  // La part moyenne de l'image fondue d'une paire fondue par endroits, en pour cent.
  const double area =
      counts_.masked > 0 ? 100.0 * counts_.masked_share / static_cast<double>(counts_.masked) : 0.0;
  log_channel("gpu")->info(
      "tri des paires par le flux : {} paires lues par la carte, {} dessins tenus copiés, {} paires "  // i18n-ok
      "impossibles fondues (dont {} sans correspondance ni caméra), {} paires fondues par endroits "  // i18n-ok
      "(sur {:.1f} % des tuiles en moyenne), {} panoramiques purs{}",
      counts_.read, counts_.held, counts_.faded, counts_.lost, counts_.masked, area, counts_.panned, why);
}

gpu::Motion RifeTriage::doubted(gpu::Backend& backend, const gpu::Motion& motion, float fade) {
  if (now_ == PairSort::Today || now_ == PairSort::Held || now_ == PairSort::Pan || refused_ ||
      grid_.weights.empty())
    return motion;
  gpu::KernelOps* ops = backend.kernels();
  const auto refuse_once = [&](const Refusal& why) {
    refused_ = true;
    ++counts_.refused;
    log_channel("gpu")->warn("tri des paires : fondu du mouvement refusé ({}), RIFE tel quel", why.detail);
    return motion;
  };
  if (ops == nullptr) return refuse_once(Refusal{RefusalCode::FormatUnsupported, "sans noyaux"});
  const auto fits = [](const std::unique_ptr<gpu::TensorBuffer>& buffer, const gpu::Tensor& like) {
    return buffer != nullptr && buffer->tensor().type == like.type && buffer->tensor().shape == like.shape;
  };
  // Le jeu de la forme du mouvement ; sinon l'autre que le dernier servi, réalloué.
  const auto suits = [&](const Buffers& set) {
    return fits(set.flow, motion.flow) && fits(set.mask, motion.mask);
  };
  const std::size_t slot = suits(buffers_[0]) ? 0 : suits(buffers_[1]) ? 1 : 1 - last_;
  Buffers& set = buffers_[slot];
  for (auto [buffer, like] : {std::pair{&set.flow, &motion.flow}, std::pair{&set.mask, &motion.mask}}) {
    if (fits(*buffer, *like)) continue;
    auto made = backend.alloc_tensor(like->type, like->shape);
    if (!made) return refuse_once(made.error());
    *buffer = std::move(*made);
  }
  last_ = slot;
  gpu::Motion out = motion;
  out.flow = set.flow->tensor();
  out.mask = set.mask->tensor();
  if (auto ok =
          ops->motion_doubt(motion, grid_.weights, grid_.cols, grid_.rows, std::clamp(fade, 0.0F, 1.0F), out);
      !ok)
    return refuse_once(ok.error());
  return out;
}

}  // namespace kirari::app::play
