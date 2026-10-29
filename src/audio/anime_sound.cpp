#include "audio/anime_sound.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <tuple>

namespace ajp::audio {
namespace {

constexpr float kPi = std::numbers::pi_v<float>;
constexpr double kPiD = std::numbers::pi;

// ---- Filtres du livre de recettes « Audio EQ Cookbook » (un état par canal, jusqu'à 2 canaux) ----------
struct Biquad {
  float b0 = 1.0F, b1 = 0.0F, b2 = 0.0F, a1 = 0.0F, a2 = 0.0F;
  std::array<float, 2> z1{}, z2{};
  float run(float x, int ch = 0) {
    const auto c = static_cast<std::size_t>(ch);
    const float y = b0 * x + z1[c];
    z1[c] = b1 * x - a1 * y + z2[c];
    z2[c] = b2 * x - a2 * y;
    return y;
  }
  void clear() {
    z1 = {};
    z2 = {};
  }
};

Biquad make(float b0, float b1, float b2, float a0, float a1, float a2) {
  return Biquad{b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0, {}, {}};
}
Biquad low_pass(float f, float q, int rate) {
  const float w = 2.0F * kPi * f / static_cast<float>(rate);
  const float cw = std::cos(w);
  const float alpha = std::sin(w) / (2.0F * q);
  return make((1 - cw) / 2, 1 - cw, (1 - cw) / 2, 1 + alpha, -2 * cw, 1 - alpha);
}
Biquad high_pass(float f, float q, int rate) {
  const float w = 2.0F * kPi * f / static_cast<float>(rate);
  const float cw = std::cos(w);
  const float alpha = std::sin(w) / (2.0F * q);
  return make((1 + cw) / 2, -(1 + cw), (1 + cw) / 2, 1 + alpha, -2 * cw, 1 - alpha);
}
Biquad band_pass(float f, float q, int rate) {  // gain 0 dB au sommet
  const float w = 2.0F * kPi * f / static_cast<float>(rate);
  const float alpha = std::sin(w) / (2.0F * q);
  return make(alpha, 0.0F, -alpha, 1 + alpha, -2 * std::cos(w), 1 - alpha);
}
Biquad peaking(float f, float gain_db, float q, int rate) {
  const float a = std::pow(10.0F, gain_db / 40.0F);
  const float w = 2.0F * kPi * f / static_cast<float>(rate);
  const float alpha = std::sin(w) / (2.0F * q);
  const float cw = std::cos(w);
  return make(1 + alpha * a, -2 * cw, 1 - alpha * a, 1 + alpha / a, -2 * cw, 1 - alpha / a);
}

// Passe-tout à retard : y[n] = -g x[n] + x[n-d] + g y[n-d] ; module 1 à toutes les fréquences.
struct Allpass {
  std::vector<float> x, y;
  std::size_t at = 0;
  float g = 0.5F;
  Allpass(int delay, float gain) : x(static_cast<std::size_t>(std::max(delay, 1)), 0.0F), y(x), g(gain) {}
  float run(float in) {
    const float out = -g * in + x[at] + g * y[at];
    x[at] = in;
    y[at] = out;
    at = (at + 1) % x.size();
    return out;
  }
  void clear() {
    std::ranges::fill(x, 0.0F);
    std::ranges::fill(y, 0.0F);
  }
};
// Trois passe-tout de retards premiers entre eux : l'ambiance est étalée sur quelques ms sans changer son
// spectre ; deux jeux différents pour deux côtés décorrélés.
struct Decorrelator {
  std::array<Allpass, 3> stages;
  Decorrelator(int rate, bool second)
      : stages{{Allpass(static_cast<int>(std::lround((second ? 1.3 : 1.0) * 0.001 * rate)), 0.55F),
                Allpass(static_cast<int>(std::lround((second ? 2.3 : 1.7) * 0.001 * rate)), 0.5F),
                Allpass(static_cast<int>(std::lround((second ? 3.7 : 2.9) * 0.001 * rate)), 0.45F)}} {}
  float run(float x) {
    for (Allpass& s : stages) x = s.run(x);
    return x;
  }
  void clear() {
    for (Allpass& s : stages) s.clear();
  }
};

float db_to_lin(float db) {
  return std::pow(10.0F, db / 20.0F);
}

// ---- Vrai-crête : interpolation ×4 par FIR polyphasé de 48 coefficients (12 par phase) -------------------
constexpr int kTpPhases = 4;
constexpr int kTpTaps = 12;
const std::array<std::array<float, kTpTaps>, kTpPhases>& tp_coefficients() {
  static const auto table = [] {
    std::array<std::array<float, kTpTaps>, kTpPhases> t{};
    constexpr int kLen = kTpPhases * kTpTaps;
    for (int p = 0; p < kTpPhases; ++p) {
      double sum = 0.0;
      for (int k = 0; k < kTpTaps; ++k) {
        const int n = p + kTpPhases * k;
        const double x = (n - (kLen - 1) / 2.0) / kTpPhases;
        const double sinc = std::fabs(x) < 1e-9 ? 1.0 : std::sin(kPiD * x) / (kPiD * x);
        const double window = 0.5 - 0.5 * std::cos(2.0 * kPiD * (n + 0.5) / kLen);
        t[static_cast<std::size_t>(p)][static_cast<std::size_t>(k)] = static_cast<float>(sinc * window);
        sum += sinc * window;
      }
      for (float& c : t[static_cast<std::size_t>(p)]) c = static_cast<float>(c / sum);  // gain continu 1
    }
    return t;
  }();
  return table;
}

}  // namespace

// ---- Réglages ---------------------------------------------------------------------------------------------
std::array<float, 16> to_wire(const AnimeTuning& t) {
  std::array<float, 16> w{};
  const std::array<float, kAnimeFields> f{
      t.enabled ? 1.0F : 0.0F, t.width,         t.ambience,  t.voice_db,
      t.presence_db,           t.voice_forward, t.bass,      t.loudness ? 1.0F : 0.0F,
      t.target_lufs,           t.compress,      t.ceiling_db};
  std::ranges::copy(f, w.begin());
  return w;
}

AnimeTuning anime_from_wire(const std::array<float, 16>& w) {
  return clamped(AnimeTuning{.enabled = w[0] >= 0.5F,
                             .width = w[1],
                             .ambience = w[2],
                             .voice_db = w[3],
                             .presence_db = w[4],
                             .voice_forward = w[5],
                             .bass = w[6],
                             .loudness = w[7] >= 0.5F,
                             .target_lufs = w[8],
                             .compress = w[9],
                             .ceiling_db = w[10]});
}

AnimeTuning clamped(AnimeTuning t) {
  const auto fin = [](float v, float fallback) { return std::isfinite(v) ? v : fallback; };
  t.width = std::clamp(fin(t.width, 0.5F), 0.0F, 1.0F);
  t.ambience = std::clamp(fin(t.ambience, 0.5F), 0.0F, 1.0F);
  t.voice_db = std::clamp(fin(t.voice_db, 2.0F), 0.0F, 4.0F);
  t.presence_db = std::clamp(fin(t.presence_db, 1.5F), 0.0F, 4.0F);
  t.voice_forward = std::clamp(fin(t.voice_forward, 0.0F), 0.0F, 1.0F);
  t.bass = std::clamp(fin(t.bass, 0.3F), 0.0F, 1.0F);
  t.target_lufs = std::clamp(fin(t.target_lufs, -20.0F), -23.0F, -16.0F);
  t.compress = std::clamp(fin(t.compress, 0.3F), 0.0F, 1.0F);
  t.ceiling_db = std::clamp(fin(t.ceiling_db, -1.0F), -3.0F, -0.5F);
  return t;
}

namespace {
const std::array<AnimeLevel, 3> kLevels{{
    {"natural", "Naturel", "Aucun traitement du module : le son de la piste, comme avant.", AnimeTuning{}},
    {"anime", "Anime",
     "Stéréo élargie avec les voix devant, présence des dialogues, graves renforcés par harmoniques, sonie "
     "ramenée à -20 LUFS, compresseur doux et limiteur.",
     AnimeTuning{.enabled = true}},
    {"strong", "Anime fort",
     "Plus large, voix en avant (+3 dB), graves plus présents, compresseur plus ferme, sonie à -18 LUFS.",
     AnimeTuning{.enabled = true,
                 .width = 0.85F,
                 .ambience = 0.75F,
                 .voice_db = 3.0F,
                 .presence_db = 2.5F,
                 .voice_forward = 0.7F,
                 .bass = 0.6F,
                 .loudness = true,
                 .target_lufs = -18.0F,
                 .compress = 0.55F,
                 .ceiling_db = -1.0F}},
}};
}  // namespace

std::span<const AnimeLevel> anime_levels() {
  return kLevels;
}

const AnimeLevel* find_anime_level(std::string_view key) {
  const auto found = std::ranges::find(kLevels, key, &AnimeLevel::key);
  return found == kLevels.end() ? nullptr : &*found;
}

// ---- Sonie BS.1770-4 ------------------------------------------------------------------------------------
LoudnessMeter::LoudnessMeter(int sample_rate) {
  // Étage 1 : étagère aiguë de la tête (+4 dB vers 1,7 kHz) ; étage 2 : passe-haut RLB (38 Hz). Formules de
  // libebur128, valables à tous les taux.
  {
    const double f0 = 1681.974450955533, g = 3.999843853973347, q = 0.7071752369554196;
    const double k = std::tan(kPiD * f0 / sample_rate);
    const double vh = std::pow(10.0, g / 20.0);
    const double vb = std::pow(vh, 0.4996667741545416);
    const double a0 = 1.0 + k / q + k * k;
    shelf_.b0 = (vh + vb * k / q + k * k) / a0;
    shelf_.b1 = 2.0 * (k * k - vh) / a0;
    shelf_.b2 = (vh - vb * k / q + k * k) / a0;
    shelf_.a1 = 2.0 * (k * k - 1.0) / a0;
    shelf_.a2 = (1.0 - k / q + k * k) / a0;
  }
  {
    const double f0 = 38.13547087602444, q = 0.5003270373238773;
    const double k = std::tan(kPiD * f0 / sample_rate);
    const double a0 = 1.0 + k / q + k * k;
    high_pass_.b0 = 1.0;
    high_pass_.b1 = -2.0;
    high_pass_.b2 = 1.0;
    high_pass_.a1 = 2.0 * (k * k - 1.0) / a0;
    high_pass_.a2 = (1.0 - k / q + k * k) / a0;
  }
  hop_ = std::max(1, sample_rate / 10);
}

bool LoudnessMeter::add(float left, float right) {
  const double l = high_pass_.run(0, shelf_.run(0, left));
  const double r = high_pass_.run(1, shelf_.run(1, right));
  sum_ += l * l + r * r;
  if (++count_ < hop_) return false;
  power_[static_cast<std::size_t>(head_)] = sum_ / hop_;
  head_ = (head_ + 1) % static_cast<int>(power_.size());
  filled_ = std::min<int>(filled_ + 1, static_cast<int>(power_.size()));
  sum_ = 0.0;
  count_ = 0;
  return true;
}

void LoudnessMeter::reset() {
  shelf_.z1 = shelf_.z2 = high_pass_.z1 = high_pass_.z2 = {};
  count_ = 0;
  sum_ = 0.0;
  head_ = 0;
  filled_ = 0;
  power_.fill(0.0);
}

double LoudnessMeter::mean_power(int blocks) const {
  const int n = std::min(blocks, filled_);
  if (n == 0) return 0.0;
  double sum = 0.0;
  const int size = static_cast<int>(power_.size());
  for (int i = 1; i <= n; ++i) sum += power_[static_cast<std::size_t>((head_ - i + size) % size)];
  return sum / n;
}

namespace {
double lufs_of(double power) {
  return power <= 0.0 ? -std::numeric_limits<double>::infinity() : -0.691 + 10.0 * std::log10(power);
}
}  // namespace

double LoudnessMeter::momentary() const {
  return filled_ < 4 ? -std::numeric_limits<double>::infinity() : lufs_of(mean_power(4));
}
double LoudnessMeter::short_term() const {
  return lufs_of(mean_power(30));  // moins de 3 s : la fenêtre disponible
}

double integrated_lufs(std::span<const float> stereo, int sample_rate) {
  // Puissances de chaque bloc de 400 ms au pas de 100 ms (via les sous-blocs de 100 ms du mètre).
  const std::size_t frames = stereo.size() / 2;
  LoudnessMeter again(sample_rate);
  std::vector<double> power;
  for (std::size_t f = 0; f < frames; ++f) {
    if (again.add(stereo[f * 2], stereo[f * 2 + 1]) && again.blocks() >= 4) {
      const double l = again.momentary();
      power.push_back(std::pow(10.0, (l + 0.691) / 10.0));
    }
  }
  if (power.empty()) return -std::numeric_limits<double>::infinity();
  const auto mean_above = [&](double floor_lufs) {
    double sum = 0.0;
    int n = 0;
    for (double p : power) {
      if (lufs_of(p) > floor_lufs) {
        sum += p;
        ++n;
      }
    }
    return std::pair{sum, n};
  };
  auto [sum, n] = mean_above(-70.0);
  if (n == 0) return -std::numeric_limits<double>::infinity();
  const double relative = lufs_of(sum / n) - 10.0;
  std::tie(sum, n) = mean_above(relative);
  return n == 0 ? -std::numeric_limits<double>::infinity() : lufs_of(sum / n);
}

double true_peak_db(std::span<const float> stereo, int /*sample_rate*/) {
  constexpr int kUp = 8;
  constexpr int kHalf = 16;
  double peak = 0.0;
  const std::size_t frames = stereo.size() / 2;
  for (std::size_t ch = 0; ch < 2; ++ch) {
    for (std::size_t f = 0; f < frames; ++f) {
      peak = std::max(peak, static_cast<double>(std::fabs(stereo[f * 2 + ch])));
      for (int p = 1; p < kUp; ++p) {
        const double frac = static_cast<double>(p) / kUp;
        double acc = 0.0;
        for (int k = -kHalf + 1; k <= kHalf; ++k) {
          const auto idx = static_cast<std::ptrdiff_t>(f) + k;
          if (idx < 0 || idx >= static_cast<std::ptrdiff_t>(frames)) continue;
          const double x = k - frac;
          const double w = 0.5 + 0.5 * std::cos(kPiD * x / (kHalf + 0.5));
          acc += stereo[static_cast<std::size_t>(idx) * 2 + ch] * std::sin(kPiD * x) / (kPiD * x) * w;
        }
        peak = std::max(peak, std::fabs(acc));
      }
    }
  }
  return peak <= 0.0 ? -std::numeric_limits<double>::infinity() : 20.0 * std::log10(peak);
}

// ---- Le module --------------------------------------------------------------------------------------------
struct AnimeSound::State {
  State(const AnimeTuning& tuning, int sample_rate, bool phones)
      : t(clamped(tuning)),
        rate(sample_rate),
        headphones(phones),
        meter(sample_rate),
        voice_band(band_pass(1000.0F, 0.6F, sample_rate)),
        side_band(band_pass(1000.0F, 0.6F, sample_rate)),
        presence(peaking(3000.0F, t.presence_db, 0.9F, sample_rate)),
        decor_a(sample_rate, false),
        decor_b(sample_rate, true),
        bass_low(low_pass(110.0F, 0.7071F, sample_rate)),
        bass_env_low(low_pass(30.0F, 0.7071F, sample_rate)),
        bass_high_a(high_pass(170.0F, 0.7071F, sample_rate)),
        bass_high_b(high_pass(170.0F, 0.7071F, sample_rate)),
        bass_soften(low_pass(900.0F, 0.7071F, sample_rate)) {
    const float cutoff = 1200.0F;
    cross_alpha = 1.0F - std::exp(-2.0F * kPi * cutoff / static_cast<float>(rate));
    cross_delay = std::clamp(static_cast<int>(std::lround(0.00022 * rate)), 1, 63);
    cross_line.assign(64, 0.0F);
    gain_voice = db_to_lin(t.voice_db);
    ceiling = db_to_lin(t.ceiling_db);
    // Anticipation du limiteur : 1,3 ms (au moins 16 trames), moitié pour la fenêtre de lissage.
    delay = std::max(16, static_cast<int>(std::lround(0.0013 * rate)));
    window = delay / 2;
    history.assign(static_cast<std::size_t>(delay), {0.0F, 0.0F});
    req_ring.assign(static_cast<std::size_t>(window), 1.0F);
    min_ring.assign(static_cast<std::size_t>(window), 1.0F);
    release_k = 1.0F - std::exp(-1.0F / (0.06F * static_cast<float>(rate)));
    comp_attack = std::exp(-1.0F / (0.010F * static_cast<float>(rate)));
    comp_release = std::exp(-1.0F / (0.200F * static_cast<float>(rate)));
    gain_smooth_k = 1.0F - std::exp(-1.0F / (0.05F * static_cast<float>(rate)));
  }

  AnimeTuning t;
  int rate;
  bool headphones;
  LoudnessMeter meter;
  // Étape 1
  Biquad voice_band, side_band, presence;
  Decorrelator decor_a, decor_b;
  float cross_alpha = 0.0F;
  int cross_delay = 1;
  std::vector<float> cross_line;
  std::size_t cross_at = 0;
  float cross_low = 0.0F;
  float gain_voice = 1.0F;
  // Étape 2
  Biquad bass_low, bass_env_low, bass_high_a, bass_high_b, bass_soften;
  float bass_env = 0.0F;
  float gain_db = 0.0F;      // gain de sonie visé, lissé par bloc de 100 ms
  float gain_lin = 1.0F;     // gain appliqué, lissé par trame
  float gain_target = 1.0F;  // vers lequel gain_lin glisse
  int since_reset_blocks = 0;
  float comp_env_db = -90.0F;
  float comp_attack = 0.0F, comp_release = 0.0F, gain_smooth_k = 0.0F;
  // Limiteur vrai-crête
  float ceiling = 0.89F;
  int delay = 64;
  int window = 32;
  std::vector<std::array<float, 2>> history;  // anticipation : trames retardées
  std::size_t history_at = 0;
  std::array<std::array<float, kTpTaps>, 2>
      tp_hist{};  // 12 dernières entrées, par canal (la plus récente à 0)
  std::vector<float> req_ring, min_ring;
  std::size_t ring_at = 0;
  float limiter_gain = 1.0F;
  float release_k = 0.0F;

  void update_loudness() {
    const double momentary = meter.momentary();
    const double short_term = meter.short_term();
    ++since_reset_blocks;
    if (!std::isfinite(short_term) || short_term < -50.0) return;  // silence : le gain est tenu
    float want = static_cast<float>(t.target_lufs - short_term);
    if (std::isfinite(momentary))  // un passage fort soudain baisse le gain sans attendre les 3 s
      want = std::min(want, static_cast<float>(t.target_lufs + 4.0 - momentary));
    want = std::clamp(want, -9.0F, 6.0F);
    // Constante de temps par bloc de 100 ms : vive pendant les 2 premières secondes, ensuite lente.
    const float tau = since_reset_blocks < 20 ? 0.5F : (want > gain_db ? 5.0F : 2.5F);
    gain_db += (want - gain_db) * (1.0F - std::exp(-0.1F / tau));
    gain_target = db_to_lin(gain_db);
  }
};

AnimeSound::AnimeSound(const AnimeTuning& tuning, int sample_rate, bool headphones)
    : state_(std::make_unique<State>(tuning, sample_rate, headphones)) {}
AnimeSound::~AnimeSound() = default;

float AnimeSound::loudness_gain_db() const {
  return state_->gain_db;
}
int AnimeSound::latency_frames() const {
  return state_->delay;
}

void AnimeSound::reset() {
  State& s = *state_;
  s.voice_band.clear();
  s.side_band.clear();
  s.presence.clear();
  s.decor_a.clear();
  s.decor_b.clear();
  std::ranges::fill(s.cross_line, 0.0F);
  s.cross_low = 0.0F;
  s.bass_low.clear();
  s.bass_env_low.clear();
  s.bass_high_a.clear();
  s.bass_high_b.clear();
  s.bass_soften.clear();
  s.bass_env = 0.0F;
  s.meter.reset();
  s.gain_db = 0.0F;
  s.gain_lin = s.gain_target = 1.0F;
  s.since_reset_blocks = 0;
  s.comp_env_db = -90.0F;
  std::ranges::fill(s.history, std::array<float, 2>{0.0F, 0.0F});
  s.tp_hist = {};
  std::ranges::fill(s.req_ring, 1.0F);
  std::ranges::fill(s.min_ring, 1.0F);
  s.limiter_gain = 1.0F;
}

PcmBlock AnimeSound::stage(PcmBlock in) {
  State& s = *state_;
  const auto frames = static_cast<std::size_t>(in.frames());
  const bool six = s.headphones && s.t.width > 0.0F;
  PcmBlock out{.pts_us = in.pts_us, .channels = six ? 6 : 2};
  out.samples.assign(frames * static_cast<std::size_t>(out.channels), 0.0F);
  const float width = s.t.width;
  const float forward = s.t.voice_forward;
  const float cross = 0.5F * width;              // diaphonie annulée sur les côtés (enceintes)
  const float side_gain = 1.0F + 0.45F * width;  // côtés un peu plus présents (enceintes)
  const float ambience = s.t.ambience * (six ? 0.8F : 0.35F);
  // Casque : part du milieu confiée à l'enceinte centrale (0°), qui n'a ni retard interaural ni ombre de tête
  // : un milieu porté par les deux avants à ±30° se creuse vers 2 kHz (peigne). Les côtés restent sur les
  // avants.
  const float share = 0.8F + 0.12F * forward;
  constexpr float kHeadMakeup = 1.1F;  // les trajets du binaural valent 0,91 pour un centre
  for (std::size_t f = 0; f < frames; ++f) {
    const float l = std::isfinite(in.samples[f * 2]) ? in.samples[f * 2] : 0.0F;
    const float r = std::isfinite(in.samples[f * 2 + 1]) ? in.samples[f * 2 + 1] : 0.0F;
    const float m = 0.5F * (l + r);
    float side = 0.5F * (l - r);
    // Voix : la bande vocale du milieu portée de voice_db, puis la présence ; le milieu garde tout le reste.
    const float band = s.voice_band.run(m);
    const float boosted = s.presence.run(m + (s.gain_voice - 1.0F) * band);
    // « Voix en avant » : les côtés creusés dans la bande vocale (le milieu n'en est pas touché).
    side -= 0.35F * forward * s.side_band.run(side);
    const float decor_in = side;
    if (!six) {
      // Transaural réduit aux côtés : S' = S + c·passe-bas(retard(S')), le milieu intact (la somme G+D
      // reste).
      const float delayed = s.cross_line[(s.cross_at + 64 - static_cast<std::size_t>(s.cross_delay)) % 64];
      s.cross_low += s.cross_alpha * (delayed - s.cross_low);
      const float widened = side + cross * s.cross_low;
      s.cross_line[s.cross_at] = widened;
      s.cross_at = (s.cross_at + 1) % 64;
      const float diffuse = ambience * s.decor_a.run(decor_in);
      const float sd = side_gain * widened + diffuse;
      out.samples[f * 2] = boosted + sd;
      out.samples[f * 2 + 1] = boosted - sd;
    } else {
      const float front = (1.0F - share) * boosted;
      out.samples[f * 6 + 0] = kHeadMakeup * (front + side);
      out.samples[f * 6 + 1] = kHeadMakeup * (front - side);
      out.samples[f * 6 + 2] = kHeadMakeup * 1.82F * share * boosted;
      out.samples[f * 6 + 4] = ambience * s.decor_a.run(decor_in);
      out.samples[f * 6 + 5] = -ambience * s.decor_b.run(decor_in);
    }
  }
  return out;
}

void AnimeSound::finish(PcmBlock& stereo) {
  State& s = *state_;
  const auto frames = static_cast<std::size_t>(stereo.frames());
  const auto& table = tp_coefficients();
  const bool bass_on = s.t.bass > 0.0F;
  const float compress = s.t.compress;
  const float threshold_db = -16.0F - 8.0F * compress;
  const float ratio = 1.0F + 2.5F * compress;
  constexpr float knee_db = 6.0F;
  for (std::size_t f = 0; f < frames; ++f) {
    float l = stereo.samples[f * 2];
    float r = stereo.samples[f * 2 + 1];
    if (!std::isfinite(l)) l = 0.0F;
    if (!std::isfinite(r)) r = 0.0F;
    // Graves : harmoniques de la bande sous 110 Hz, mises au niveau de la bande (le signal normalisé est
    // saturé doucement : 3f, 5f), fondamental retiré, ajoutées en mono aux deux canaux.
    if (bass_on) {
      const float low = s.bass_low.run(0.5F * (l + r));
      s.bass_env += 0.002F * (std::fabs(s.bass_env_low.run(std::fabs(low))) - s.bass_env);
      const float env = std::max(s.bass_env * 1.5F, 1e-4F);
      const float shaped = std::tanh(2.5F * low / env) * env;
      const float harmonics = s.bass_soften.run(s.bass_high_b.run(s.bass_high_a.run(shaped)));
      const float add = 0.6F * s.t.bass * harmonics;
      l += add;
      r += add;
    }
    // Sonie : la mesure se fait avant le gain, qui glisse vers sa cible.
    if (s.t.loudness) {
      if (s.meter.add(l, r)) s.update_loudness();
      s.gain_lin += s.gain_smooth_k * (s.gain_target - s.gain_lin);
      l *= s.gain_lin;
      r *= s.gain_lin;
    }
    // Compresseur doux, lié, courbe à genou doux, sans gain de rattrapage (la sonie s'en charge).
    if (compress > 0.0F) {
      const float level_db = 20.0F * std::log10(std::max(std::max(std::fabs(l), std::fabs(r)), 1e-5F));
      const float k = level_db > s.comp_env_db ? s.comp_attack : s.comp_release;
      s.comp_env_db = k * s.comp_env_db + (1.0F - k) * level_db;
      const float over = s.comp_env_db - threshold_db;
      float reduction_db = 0.0F;
      if (2.0F * over > knee_db) {
        reduction_db = over * (1.0F - 1.0F / ratio);
      } else if (2.0F * over > -knee_db) {
        const float x = over + knee_db / 2.0F;
        reduction_db = (1.0F - 1.0F / ratio) * x * x / (2.0F * knee_db);
      }
      const float g = db_to_lin(-reduction_db);
      l *= g;
      r *= g;
    }
    // Limiteur vrai-crête : crête des 4 phases du FIR ×4, gain requis, minimum glissant puis moyenne
    // glissante sur la fenêtre (le gain lissé ne dépasse jamais le requis), relâchement de 60 ms.
    float peak = std::max(std::fabs(l), std::fabs(r));
    const std::array<float, 2> in{l, r};
    for (std::size_t c = 0; c < 2; ++c) {
      auto& h = s.tp_hist[c];
      for (std::size_t i = kTpTaps - 1; i > 0; --i) h[i] = h[i - 1];
      h[0] = in[c];
      for (const auto& phase : table) {
        float acc = 0.0F;
        for (std::size_t k = 0; k < kTpTaps; ++k) acc += phase[k] * h[k];
        peak = std::max(peak, std::fabs(acc));
      }
    }
    const float required = peak > s.ceiling ? s.ceiling / peak : 1.0F;
    s.req_ring[s.ring_at] = required;
    float min_gain = 1.0F;
    for (float v : s.req_ring) min_gain = std::min(min_gain, v);
    s.min_ring[s.ring_at] = min_gain;
    s.ring_at = (s.ring_at + 1) % s.req_ring.size();
    float smoothed = 0.0F;
    for (float v : s.min_ring) smoothed += v;
    smoothed /= static_cast<float>(s.min_ring.size());
    s.limiter_gain =
        smoothed < s.limiter_gain ? smoothed : s.limiter_gain + (smoothed - s.limiter_gain) * s.release_k;
    // Trame retardée de l'anticipation.
    std::array<float, 2>& slot = s.history[s.history_at];
    const std::array<float, 2> delayed = slot;
    slot = in;
    s.history_at = (s.history_at + 1) % s.history.size();
    stereo.samples[f * 2] = std::clamp(delayed[0] * s.limiter_gain, -1.0F, 1.0F);
    stereo.samples[f * 2 + 1] = std::clamp(delayed[1] * s.limiter_gain, -1.0F, 1.0F);
  }
}

}  // namespace ajp::audio
