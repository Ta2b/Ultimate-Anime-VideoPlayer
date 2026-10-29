// « Son anime » (audio/anime_sound.h, docs/audio-anime.md) : mesures chiffrées sur des signaux de synthèse.
// Le test caché [.anime_echantillons] rend les WAV avant/après et le tableau de mesures du rapport.
#include "audio/anime_sound.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "audio/mixer.h"

using namespace ajp::audio;
using Catch::Approx;

namespace {

constexpr int kRate = 48'000;
constexpr double kTwoPi = 2.0 * std::numbers::pi;

using Stereo = std::vector<float>;  // entrelacé G D

// Bruit blanc déterministe dans [-1, 1].
struct Noise {
  std::uint32_t state = 12345U;
  float next() {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(static_cast<std::int32_t>(state >> 8)) / 8388608.0F - 1.0F;
  }
};

Stereo sine_stereo(double hz, double amp, double seconds, double right_scale = 1.0) {
  const auto frames = static_cast<std::size_t>(seconds * kRate);
  Stereo out(frames * 2);
  for (std::size_t f = 0; f < frames; ++f) {
    const auto v = static_cast<float>(amp * std::sin(kTwoPi * hz * static_cast<double>(f) / kRate));
    out[f * 2] = v;
    out[f * 2 + 1] = static_cast<float>(v * right_scale);
  }
  return out;
}

// Une voix de synthèse : hauteur 170-250 Hz, harmoniques sous deux formants, syllabes à 3,5 Hz.
std::vector<float> synth_voice(double seconds, double amp) {
  const auto frames = static_cast<std::size_t>(seconds * kRate);
  std::vector<float> v(frames);
  double phase = 0.0;
  for (std::size_t f = 0; f < frames; ++f) {
    const double t = static_cast<double>(f) / kRate;
    const double pitch = 210.0 + 40.0 * std::sin(kTwoPi * 0.4 * t);
    phase += kTwoPi * pitch / kRate;
    double s = 0.0;
    for (int k = 1; k <= 14; ++k) {
      const double fk = pitch * k;
      const double formant = std::exp(-std::pow((fk - 700.0) / 350.0, 2)) +
                             0.7 * std::exp(-std::pow((fk - 1800.0) / 600.0, 2)) +
                             0.25 * std::exp(-std::pow((fk - 3200.0) / 900.0, 2));
      s += formant / k * std::sin(phase * k);
    }
    const double syllable = std::max(0.0, std::sin(kTwoPi * 3.5 * t + 0.5));
    v[f] = static_cast<float>(amp * s * std::pow(syllable, 0.6));
  }
  return v;
}

// Une musique de synthèse : accords arpégés, basse à 55 Hz, coups de grosse caisse, nappe large (bruit
// passe-bas indépendant à gauche et à droite).
Stereo synth_music(double seconds, double amp) {
  const auto frames = static_cast<std::size_t>(seconds * kRate);
  Stereo out(frames * 2);
  Noise nl{777U}, nr{4242U};
  float lowl = 0.0F, lowr = 0.0F;
  constexpr std::array<double, 4> kChord{261.6, 329.6, 392.0, 523.3};
  for (std::size_t f = 0; f < frames; ++f) {
    const double t = static_cast<double>(f) / kRate;
    lowl += 0.05F * (nl.next() - lowl);
    lowr += 0.05F * (nr.next() - lowr);
    double l = 0.0, r = 0.0;
    for (std::size_t i = 0; i < kChord.size(); ++i) {
      const double gate = 0.5 + 0.5 * std::sin(kTwoPi * (0.5 + 0.25 * static_cast<double>(i)) * t);
      const double s = gate * std::sin(kTwoPi * kChord[i] * t) / 4.0;
      const double pan = -0.7 + 0.47 * static_cast<double>(i);  // étalés de gauche à droite
      l += s * (0.5 - 0.5 * pan);
      r += s * (0.5 + 0.5 * pan);
    }
    const double bass = 0.5 * (std::sin(kTwoPi * 55.0 * t) + 0.3 * std::sin(kTwoPi * 110.0 * t));
    const double beat = std::fmod(t, 0.5);
    const double kick =
        std::exp(-beat * 18.0) * std::sin(kTwoPi * (50.0 + 90.0 * std::exp(-beat * 30.0)) * beat);
    const double pad = 3.0;
    out[f * 2] = static_cast<float>(amp * (l + 0.5 * bass + 0.5 * kick + pad * lowl));
    out[f * 2 + 1] = static_cast<float>(amp * (r + 0.5 * bass + 0.5 * kick + pad * lowr));
  }
  return out;
}

// Voix au centre (G = D) + musique, comme un mixage d'anime.
Stereo anime_mix(double seconds, double voice_amp = 0.30, double music_amp = 0.18) {
  Stereo music = synth_music(seconds, music_amp);
  const std::vector<float> voice = synth_voice(seconds, voice_amp);
  for (std::size_t f = 0; f < voice.size(); ++f) {
    music[f * 2] += voice[f];
    music[f * 2 + 1] += voice[f];
  }
  return music;
}

// Passe tout le signal dans un mixeur, par blocs de block_frames trames.
Stereo render(MixConfig config, const Stereo& in, std::size_t block_frames = 480, int source_channels = 2) {
  Mixer mixer(std::move(config));
  mixer.note_source(source_channels);
  Stereo out;
  out.reserve(in.size());
  const std::size_t frames = in.size() / 2;
  for (std::size_t at = 0; at < frames; at += block_frames) {
    const std::size_t n = std::min(block_frames, frames - at);
    PcmBlock block{.pts_us = 0, .channels = 2};
    block.samples.assign(in.begin() + static_cast<std::ptrdiff_t>(at * 2),
                         in.begin() + static_cast<std::ptrdiff_t>((at + n) * 2));
    PcmBlock result = mixer.process(std::move(block));
    out.insert(out.end(), result.samples.begin(), result.samples.end());
  }
  return out;
}

MixConfig speakers(AnimeTuning t) {
  return MixConfig{.listen = Listen::Speakers, .anime = t, .sample_rate = kRate};
}
MixConfig phones(AnimeTuning t) {
  return MixConfig{.listen = Listen::Headphones, .anime = t, .sample_rate = kRate};
}

// Le niveau désiré tout seul : aucune dynamique, pour isoler l'effet mesuré.
AnimeTuning dry(AnimeTuning t = AnimeTuning{.enabled = true}) {
  t.loudness = false;
  t.compress = 0.0F;
  t.bass = 0.0F;
  return t;
}

double rms_db(const Stereo& s, std::size_t from_frame = 0) {
  double sum = 0.0;
  std::size_t n = 0;
  for (std::size_t i = from_frame * 2; i < s.size(); ++i, ++n) sum += static_cast<double>(s[i]) * s[i];
  return 10.0 * std::log10(std::max(sum / static_cast<double>(std::max<std::size_t>(n, 1)), 1e-20));
}

double peak_abs(const Stereo& s) {
  double p = 0.0;
  for (float v : s) p = std::max(p, static_cast<double>(std::fabs(v)));
  return p;
}

// Corrélation de Pearson entre G et D.
double correlation(const Stereo& s, std::size_t from_frame = 0) {
  double sl = 0, sr = 0, slr = 0;
  for (std::size_t f = from_frame; f < s.size() / 2; ++f) {
    const double l = s[f * 2], r = s[f * 2 + 1];
    sl += l * l;
    sr += r * r;
    slr += l * r;
  }
  return slr / std::sqrt(std::max(sl * sr, 1e-20));
}

// Largeur : énergie des côtés sur celle du milieu, en dB.
double width_db(const Stereo& s, std::size_t from_frame = 0) {
  double m = 0, d = 0;
  for (std::size_t f = from_frame; f < s.size() / 2; ++f) {
    const double a = 0.5 * (s[f * 2] + s[f * 2 + 1]);
    const double b = 0.5 * (s[f * 2] - s[f * 2 + 1]);
    m += a * a;
    d += b * b;
  }
  return 10.0 * std::log10(std::max(d, 1e-20) / std::max(m, 1e-20));
}

Stereo mono_sum(const Stereo& s) {
  Stereo out(s.size());
  for (std::size_t f = 0; f < s.size() / 2; ++f)
    out[f * 2] = out[f * 2 + 1] = 0.5F * (s[f * 2] + s[f * 2 + 1]);
  return out;
}

// Amplitude d'une raie (DFT en un point) sur la voie ch, entre les trames from et to.
double tone_amplitude(const Stereo& s, double hz, std::size_t ch, std::size_t from, std::size_t to) {
  std::complex<double> acc{};
  for (std::size_t f = from; f < to; ++f) {
    const double ph = kTwoPi * hz * static_cast<double>(f) / kRate;
    acc += static_cast<double>(s[f * 2 + ch]) * std::complex<double>(std::cos(ph), -std::sin(ph));
  }
  return 2.0 * std::abs(acc) / static_cast<double>(to - from);
}

// Gain (dB) d'un sinus centré de fréquence hz à travers le mixeur.
double centre_gain_db(const MixConfig& config, double hz, double amp = 0.1) {
  const Stereo in = sine_stereo(hz, amp, 2.0);
  const Stereo out = render(config, in);
  // Une fréquence entière de périodes dans la fenêtre : on la veut sur les 12000 dernières trames.
  const std::size_t from = 60'000;
  const std::size_t to = 96'000;
  return 20.0 * std::log10(tone_amplitude(out, hz, 0, from, to) / tone_amplitude(in, hz, 0, from, to));
}

std::filesystem::path out_dir() {
  const std::filesystem::path dir = "/tmp/claude-1000/son-anime";
  std::filesystem::create_directories(dir);
  return dir;
}

void write_wav(const std::filesystem::path& path, const Stereo& s) {
  std::ofstream f(path, std::ios::binary);
  const auto u32 = [&](std::uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  const auto u16 = [&](std::uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
  const auto data = static_cast<std::uint32_t>(s.size() * 2);
  f.write("RIFF", 4);
  u32(36 + data);
  f.write("WAVEfmt ", 8);
  u32(16);
  u16(1);
  u16(2);
  u32(kRate);
  u32(kRate * 4);
  u16(4);
  u16(16);
  f.write("data", 4);
  u32(data);
  for (float v : s) {
    const auto q = static_cast<std::int16_t>(std::lround(std::clamp(v, -1.0F, 1.0F) * 32767.0F));
    u16(static_cast<std::uint16_t>(q));
  }
}

}  // namespace

TEST_CASE("Naturel : le module est absent et le son passe bit à bit", "[anime]") {
  const Stereo in = anime_mix(1.0);
  for (const Listen listen : {Listen::Speakers, Listen::Headphones, Listen::Cinema}) {
    const Stereo without = render(MixConfig{.listen = listen, .sample_rate = kRate}, in);
    const Stereo natural = render(
        MixConfig{.listen = listen, .anime = find_anime_level("natural")->tuning, .sample_rate = kRate}, in);
    CHECK(without == natural);
    CHECK_FALSE(find_anime_level("natural")->tuning.enabled);
  }
  // Enceintes sans profil : la stéréo est rendue telle quelle.
  CHECK(render(speakers(AnimeTuning{}), in) == in);
}

TEST_CASE("repli propre : mono, 5.1, Home cinéma, Fidèle et appareil multicanal ne sont pas touchés",
          "[anime]") {
  const Stereo in = anime_mix(0.5);
  const AnimeTuning on = find_anime_level("strong")->tuning;
  // Source mono doublée en stéréo par le décodeur.
  CHECK(render(speakers(on), in, 480, 1) == render(speakers(AnimeTuning{}), in, 480, 1));
  CHECK(render(phones(on), in, 480, 1) == render(phones(AnimeTuning{}), in, 480, 1));
  // 5.1 vers des enceintes stéréo : le downmix d'avant, sans module.
  Mixer with_anime(speakers(on)), without(speakers(AnimeTuning{}));
  with_anime.note_source(6);
  without.note_source(6);
  PcmBlock surround{.pts_us = 0, .channels = 6};
  surround.samples.assign(6 * 960, 0.0F);
  for (std::size_t f = 0; f < 960; ++f)
    surround.samples[f * 6 + 2] = 0.3F * std::sin(0.05F * static_cast<float>(f));
  CHECK(with_anime.process(surround).samples == without.process(surround).samples);
  // Home cinéma, Fidèle, sortie multicanal.
  CHECK(render(MixConfig{.listen = Listen::Cinema, .anime = on, .sample_rate = kRate}, in) == in);
  CHECK(render(MixConfig{.listen = Listen::Speakers, .anime = on, .faithful = true, .sample_rate = kRate},
               in) == in);
  Mixer wide(MixConfig{.listen = Listen::Speakers, .anime = on, .sink_channels = 6, .sample_rate = kRate});
  CHECK(wide.process(surround).samples == surround.samples);
}

TEST_CASE("la sonie BS.1770 : un sinus 1 kHz à -23 dBFS (deux voies) mesure -23,0 LUFS", "[anime][sonie]") {
  const double amp = std::pow(10.0, -23.0 / 20.0);
  for (const double hz : {1000.0, 997.0}) {
    const Stereo in = sine_stereo(hz, amp, 12.0);
    const double lufs = integrated_lufs(in, kRate);
    INFO("hz " << hz << " intégrée " << lufs);
    CHECK(lufs == Approx(-23.0).margin(0.1));
    LoudnessMeter meter(kRate);
    for (std::size_t f = 0; f < in.size() / 2; ++f) meter.add(in[f * 2], in[f * 2 + 1]);
    CHECK(meter.short_term() == Approx(-23.0).margin(0.1));
    CHECK(meter.momentary() == Approx(-23.0).margin(0.1));
  }
  // Une voie seule : 3 LU de moins (la puissance est celle d'une voie sur deux).
  Stereo left(sine_stereo(1000.0, amp, 6.0));
  for (std::size_t f = 0; f < left.size() / 2; ++f) left[f * 2 + 1] = 0.0F;
  CHECK(integrated_lufs(left, kRate) == Approx(-26.0).margin(0.1));
  // Le silence ne mesure rien et ne fait pas -inf + NaN.
  CHECK(integrated_lufs(Stereo(48'000 * 2, 0.0F), kRate) < -100.0);
}

TEST_CASE("la sonie vise la cible sans dépasser les bornes de gain", "[anime][sonie]") {
  // Trop faible : relevée de +6 dB au plus ; trop fort : baissé.
  AnimeTuning t = dry();
  t.loudness = true;
  t.target_lufs = -20.0F;
  const Stereo quiet = anime_mix(20.0, 0.06, 0.036);  // environ -34 LUFS
  const double quiet_in = integrated_lufs(quiet, kRate);
  const Stereo quiet_out = render(speakers(t), quiet);
  const Stereo tail(quiet_out.begin() + 12 * kRate * 2, quiet_out.end());
  const double raised = integrated_lufs(tail, kRate);
  // Référence : le même traitement sans la sonie (la voix portée et la présence relèvent déjà un peu).
  AnimeTuning no_loudness = t;
  no_loudness.loudness = false;
  const Stereo flat_out = render(speakers(no_loudness), quiet);
  const double flat = integrated_lufs(Stereo(flat_out.begin() + 12 * kRate * 2, flat_out.end()), kRate);
  INFO("faible : " << quiet_in << " -> " << raised);
  CHECK(raised > flat + 4.0);
  CHECK(raised < flat + 6.3);  // borne +6 dB
  const Stereo mid = anime_mix(20.0, 0.12, 0.07);
  const Stereo mid_out = render(speakers(t), mid);
  const double mid_in = integrated_lufs(Stereo(mid.begin() + 12 * kRate * 2, mid.end()), kRate);
  const double mid_after = integrated_lufs(Stereo(mid_out.begin() + 12 * kRate * 2, mid_out.end()), kRate);
  INFO("moyen : " << mid_in << " -> " << mid_after);
  CHECK(mid_after == Approx(-20.0).margin(1.5));
  const Stereo loud = anime_mix(20.0, 0.7, 0.4);
  const Stereo loud_out = render(speakers(t), loud);
  const double loud_after = integrated_lufs(Stereo(loud_out.begin() + 12 * kRate * 2, loud_out.end()), kRate);
  INFO("fort : -> " << loud_after);
  CHECK(loud_after == Approx(-20.0).margin(2.0));
}

TEST_CASE("compatibilité mono : la somme G+D ne perd rien par rapport à Naturel", "[anime][mono]") {
  const Stereo in = anime_mix(6.0);
  const double reference = rms_db(mono_sum(render(speakers(AnimeTuning{}), in)), kRate);
  for (const char* key : {"anime", "strong"}) {
    for (const bool headphones : {false, true}) {
      AnimeTuning t = dry(find_anime_level(key)->tuning);  // sans sonie : on isole l'annulation de phase
      const Stereo out = render(headphones ? phones(t) : speakers(t), in);
      const double sum = rms_db(mono_sum(out), kRate);
      INFO(key << (headphones ? " casque" : " enceintes") << " : " << sum - reference << " dB");
      CHECK(sum - reference > -3.0);
    }
  }
  // Aux enceintes, le milieu n'est jamais retiré : la somme ne baisse même pas d'un dixième de dB hors voix.
  AnimeTuning flat = dry();
  flat.voice_db = 0.0F;
  flat.presence_db = 0.0F;
  const Stereo out = render(speakers(flat), in);
  CHECK(rms_db(mono_sum(out), kRate) - reference == Approx(0.0).margin(0.1));
}

TEST_CASE("aucun écrêtage, y compris en vrai-crête", "[anime][limiteur]") {
  // Sinus à fs/4 avec une phase de 45 degrés : les échantillons valent 0,707 A mais la crête vraie vaut A.
  const double amp = 1.4;  // une piste mal masterisée, au-delà de 0 dBFS
  Stereo in(48'000 * 2 * 3);
  for (std::size_t f = 0; f < in.size() / 2; ++f) {
    const auto v = static_cast<float>(
        amp * std::sin(kTwoPi * 12000.0 * static_cast<double>(f) / kRate + std::numbers::pi / 4));
    in[f * 2] = v;
    in[f * 2 + 1] = v;
  }
  for (const bool headphones : {false, true}) {
    AnimeTuning t = find_anime_level("anime")->tuning;
    t.loudness = false;
    t.ceiling_db = -1.0F;
    const Stereo out = render(headphones ? phones(t) : speakers(t), in);
    const Stereo settled(out.begin() + 4800 * 2, out.end());
    const double tp = true_peak_db(settled, kRate);
    INFO((headphones ? "casque" : "enceintes") << " crête échantillons " << peak_abs(out) << " vraie " << tp);
    CHECK(peak_abs(out) <= 1.0);
    CHECK(tp <= -1.0 + 0.3);  // le plafond de -1 dBTP, à la précision de l'interpolation près
  }
  // Une musique forte, par blocs de tailles variées.
  const Stereo loud = anime_mix(6.0, 1.2, 0.8);
  AnimeTuning t = find_anime_level("strong")->tuning;
  const Stereo out = render(speakers(t), loud, 333);
  CHECK(peak_abs(out) <= 1.0);
  CHECK(true_peak_db(Stereo(out.begin() + 2 * kRate, out.end()), kRate) <= -1.0 + 0.3);
}

TEST_CASE("latence mesurée par impulsion : 20 ms au plus", "[anime][latence]") {
  Stereo in(kRate * 2, 0.0F);
  in[2000 * 2] = 0.5F;
  in[2000 * 2 + 1] = 0.5F;
  for (const bool headphones : {false, true}) {
    for (const char* key : {"anime", "strong"}) {
      const Stereo out = render(
          headphones ? phones(find_anime_level(key)->tuning) : speakers(find_anime_level(key)->tuning), in);
      const double top = peak_abs(out);
      std::size_t first = 0;
      for (std::size_t f = 0; f < out.size() / 2; ++f) {
        if (std::fabs(out[f * 2]) > 0.3 * top || std::fabs(out[f * 2 + 1]) > 0.3 * top) {
          first = f;
          break;
        }
      }
      const double ms = 1000.0 * static_cast<double>(first - 2000) / kRate;
      INFO(key << (headphones ? " casque" : " enceintes") << " : " << ms << " ms");
      CHECK(ms >= 0.0);
      CHECK(ms <= 20.0);
    }
  }
  AnimeSound module(find_anime_level("anime")->tuning, kRate, false);
  CHECK(1000.0 * module.latency_frames() / kRate < 5.0);
}

TEST_CASE("réponse en fréquence : plate à 1 dB près hors de l'égaliseur voulu", "[anime][reponse]") {
  // Module neutre (voix, présence, graves, sonie, compresseur à zéro) : un sinus centré garde son niveau.
  AnimeTuning neutral = dry();
  neutral.voice_db = 0.0F;
  neutral.presence_db = 0.0F;
  for (const double hz : {50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, 16000.0}) {
    const double g = centre_gain_db(speakers(neutral), hz);
    INFO(hz << " Hz : " << g << " dB");
    CHECK(std::fabs(g) <= 1.0);
  }
  // Réglages « Anime » par défaut (sans dynamique) : plats hors de la bande de la voix (300 Hz à 5 kHz) ; les
  // graves gardent leur fondamental malgré les harmoniques.
  AnimeTuning anime = find_anime_level("anime")->tuning;
  anime.loudness = false;
  anime.compress = 0.0F;
  for (const double hz : {45.0, 60.0, 90.0, 150.0, 250.0, 6000.0, 9000.0, 14000.0}) {
    const double g = centre_gain_db(speakers(anime), hz);
    INFO(hz << " Hz : " << g << " dB");
    CHECK(std::fabs(g) <= 1.0);
  }
  // L'égaliseur voulu : la voix est bien portée dans sa bande (+2 dB vers 1 kHz, +1,5 dB de présence à 3
  // kHz).
  CHECK(centre_gain_db(speakers(anime), 1000.0) == Approx(2.0).margin(0.7));
  CHECK(centre_gain_db(speakers(anime), 3000.0) > 1.0);
  // Casque : le centre passe par l'enceinte virtuelle à 0° (ni retard ni ombre) ; la réponse du binaural
  // reste dans ±1,5 dB de 50 Hz à 16 kHz (le crossfeed d'avant : -3,7 dB à 1 kHz).
  for (const double hz : {50.0, 100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0}) {
    const double g = centre_gain_db(phones(neutral), hz);
    INFO("casque " << hz << " Hz : " << g << " dB");
    CHECK(std::fabs(g) <= 1.5);
  }
}

TEST_CASE("la spatialisation change la corrélation mesurablement, la voix reste au centre",
          "[anime][stereo]") {
  const Stereo music = synth_music(8.0, 0.2);
  const double natural = correlation(music, kRate);
  for (const bool headphones : {false, true}) {
    AnimeTuning t = dry(find_anime_level("anime")->tuning);
    const Stereo out = render(headphones ? phones(t) : speakers(t), music);
    const double after = correlation(out, kRate);
    INFO((headphones ? "casque " : "enceintes ")
         << "corrélation " << natural << " -> " << after << ", largeur " << width_db(music, kRate) << " -> "
         << width_db(out, kRate));
    if (!headphones) {
      CHECK(std::fabs(after - natural) > 0.03);
      CHECK(width_db(out, kRate) > width_db(music, kRate) + 0.5);
    } else {
      CHECK(std::fabs(after - natural) > 0.03);
    }
  }
  // Une voix seule, exactement au centre (G = D) : elle reste centrée de part et d'autre.
  const std::vector<float> voice = synth_voice(6.0, 0.3);
  Stereo centred(voice.size() * 2);
  for (std::size_t f = 0; f < voice.size(); ++f) centred[f * 2] = centred[f * 2 + 1] = voice[f];
  for (const bool headphones : {false, true}) {
    AnimeTuning t = dry(find_anime_level("strong")->tuning);
    const Stereo out = render(headphones ? phones(t) : speakers(t), centred);
    INFO((headphones ? "casque" : "enceintes")
         << " corrélation de la voix " << correlation(out, kRate) << " largeur " << width_db(out, kRate));
    CHECK(correlation(out, kRate) > 0.999);
    CHECK(width_db(out, kRate) < -40.0);
  }
}

TEST_CASE("la voix est portée : le centre gagne de 1 à 3 dB, les côtés n'en reçoivent pas", "[anime][voix]") {
  AnimeTuning t = dry();
  t.voice_db = 3.0F;
  t.presence_db = 0.0F;
  CHECK(centre_gain_db(speakers(t), 1000.0) == Approx(3.0).margin(0.5));
  // Un sinus d'un seul côté (G seul) : milieu et côtés à égalité, la part du milieu seule est portée, la
  // somme G+D prend les +3 dB de la voix mais pas les côtés.
  const Stereo left = sine_stereo(1000.0, 0.1, 2.0, 0.0);
  const Stereo out = render(speakers(t), left);
  const double sum_in = tone_amplitude(mono_sum(left), 1000.0, 0, 60'000, 96'000);
  const double sum_out = tone_amplitude(mono_sum(out), 1000.0, 0, 60'000, 96'000);
  CHECK(20.0 * std::log10(sum_out / sum_in) == Approx(3.0).margin(0.5));
}

TEST_CASE("graves : harmoniques ajoutées, fondamental conservé, jamais d'écrêtage", "[anime][basses]") {
  AnimeTuning t = dry();
  t.bass = 1.0F;
  t.voice_db = 0.0F;
  t.presence_db = 0.0F;
  const Stereo in = sine_stereo(60.0, 0.3, 3.0);
  const Stereo out = render(speakers(t), in);
  const std::size_t from = 60'000, to = 132'000;  // 1,5 s de plus : 90 périodes entières
  const double fundamental =
      20.0 * std::log10(tone_amplitude(out, 60.0, 0, from, to) / tone_amplitude(in, 60.0, 0, from, to));
  const double third = tone_amplitude(out, 180.0, 0, from, to);
  INFO("fondamental " << fundamental << " dB, 3f " << third);
  CHECK(std::fabs(fundamental) <= 1.0);
  CHECK(third > 0.005);  // des harmoniques existent (3f à 180 Hz)
  CHECK(tone_amplitude(in, 180.0, 0, from, to) < 1e-4);
  CHECK(peak_abs(out) <= 1.0);
}

TEST_CASE("NaN, infini et valeurs énormes : la sortie reste finie et se rétablit", "[anime][robuste]") {
  Stereo in = anime_mix(1.0);
  in[100] = std::numeric_limits<float>::quiet_NaN();
  in[201] = std::numeric_limits<float>::infinity();
  in[300] = -std::numeric_limits<float>::infinity();
  in[400] = 1.0e30F;
  for (const bool headphones : {false, true}) {
    const Stereo out = render(headphones ? phones(find_anime_level("strong")->tuning)
                                         : speakers(find_anime_level("strong")->tuning),
                              in, 256);
    bool finite = true;
    for (float v : out) finite = finite && std::isfinite(v) && std::fabs(v) <= 1.0F;
    CHECK(finite);
    CHECK(rms_db(Stereo(out.begin() + kRate, out.end())) > -80.0);  // pas de silence bloqué après l'incident
  }
}

TEST_CASE("silence : rien à la sortie, aucun gain emballé, reprise normale", "[anime][silence]") {
  Stereo in(kRate * 4 * 2, 0.0F);
  const Stereo voice = anime_mix(2.0, 0.2, 0.1);
  in.insert(in.end(), voice.begin(), voice.end());
  const Stereo out = render(speakers(find_anime_level("strong")->tuning), in);
  CHECK(peak_abs(Stereo(out.begin(), out.begin() + kRate * 4 * 2)) < 1e-6);
  CHECK(rms_db(Stereo(out.begin() + kRate * 5 * 2, out.end())) > -50.0);
  AnimeSound module(find_anime_level("strong")->tuning, kRate, false);
  PcmBlock silent{.pts_us = 0, .channels = 2};
  silent.samples.assign(kRate * 2 * 5, 0.0F);
  module.finish(silent);
  CHECK(module.loudness_gain_db() == 0.0F);  // le silence est sous la porte : le gain est tenu
  // Un bloc vide ne casse rien.
  PcmBlock empty{.pts_us = 0, .channels = 2};
  module.finish(empty);
  CHECK(module.stage(PcmBlock{.pts_us = 0, .channels = 2}).samples.empty());
}

TEST_CASE("des blocs de tailles variées donnent exactement le même son", "[anime][blocs]") {
  const Stereo in = anime_mix(3.0);
  for (const bool headphones : {false, true}) {
    const MixConfig config = headphones ? phones(find_anime_level("strong")->tuning)
                                        : speakers(find_anime_level("strong")->tuning);
    const Stereo whole = render(config, in, in.size() / 2);
    for (const std::size_t block : {1UL, 7UL, 64UL, 333UL, 1024UL, 4799UL}) {
      const Stereo split = render(config, in, block);
      REQUIRE(split.size() == whole.size());
      double worst = 0.0;
      for (std::size_t i = 0; i < whole.size(); ++i)
        worst = std::max(worst, static_cast<double>(std::fabs(whole[i] - split[i])));
      INFO((headphones ? "casque" : "enceintes") << " blocs de " << block);
      CHECK(worst < 1e-6);
    }
  }
}

TEST_CASE("réglages : tableau de 16 cases, bornes, niveaux nommés", "[anime][reglages]") {
  for (const AnimeLevel& level : anime_levels()) {
    const AnimeTuning back = anime_from_wire(to_wire(level.tuning));
    INFO(level.key);
    CHECK(back == clamped(level.tuning));
  }
  CHECK(find_anime_level("anime")->tuning.enabled);
  CHECK(find_anime_level("strong")->tuning.voice_db > find_anime_level("anime")->tuning.voice_db);
  CHECK(find_anime_level("inconnu") == nullptr);
  const AnimeTuning wild = clamped(AnimeTuning{.enabled = true,
                                               .width = 9.0F,
                                               .voice_db = -4.0F,
                                               .target_lufs = -5.0F,
                                               .compress = std::numeric_limits<float>::quiet_NaN(),
                                               .ceiling_db = 3.0F});
  CHECK(wild.width == 1.0F);
  CHECK(wild.voice_db == 0.0F);
  CHECK(wild.target_lufs == -16.0F);
  CHECK(std::isfinite(wild.compress));
  CHECK(wild.ceiling_db == -0.5F);
  // Les champs existants du tableau MixTuning restent à leur place (compatibilité).
  CHECK(to_wire(MixTuning{}).size() == to_wire(AnimeTuning{}).size());
}

// ---- Échantillons avant/après et tableau de mesures (test caché)
// --------------------------------------------
namespace {

struct Bands {
  std::array<double, 6> db{};
};

// Spectre en 6 bandes (dBFS de l'énergie de la bande), par FFT de 4096 points moyennée (Hann), voie milieu.
Bands band_levels(const Stereo& s) {
  constexpr std::size_t kN = 4096;
  std::vector<double> power(kN / 2, 0.0);
  const std::size_t frames = s.size() / 2;
  int windows = 0;
  for (std::size_t at = kRate; at + kN <= frames; at += kN) {
    std::vector<std::complex<double>> x(kN);
    for (std::size_t i = 0; i < kN; ++i) {
      const double w = 0.5 - 0.5 * std::cos(kTwoPi * static_cast<double>(i) / kN);
      x[i] = w * 0.5 * (s[(at + i) * 2] + s[(at + i) * 2 + 1]);
    }
    // FFT itérative radix-2.
    for (std::size_t i = 1, j = 0; i < kN; ++i) {
      std::size_t bit = kN >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) std::swap(x[i], x[j]);
    }
    for (std::size_t len = 2; len <= kN; len <<= 1) {
      const std::complex<double> wl = std::polar(1.0, -kTwoPi / static_cast<double>(len));
      for (std::size_t i = 0; i < kN; i += len) {
        std::complex<double> w = 1.0;
        for (std::size_t k = 0; k < len / 2; ++k) {
          const auto u = x[i + k];
          const auto v = x[i + k + len / 2] * w;
          x[i + k] = u + v;
          x[i + k + len / 2] = u - v;
          w *= wl;
        }
      }
    }
    for (std::size_t k = 0; k < kN / 2; ++k) power[k] += std::norm(x[k]);
    ++windows;
  }
  const std::array<double, 7> edges{20.0, 150.0, 500.0, 2000.0, 4000.0, 8000.0, 20000.0};
  Bands out;
  for (std::size_t b = 0; b < 6; ++b) {
    double sum = 0.0;
    for (std::size_t k = 0; k < kN / 2; ++k) {
      const double hz = static_cast<double>(k) * kRate / kN;
      if (hz >= edges[b] && hz < edges[b + 1]) sum += power[k];
    }
    out.db[b] = 10.0 * std::log10(std::max(sum / std::max(windows, 1) / (kN * kN) * 8.0 / 3.0, 1e-20));
  }
  return out;
}

void report(const char* title, const char* name, const Stereo& s) {
  const Bands b = band_levels(s);
  std::printf("%-26s %-14s %6.1f %7.1f %6.2f %6.1f %6.1f | %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f\n", title,
              name, rms_db(s), integrated_lufs(s, kRate), correlation(s), width_db(s), true_peak_db(s, kRate),
              b.db[0], b.db[1], b.db[2], b.db[3], b.db[4], b.db[5]);
}

}  // namespace

TEST_CASE("échantillons avant/après et tableau de mesures", "[.anime_echantillons]") {
  const auto dir = out_dir();
  std::printf("\n%-26s %-14s %6s %7s %6s %6s %6s | %6s %6s %6s %6s %6s %6s\n", "signal", "version", "RMS",
              "LUFS", "corr", "larg.", "TP", "<150", "-500", "-2k", "-4k", "-8k", ">8k");
  std::vector<std::pair<std::string, Stereo>> sources;
  sources.emplace_back("synthese", anime_mix(14.0));
  // Un extrait de la bande son d'un vrai épisode, si présent (préparé par ffmpeg en WAV stéréo 48 kHz).
  const std::filesystem::path real = dir / "extrait_reel_avant.wav";
  if (std::filesystem::exists(real)) {
    std::ifstream f(real, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::string head(raw.begin(),
                           raw.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(raw.size(), 512)));
    const std::size_t at = head.find("data");  // ffmpeg ajoute un bloc LIST avant les données
    if (at != std::string::npos && raw.size() > at + 8) {
      const std::size_t start = at + 8;
      Stereo s((raw.size() - start) / 2);
      for (std::size_t i = 0; i < s.size(); ++i) {
        std::int16_t q;
        std::memcpy(&q, raw.data() + start + i * 2, 2);
        s[i] = static_cast<float>(q) / 32768.0F;
      }
      sources.emplace_back("episode", std::move(s));
    }
  }
  for (auto& [name, in] : sources) {
    write_wav(dir / (name + "_avant.wav"), in);
    report(name.c_str(), "naturel", in);
    report(name.c_str(), "casque_naturel",
           render(MixConfig{.listen = Listen::Headphones, .sample_rate = kRate}, in));
    for (const bool headphones : {false, true}) {
      for (const char* key : {"anime", "strong"}) {
        const Stereo out = render(
            headphones ? phones(find_anime_level(key)->tuning) : speakers(find_anime_level(key)->tuning), in);
        const std::string tag = std::string(headphones ? "casque_" : "enceintes_") + key;
        write_wav(dir / (name + "_" + tag + ".wav"), out);
        report(name.c_str(), tag.c_str(), out);
      }
    }
  }
}

TEST_CASE("tableau : réponse d'un sinus centré (dB) par mode", "[.anime_reponse]") {
  AnimeTuning neutral = dry();
  neutral.voice_db = 0.0F;
  neutral.presence_db = 0.0F;
  AnimeTuning anime = find_anime_level("anime")->tuning;
  anime.loudness = false;
  anime.compress = 0.0F;
  std::printf("\n%-22s", "Hz");
  const std::array<double, 12> freqs{50, 100, 200, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000};
  for (double hz : freqs) std::printf("%7.0f", hz);
  std::printf("\n");
  const auto row = [&](const char* name, const MixConfig& config) {
    std::printf("%-22s", name);
    for (double hz : freqs) std::printf("%+7.2f", centre_gain_db(config, hz));
    std::printf("\n");
  };
  row("enceintes neutre", speakers(neutral));
  row("enceintes anime", speakers(anime));
  row("casque neutre", phones(neutral));
  row("casque anime", phones(anime));
  row("casque naturel", MixConfig{.listen = Listen::Headphones, .sample_rate = kRate});
}
