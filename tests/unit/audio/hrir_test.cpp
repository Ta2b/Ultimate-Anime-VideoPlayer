#include "audio/hrir.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "audio/mixer.h"

using namespace ajp::audio;
using Catch::Approx;

namespace {

void put16(std::vector<std::byte>& out, std::uint32_t v) {
  out.push_back(static_cast<std::byte>(v & 0xFFU));
  out.push_back(static_cast<std::byte>((v >> 8U) & 0xFFU));
}
void put32(std::vector<std::byte>& out, std::uint32_t v) {
  put16(out, v & 0xFFFFU);
  put16(out, v >> 16U);
}
void put_tag(std::vector<std::byte>& out, const char* tag) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>(tag[i]));
}

// Un WAV synthétique : lanes[canal][trame] dans [-1, 1], en PCM 16 ou 24 bits, ou en float 32.
std::vector<std::byte> make_wav(const std::vector<std::vector<float>>& lanes, int rate, int bits,
                                bool floating, int format_override = -1) {
  const auto channels = static_cast<std::uint32_t>(lanes.size());
  const std::size_t frames = lanes.empty() ? 0 : lanes[0].size();
  const auto bytes = static_cast<std::uint32_t>(bits / 8);
  const auto data_size = static_cast<std::uint32_t>(frames) * channels * bytes;
  std::vector<std::byte> wav;
  put_tag(wav, "RIFF");
  put32(wav, 36 + data_size);
  put_tag(wav, "WAVE");
  put_tag(wav, "fmt ");
  put32(wav, 16);
  put16(wav, format_override >= 0 ? static_cast<std::uint32_t>(format_override) : (floating ? 3U : 1U));
  put16(wav, channels);
  put32(wav, static_cast<std::uint32_t>(rate));
  put32(wav, static_cast<std::uint32_t>(rate) * channels * bytes);
  put16(wav, channels * bytes);
  put16(wav, static_cast<std::uint32_t>(bits));
  put_tag(wav, "data");
  put32(wav, data_size);
  for (std::size_t f = 0; f < frames; ++f) {
    for (const auto& lane : lanes) {
      const float v = lane[f];
      if (floating) {
        std::uint32_t raw = 0;
        std::memcpy(&raw, &v, 4);
        put32(wav, raw);
      } else if (bits == 16) {
        put16(wav,
              static_cast<std::uint32_t>(static_cast<std::int32_t>(std::lround(v * 32767.0F))) & 0xFFFFU);
      } else {
        const auto s = static_cast<std::int32_t>(std::lround(v * 8388607.0F));
        const auto u = static_cast<std::uint32_t>(s) & 0xFFFFFFU;
        wav.push_back(static_cast<std::byte>(u & 0xFFU));
        wav.push_back(static_cast<std::byte>((u >> 8U) & 0xFFU));
        wav.push_back(static_cast<std::byte>((u >> 16U) & 0xFFU));
      }
    }
  }
  return wav;
}

// Une suite pseudo-aléatoire reproductible dans [-1, 1].
std::vector<float> noise(std::size_t n, std::uint32_t seed) {
  std::vector<float> out(n);
  std::uint32_t state = seed;
  for (float& v : out) {
    state = state * 1664525U + 1013904223U;
    v = static_cast<float>(state >> 8U) / 8388608.0F - 1.0F;
  }
  return out;
}

// Une réponse stéréo (FL et FR) : la même impulsion décalée pour chaque oreille, pour des filtres simples.
Hrir simple_hrir(std::size_t length, std::uint32_t seed) {
  Hrir hrir;
  hrir.sample_rate = 48'000;
  std::uint32_t s = seed;
  for (std::size_t p = 0; p < 2; ++p) {
    for (std::size_t e = 0; e < 2; ++e) {
      hrir.taps[p][e] = noise(length, s++);
      for (float& v : hrir.taps[p][e]) v *= 0.01F;
    }
  }
  return hrir;
}

// La convolution directe d'un canal : référence.
std::vector<float> direct(const std::vector<float>& x, const std::vector<float>& h, std::size_t out_len) {
  std::vector<float> y(out_len, 0.0F);
  for (std::size_t n = 0; n < out_len; ++n)
    for (std::size_t k = 0; k < h.size() && k <= n; ++k)
      if (n - k < x.size()) y[n] += h[k] * x[n - k];
  return y;
}

}  // namespace

TEST_CASE("la FFT redonne le signal et reconnaît une sinusoïde", "[hrir]") {
  constexpr std::size_t kN = 64;
  Fft fft(kN);
  std::vector<Cpx> data(kN);
  for (std::size_t i = 0; i < kN; ++i)
    data[i].re = std::cos(2.0F * std::numbers::pi_v<float> * 5.0F * static_cast<float>(i) / kN);
  const std::vector<Cpx> original = data;
  fft.forward(data);
  CHECK(data[5].re == Approx(kN / 2.0F).margin(1e-3));
  CHECK(data[3].re == Approx(0.0F).margin(1e-3));
  fft.inverse(data);
  for (std::size_t i = 0; i < kN; ++i) CHECK(data[i].re == Approx(original[i].re).margin(1e-4));
}

TEST_CASE("lecture d'un WAV de 14 canaux HeSuVi : chaque canal à sa place", "[hrir]") {
  std::vector<std::vector<float>> lanes(14, std::vector<float>(8, 0.0F));
  for (std::size_t c = 0; c < 14; ++c) lanes[c][0] = 0.05F * static_cast<float>(c + 1);
  const auto hrir = parse_hrir(make_wav(lanes, 44'100, 24, false));
  REQUIRE(hrir);
  CHECK(hrir->sample_rate == 44'100);
  CHECK(hrir->length() == 8);
  CHECK(hrir->taps[0][0][0] == Approx(0.05F).margin(1e-5));  // FL→G : canal 0
  CHECK(hrir->taps[0][1][0] == Approx(0.10F).margin(1e-5));  // FL→D : canal 1
  CHECK(hrir->taps[2][0][0] == Approx(0.35F).margin(1e-5));  // FC→G : canal 6
  CHECK(hrir->taps[2][1][0] == Approx(0.70F).margin(1e-5));  // FC→D : canal 13
  CHECK(hrir->taps[1][1][0] == Approx(0.40F).margin(1e-5));  // FR→D : canal 7
  CHECK(hrir->taps[1][0][0] == Approx(0.45F).margin(1e-5));  // FR→G : canal 8
  CHECK(hrir->taps[6][0][0] == Approx(0.15F).margin(1e-5));  // SL→G : canal 2
  CHECK(hrir->taps[5][0][0] == Approx(0.65F).margin(1e-5));  // BR→G : canal 12
  CHECK(hrir->taps[3][0].empty());                           // pas de LFE
}

TEST_CASE("lecture de PCM 16 bits, 24 bits et float 32, en 16 canaux et en HRIR stéréo", "[hrir]") {
  const std::vector<float> left{0.5F, -0.25F, 0.125F};
  const std::vector<float> right{-0.5F, 0.25F, 0.0F};
  for (const auto& [bits, floating] : {std::pair{16, false}, std::pair{24, false}, std::pair{32, true}}) {
    const auto hrir = parse_hrir(make_wav({left, right}, 48'000, bits, floating));
    REQUIRE(hrir);
    for (std::size_t i = 0; i < 3; ++i) {
      CHECK(hrir->taps[0][0][i] == Approx(left[i]).margin(1e-3));
      CHECK(hrir->taps[0][1][i] == Approx(right[i]).margin(1e-3));
      CHECK(hrir->taps[1][0][i] == Approx(right[i]).margin(1e-3));  // la droite : le miroir
      CHECK(hrir->taps[1][1][i] == Approx(left[i]).margin(1e-3));
    }
  }
  std::vector<std::vector<float>> lanes(16, std::vector<float>(4, 0.0F));
  lanes[7][1] = 0.5F;  // haut-parleur 3 (LFE), oreille droite
  const auto sixteen = parse_hrir(make_wav(lanes, 48'000, 32, true));
  REQUIRE(sixteen);
  CHECK(sixteen->taps[3][1][1] == Approx(0.5F));
}

TEST_CASE("fichiers invalides : refus nommés", "[hrir]") {
  const std::vector<std::vector<float>> stereo(2, std::vector<float>(16, 0.1F));
  CHECK(parse_hrir({}).error().code == ajp::RefusalCode::FileUnreadable);
  std::vector<std::byte> junk(64, std::byte{0x41});
  CHECK(parse_hrir(junk).error().code == ajp::RefusalCode::FormatUnsupported);
  // nombre de canaux inconnu
  CHECK(parse_hrir(make_wav(std::vector<std::vector<float>>(3, std::vector<float>(16)), 48'000, 16, false))
            .error()
            .code == ajp::RefusalCode::FormatUnsupported);
  // ADPCM, ou 8 bits : format inconnu
  CHECK(parse_hrir(make_wav(stereo, 48'000, 16, false, 2)).error().code ==
        ajp::RefusalCode::FormatUnsupported);
  CHECK(parse_hrir(make_wav(stereo, 48'000, 8, false)).error().code == ajp::RefusalCode::FormatUnsupported);
  // trop long
  const std::vector<std::vector<float>> big(2,
                                            std::vector<float>(static_cast<std::size_t>(kHrirMaxTaps) + 1));
  CHECK(parse_hrir(make_wav(big, 48'000, 16, false)).error().code == ajp::RefusalCode::FieldOutOfRange);
  const std::vector<std::vector<float>> max(2, std::vector<float>(static_cast<std::size_t>(kHrirMaxTaps)));
  CHECK(parse_hrir(make_wav(max, 48'000, 16, false)));
  // aucune trame
  CHECK(parse_hrir(make_wav(std::vector<std::vector<float>>(2), 48'000, 16, false)).error().code ==
        ajp::RefusalCode::FileUnreadable);
  // bloc fmt tronqué
  std::vector<std::byte> cut = make_wav(stereo, 48'000, 16, false);
  cut.resize(24);
  CHECK(parse_hrir(cut).error().code == ajp::RefusalCode::FileUnreadable);
  // fichier absent
  CHECK(load_hrir("/nonexistent/ajp/none.wav").error().code == ajp::RefusalCode::FileUnreadable);
}

TEST_CASE("rééchantillonnage : longueur, impulsion et sinusoïde", "[hrir]") {
  std::vector<float> impulse(100, 0.0F);
  impulse[0] = 1.0F;
  const auto same = resample(impulse, 48'000, 48'000);
  CHECK(same == impulse);
  const auto up = resample(impulse, 24'000, 48'000);
  CHECK(up.size() == 200);
  CHECK(up[0] == Approx(1.0F).margin(0.05));
  const auto down = resample(impulse, 96'000, 48'000);
  CHECK(down.size() == 50);
  CHECK(down[0] == Approx(0.5F).margin(0.05));  // l'impulsion garde son énergie en 1/2 d'amplitude (DC)
  // une sinusoïde de 1 kHz à 44,1 kHz garde son amplitude à 48 kHz
  std::vector<float> sine(4000);
  for (std::size_t i = 0; i < sine.size(); ++i)
    sine[i] = std::sin(2.0F * std::numbers::pi_v<float> * 1000.0F * static_cast<float>(i) / 44'100.0F);
  const auto converted = resample(sine, 44'100, 48'000);
  CHECK(converted.size() == 4354);
  for (std::size_t j = 200; j < 4000; j += 37) {
    const float expected =
        std::sin(2.0F * std::numbers::pi_v<float> * 1000.0F * static_cast<float>(j) / 48'000.0F);
    CHECK(converted[j] == Approx(expected).margin(0.01));
  }
  // trop long après rééchantillonnage : tronqué
  const std::vector<float> long_one(static_cast<std::size_t>(kHrirMaxTaps), 0.001F);
  CHECK(resample(long_one, 44'100, 48'000).size() == static_cast<std::size_t>(kHrirMaxTaps));
}

TEST_CASE("gain de sécurité : le pire signal plein-échelle reste sous 0,9", "[hrir]") {
  Hrir loud = simple_hrir(64, 7);
  for (auto& pair : loud.taps)
    for (auto& response : pair)
      for (float& v : response) v *= 30.0F;  // des réponses fortes
  const float gain = normalize(loud);
  CHECK(gain < 1.0F);
  // le signal de pire cas est le signe de la réponse renversée : sa sortie vaut la somme des |h|
  float worst = 0.0F;
  for (const auto& pair : loud.taps)
    for (const auto& response : pair) {
      float sum = 0.0F;
      for (const float v : response) sum += std::fabs(v);
      worst = std::max(worst, sum);
    }
  CHECK(worst == Approx(kHrirPeak).margin(1e-4));
  // des réponses faibles ne sont pas amplifiées
  Hrir quiet = simple_hrir(8, 3);
  const Hrir copy = quiet;
  CHECK(normalize(quiet) == 1.0F);
  CHECK(quiet.taps[0][0] == copy.taps[0][0]);
  // prepared() : taux de sortie et gain
  const Hrir at96 = prepared(loud, 96'000);
  CHECK(at96.sample_rate == 96'000);
  CHECK(at96.length() >= 120);
}

TEST_CASE("réponses de l'utilisateur : le niveau est respecté comme dans HeSuVi", "[hrir]") {
  // Une réponse dont la somme des |h| vaut 3 (bien au-dessus de 0,9) garde ses valeurs : pas de gain caché.
  Hrir strong;
  strong.sample_rate = 48'000;
  strong.taps[0][0] = {1.5F, 1.0F, 0.5F};
  strong.taps[0][1] = {0.2F};
  const Hrir kept = prepared(strong, 48'000);
  CHECK(kept.taps[0][0] == strong.taps[0][0]);
  CHECK(kept.taps[0][1] == strong.taps[0][1]);
}

TEST_CASE("la convolution d'une impulsion rend la réponse", "[hrir]") {
  const Hrir hrir = simple_hrir(1500, 11);  // trois partitions
  HrirConvolver conv(hrir, 2, 0.0F);
  constexpr std::size_t kFrames = 2048;
  std::vector<float> in(kFrames * 2, 0.0F);
  in[0] = 1.0F;  // impulsion sur FL
  std::vector<float> out;
  conv.process(in, out);
  REQUIRE(out.size() == kFrames * 2);
  const std::size_t latency = static_cast<std::size_t>(kHrirBlock);
  for (std::size_t i = 0; i < 1500; ++i) {
    CHECK(out[(latency + i) * 2] == Approx(hrir.taps[0][0][i]).margin(1e-5));
    CHECK(out[(latency + i) * 2 + 1] == Approx(hrir.taps[0][1][i]).margin(1e-5));
  }
  for (std::size_t i = 0; i < latency; ++i) CHECK(out[i * 2] == 0.0F);  // le bloc de latence
}

TEST_CASE("la convolution partitionnée égale la convolution directe", "[hrir]") {
  const Hrir hrir = simple_hrir(1300, 21);
  constexpr std::size_t kFrames = 3000;
  const std::vector<float> left = noise(kFrames, 5);
  const std::vector<float> right = noise(kFrames, 6);
  std::vector<float> in(kFrames * 2);
  for (std::size_t f = 0; f < kFrames; ++f) {
    in[f * 2] = left[f];
    in[f * 2 + 1] = right[f];
  }
  HrirConvolver conv(hrir, 2, 0.0F);
  // par morceaux de tailles quelconques, comme le flux réel
  std::vector<float> all;
  std::size_t at = 0;
  for (const std::size_t chunk : {1U, 700U, 333U, 1024U, 942U}) {
    std::vector<float> out;
    conv.process({in.data() + at * 2, chunk * 2}, out);
    REQUIRE(out.size() == chunk * 2);
    all.insert(all.end(), out.begin(), out.end());
    at += chunk;
  }
  REQUIRE(at == kFrames);
  const std::size_t latency = static_cast<std::size_t>(kHrirBlock);
  for (std::size_t ear = 0; ear < 2; ++ear) {
    const std::vector<float> from_left = direct(left, hrir.taps[0][ear], kFrames);
    const std::vector<float> from_right = direct(right, hrir.taps[1][ear], kFrames);
    for (std::size_t n = 0; n + latency < kFrames; n += 7) {
      const float expected = from_left[n] + from_right[n];
      CHECK(all[(n + latency) * 2 + ear] == Approx(expected).margin(2e-4));
    }
  }
  conv.reset();
  std::vector<float> after;
  conv.process({in.data(), 64}, after);
  CHECK(std::ranges::all_of(after, [](float v) { return v == 0.0F; }));
}

TEST_CASE("le LFE sans réponse passe directement aux deux oreilles", "[hrir]") {
  const Hrir hrir = simple_hrir(16, 2);
  HrirConvolver conv(hrir, 6, 0.3F);
  std::vector<float> in(1024 * 6, 0.0F);
  in[3] = 1.0F;  // LFE de la première trame
  std::vector<float> out;
  conv.process(in, out);
  const std::size_t latency = static_cast<std::size_t>(kHrirBlock);
  CHECK(out[latency * 2] == Approx(0.3F * 0.7071F).margin(1e-5));
  CHECK(out[latency * 2 + 1] == Approx(0.3F * 0.7071F).margin(1e-5));
}

TEST_CASE("le mixeur au casque : les réponses remplacent le modèle", "[hrir][mixer]") {
  auto hrir = std::make_shared<Hrir>(simple_hrir(200, 31));
  PcmBlock block{.pts_us = 0, .channels = 2};
  block.samples = noise(2 * 2048, 9);

  Mixer with({.listen = Listen::Headphones, .hrir = hrir});
  Mixer without({.listen = Listen::Headphones});
  const PcmBlock a = with.process(block);
  const PcmBlock b = without.process(block);
  CHECK(a.channels == 2);
  CHECK(a.frames() == block.frames());
  CHECK(a.samples != b.samples);
  // la latence : le premier bloc est du silence
  CHECK(std::ranges::all_of(std::span(a.samples).first(2 * kHrirBlock), [](float v) { return v == 0.0F; }));
  // sans HRIR, rien ne change pour les enceintes ni pour fidèle
  Mixer speakers({.listen = Listen::Speakers, .hrir = hrir});
  CHECK(speakers.process(block).samples == block.samples);
  Mixer faithful({.listen = Listen::Headphones, .faithful = true, .hrir = hrir});
  CHECK(faithful.process(block).samples == block.samples);
  // 5.1 : une impulsion sur l'avant gauche arrive aux deux oreilles, avec la latence
  PcmBlock surround{.pts_us = 0, .channels = 6};
  surround.samples.assign(6 * 1024, 0.0F);
  surround.samples[0] = 0.5F;
  Mixer five({.listen = Listen::Headphones, .hrir = hrir});
  const PcmBlock rendered = five.process(surround);
  REQUIRE(rendered.channels == 2);
  REQUIRE(rendered.frames() == 1024);
  float energy = 0.0F;
  for (const float v : rendered.samples) energy += v * v;
  CHECK(energy > 0.0F);
  // reset() oublie l'état
  five.reset();
  PcmBlock silent{.pts_us = 0, .channels = 6};
  silent.samples.assign(6 * 600, 0.0F);
  const PcmBlock quiet = five.process(silent);
  CHECK(std::ranges::all_of(quiet.samples, [](float v) { return std::fabs(v) < 1e-6F; }));
}
