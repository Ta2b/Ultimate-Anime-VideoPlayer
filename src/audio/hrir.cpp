#include "audio/hrir.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <numbers>
#include <string_view>

#include "audio/refuse.h"

namespace ajp::audio {
namespace {

constexpr std::uintmax_t kMaxFileBytes = 16U * 1024U * 1024U;  // 16 voies de 32 768 float = 2 Mio

std::uint32_t le32(const std::byte* p) {
  return std::to_integer<std::uint32_t>(p[0]) | (std::to_integer<std::uint32_t>(p[1]) << 8U) |
         (std::to_integer<std::uint32_t>(p[2]) << 16U) | (std::to_integer<std::uint32_t>(p[3]) << 24U);
}
std::uint32_t le16(const std::byte* p) {
  return std::to_integer<std::uint32_t>(p[0]) | (std::to_integer<std::uint32_t>(p[1]) << 8U);
}

bool tag(const std::byte* p, std::string_view name) {
  return std::memcmp(p, name.data(), 4) == 0;
}

// Où va le canal d'un fichier de 14 canaux HeSuVi : {position, oreille}.
constexpr std::array<std::array<int, 2>, 14> kHesuvi{{{0, 0},
                                                      {0, 1},
                                                      {6, 0},
                                                      {6, 1},
                                                      {4, 0},
                                                      {4, 1},
                                                      {2, 0},
                                                      {1, 1},
                                                      {1, 0},
                                                      {7, 1},
                                                      {7, 0},
                                                      {5, 1},
                                                      {5, 0},
                                                      {2, 1}}};

// Un échantillon, ramené à [-1, 1[ ; format : 1 PCM entier, 3 float.
float sample_at(const std::byte* p, int format, int bits) {
  if (format == 3) {
    const std::uint32_t raw = le32(p);
    return std::bit_cast<float>(raw);
  }
  if (bits == 16) {
    const auto v = static_cast<std::int16_t>(le16(p));
    return static_cast<float>(v) / 32768.0F;
  }
  std::int32_t v = static_cast<std::int32_t>(le16(p) | (std::to_integer<std::uint32_t>(p[2]) << 16U));
  if ((v & 0x800000) != 0) v -= 0x1000000;
  return static_cast<float>(v) / 8388608.0F;
}

}  // namespace

std::size_t Hrir::length() const {
  std::size_t longest = 0;
  for (const auto& pair : taps)
    for (const auto& response : pair) longest = std::max(longest, response.size());
  return longest;
}

std::expected<Hrir, Refusal> parse_hrir(std::span<const std::byte> wav) {
  if (wav.size() < 12)
    return refuse(RefusalCode::FileUnreadable, "réponse impulsionnelle : fichier vide ou trop court");
  if (!tag(wav.data(), "RIFF") || !tag(wav.data() + 8, "WAVE"))
    return refuse(RefusalCode::FormatUnsupported, "réponse impulsionnelle : ce n'est pas un fichier WAV");
  int format = 0;
  int channels = 0;
  int rate = 0;
  int bits = 0;
  int block_align = 0;
  const std::byte* data = nullptr;
  std::size_t data_bytes = 0;
  bool have_format = false;
  std::size_t at = 12;
  while (at + 8 <= wav.size()) {
    const std::byte* chunk = wav.data() + at;
    const std::size_t size = le32(chunk + 4);
    const std::size_t body = at + 8;
    const std::size_t available = std::min(size, wav.size() - body);
    if (tag(chunk, "fmt ")) {
      if (available < 16)
        return refuse(RefusalCode::FileUnreadable, "réponse impulsionnelle : bloc fmt tronqué");
      format = static_cast<int>(le16(chunk + 8));
      channels = static_cast<int>(le16(chunk + 10));
      rate = static_cast<int>(le32(chunk + 12));
      block_align = static_cast<int>(le16(chunk + 20));
      bits = static_cast<int>(le16(chunk + 22));
      if (format == 0xFFFE) {  // WAVE_FORMAT_EXTENSIBLE : le vrai format ouvre le sous-format
        if (available < 26)
          return refuse(RefusalCode::FileUnreadable, "réponse impulsionnelle : bloc fmt tronqué");
        format = static_cast<int>(le16(chunk + 8 + 24));
      }
      have_format = true;
    } else if (tag(chunk, "data")) {
      data = chunk + 8;
      data_bytes = available;
      break;  // le bloc fmt précède toujours data dans un WAV valide
    }
    at = body + size + (size & 1U);  // les blocs sont alignés sur 2 octets
  }
  if (!have_format || data == nullptr)
    return refuse(RefusalCode::FileUnreadable, "réponse impulsionnelle : blocs fmt ou data absents");
  const bool known = (format == 1 && (bits == 16 || bits == 24)) || (format == 3 && bits == 32);
  if (!known)
    return refuse(
        RefusalCode::FormatUnsupported,
        std::format("réponse impulsionnelle : format {} sur {} bits (PCM 16 ou 24 bits, ou float 32)", format,
                    bits));
  if (channels != 2 && channels != 14 && channels != 16)
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("réponse impulsionnelle : {} canaux (2, 14 HeSuVi ou 16)", channels));
  if (rate < 8'000 || rate > 384'000)
    return refuse(RefusalCode::FormatUnsupported, std::format("réponse impulsionnelle : taux {} Hz", rate));
  const int bytes = bits / 8;
  if (block_align != bytes * channels)
    return refuse(RefusalCode::FormatUnsupported, "réponse impulsionnelle : alignement des blocs inattendu");
  const std::size_t frames = data_bytes / static_cast<std::size_t>(block_align);
  if (frames == 0) return refuse(RefusalCode::FileUnreadable, "réponse impulsionnelle : aucune donnée");
  if (frames > static_cast<std::size_t>(kHrirMaxTaps))
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("réponse impulsionnelle : {} échantillons (au plus {})", frames, kHrirMaxTaps));

  // Les canaux du fichier, désentrelacés.
  std::vector<std::vector<float>> lanes(static_cast<std::size_t>(channels), std::vector<float>(frames));
  for (std::size_t f = 0; f < frames; ++f) {
    for (std::size_t c = 0; c < lanes.size(); ++c) {
      const std::byte* p = data + (f * lanes.size() + c) * static_cast<std::size_t>(bytes);
      const float v = sample_at(p, format, bits);
      lanes[c][f] = std::isfinite(v) ? v : 0.0F;
    }
  }
  Hrir hrir;
  hrir.sample_rate = rate;
  const auto put = [&](int position, int ear, std::vector<float> response) {
    hrir.taps[static_cast<std::size_t>(position)][static_cast<std::size_t>(ear)] = std::move(response);
  };
  if (channels == 14) {
    for (std::size_t c = 0; c < kHesuvi.size(); ++c) put(kHesuvi[c][0], kHesuvi[c][1], lanes[c]);
  } else if (channels == 16) {
    for (std::size_t c = 0; c < lanes.size(); ++c)
      put(static_cast<int>(c / 2), static_cast<int>(c % 2), lanes[c]);
  } else {
    std::vector<float> centre(frames);
    for (std::size_t f = 0; f < frames; ++f) centre[f] = 0.5F * (lanes[0][f] + lanes[1][f]);
    put(0, 0, lanes[0]);  // l'enceinte gauche
    put(0, 1, lanes[1]);
    put(1, 0, lanes[1]);  // la droite : le miroir
    put(1, 1, lanes[0]);
    put(2, 0, centre);
    put(2, 1, centre);
    for (const int base : {4, 6}) {  // arrières et côtés : la même direction, par côté
      put(base, 0, lanes[0]);
      put(base, 1, lanes[1]);
      put(base + 1, 0, lanes[1]);
      put(base + 1, 1, lanes[0]);
    }
  }
  return hrir;
}

std::expected<std::shared_ptr<const Hrir>, Refusal> load_hrir(const std::filesystem::path& path) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec) return refuse(RefusalCode::FileUnreadable, std::format("{} : {}", path.string(), ec.message()));
  if (size > kMaxFileBytes)
    return refuse(RefusalCode::FileUnreadable,
                  std::format("{} : fichier trop gros pour une réponse", path.string()));
  std::ifstream in(path, std::ios::binary);
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  if (!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
    return refuse(RefusalCode::FileUnreadable, std::format("{} : lecture impossible", path.string()));
  auto parsed = parse_hrir(bytes);
  if (!parsed) {
    parsed.error().detail = std::format("{} : {}", path.string(), parsed.error().detail);
    return std::unexpected(std::move(parsed.error()));
  }
  return std::make_shared<const Hrir>(std::move(*parsed));
}

std::vector<float> resample(std::span<const float> in, int from_rate, int to_rate) {
  if (in.empty() || from_rate <= 0 || to_rate <= 0) return {};
  if (from_rate == to_rate) {
    const std::size_t keep = std::min<std::size_t>(in.size(), kHrirMaxTaps);
    return {in.begin(), in.begin() + static_cast<std::ptrdiff_t>(keep)};
  }
  const double ratio = static_cast<double>(to_rate) / static_cast<double>(from_rate);
  const double cutoff = std::min(1.0, ratio);  // relative au Nyquist de l'entrée
  const double half = 16.0 / cutoff;           // demi-largeur du noyau, en échantillons d'entrée
  constexpr int kSteps = 256;                  // finesse de la table du noyau
  const auto table_size = static_cast<std::size_t>(std::ceil(half * kSteps)) + 2;
  std::vector<float> kernel(table_size);
  for (std::size_t i = 0; i < table_size; ++i) {
    const double x = static_cast<double>(i) / kSteps;
    const double arg = std::numbers::pi * cutoff * x;
    const double sinc = x == 0.0 ? 1.0 : std::sin(arg) / arg;
    const double window = x >= half ? 0.0 : 0.5 + 0.5 * std::cos(std::numbers::pi * x / half);
    kernel[i] = static_cast<float>(cutoff * sinc * window);
  }
  const auto wanted = static_cast<std::size_t>(std::ceil(static_cast<double>(in.size()) * ratio));
  std::vector<float> out(std::min<std::size_t>(wanted, kHrirMaxTaps));
  const auto last = static_cast<std::ptrdiff_t>(in.size()) - 1;
  for (std::size_t j = 0; j < out.size(); ++j) {
    const double t = static_cast<double>(j) / ratio;
    const auto lo = std::max<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(std::ceil(t - half)));
    const auto hi = std::min<std::ptrdiff_t>(last, static_cast<std::ptrdiff_t>(std::floor(t + half)));
    float sum = 0.0F;
    for (std::ptrdiff_t i = lo; i <= hi; ++i) {
      const double pos = std::fabs(t - static_cast<double>(i)) * kSteps;
      const auto k = static_cast<std::size_t>(pos);
      const auto frac = static_cast<float>(pos - static_cast<double>(k));
      sum += in[static_cast<std::size_t>(i)] * (kernel[k] + frac * (kernel[k + 1] - kernel[k]));
    }
    out[j] = sum;
  }
  return out;
}

float normalize(Hrir& hrir) {
  float loudest = 0.0F;
  for (const auto& pair : hrir.taps) {
    for (const auto& response : pair) {
      float sum = 0.0F;
      for (const float v : response) sum += std::fabs(v);
      loudest = std::max(loudest, sum);
    }
  }
  if (loudest <= kHrirPeak) return 1.0F;
  const float gain = kHrirPeak / loudest;
  for (auto& pair : hrir.taps)
    for (auto& response : pair)
      for (float& v : response) v *= gain;
  return gain;
}

Hrir prepared(const Hrir& hrir, int sample_rate) {
  Hrir out;
  out.sample_rate = sample_rate;
  for (std::size_t p = 0; p < hrir.taps.size(); ++p)
    for (std::size_t e = 0; e < 2; ++e)
      out.taps[p][e] = resample(hrir.taps[p][e], hrir.sample_rate, sample_rate);
  normalize(out);
  return out;
}

// ---- FFT ----

Fft::Fft(std::size_t size) : size_(size), reverse_(size), twiddle_(size / 2) {
  const int bits = std::countr_zero(size);
  for (std::size_t i = 0; i < size; ++i) {
    std::size_t r = 0;
    for (int b = 0; b < bits; ++b)
      if ((i >> b) & 1U) r |= std::size_t{1} << (bits - 1 - b);
    reverse_[i] = r;
  }
  for (std::size_t k = 0; k < size / 2; ++k) {
    const double angle = -2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(size);
    twiddle_[k] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
  }
}

void Fft::run(std::span<Cpx> data, bool inverse) const {
  for (std::size_t i = 0; i < size_; ++i)
    if (i < reverse_[i]) std::swap(data[i], data[reverse_[i]]);
  for (std::size_t len = 2; len <= size_; len <<= 1U) {
    const std::size_t step = size_ / len;
    const std::size_t half = len / 2;
    for (std::size_t start = 0; start < size_; start += len) {
      for (std::size_t k = 0; k < half; ++k) {
        Cpx w = twiddle_[k * step];
        if (inverse) w.im = -w.im;
        Cpx& a = data[start + k];
        Cpx& b = data[start + k + half];
        const Cpx t{b.re * w.re - b.im * w.im, b.re * w.im + b.im * w.re};
        b = {a.re - t.re, a.im - t.im};
        a = {a.re + t.re, a.im + t.im};
      }
    }
  }
}

void Fft::forward(std::span<Cpx> data) const {
  run(data, false);
}

void Fft::inverse(std::span<Cpx> data) const {
  run(data, true);
  const float scale = 1.0F / static_cast<float>(size_);
  for (Cpx& c : data) {
    c.re *= scale;
    c.im *= scale;
  }
}

// ---- convolution ----

namespace {

constexpr std::size_t kBlock = static_cast<std::size_t>(kHrirBlock);
constexpr std::size_t kFft = kBlock * 2;
constexpr std::size_t kBins = kBlock + 1;

// Le haut-parleur (position dans Hrir::taps) du canal c d'une source à 2, 6 ou 8 canaux.
std::size_t position_of(int source_channels, std::size_t c) {
  if (source_channels == 6 && c >= 4) return c + 2;  // SL SR : après les arrières
  return c;
}

}  // namespace

HrirConvolver::HrirConvolver(const Hrir& hrir, int source_channels, float lfe_gain)
    : channels_(source_channels), fft_(kFft) {
  const auto n = static_cast<std::size_t>(channels_);
  filters_.resize(n);
  history_.resize(n);
  previous_.assign(n, std::vector<float>(kBlock, 0.0F));
  work_.resize(kFft);
  accumulator_.resize(kBins);
  const std::size_t length = hrir.length();
  partitions_ = std::max<std::size_t>(1, (length + kBlock - 1) / kBlock);
  for (std::size_t c = 0; c < n; ++c) {
    const std::size_t position = position_of(channels_, c);
    for (std::size_t ear = 0; ear < 2; ++ear) {
      std::vector<float> response = hrir.taps[position][ear];
      if (response.empty() && position == 3 && lfe_gain > 0.0F) response = {lfe_gain * 0.7071F};
      Filter& filter = filters_[c][ear];
      for (std::size_t offset = 0; offset < response.size(); offset += kBlock) {
        std::ranges::fill(work_, Cpx{});
        const std::size_t count = std::min(kBlock, response.size() - offset);
        for (std::size_t i = 0; i < count; ++i) work_[i].re = response[offset + i];
        fft_.forward(work_);
        filter.parts.emplace_back(work_.begin(), work_.begin() + static_cast<std::ptrdiff_t>(kBins));
      }
    }
    if (!filters_[c][0].parts.empty() || !filters_[c][1].parts.empty())
      history_[c].assign(partitions_, Spectrum(kBins));
  }
  reset();
}

void HrirConvolver::reset() {
  for (auto& ring : history_)
    for (Spectrum& s : ring) std::ranges::fill(s, Cpx{});
  for (auto& p : previous_) std::ranges::fill(p, 0.0F);
  head_ = 0;
  pending_in_.clear();
  ready_out_.assign(kBlock * 2, 0.0F);  // la latence : un bloc de silence
}

void HrirConvolver::process_block(std::span<const float> in, std::span<float> out) {
  const auto n = static_cast<std::size_t>(channels_);
  head_ = (head_ + 1) % partitions_;
  for (std::size_t c = 0; c < n; ++c) {
    if (history_[c].empty()) continue;
    std::vector<float>& prev = previous_[c];
    for (std::size_t i = 0; i < kBlock; ++i) {
      work_[i] = {prev[i], 0.0F};
      const float v = in[i * n + c];
      work_[kBlock + i] = {v, 0.0F};
      prev[i] = v;
    }
    fft_.forward(work_);
    std::copy_n(work_.begin(), kBins, history_[c][head_].begin());
  }
  for (std::size_t ear = 0; ear < 2; ++ear) {
    std::ranges::fill(accumulator_, Cpx{});
    for (std::size_t c = 0; c < n; ++c) {
      const Filter& filter = filters_[c][ear];
      for (std::size_t p = 0; p < filter.parts.size(); ++p) {
        const Spectrum& x = history_[c][(head_ + partitions_ - p) % partitions_];
        const Spectrum& h = filter.parts[p];
        for (std::size_t k = 0; k < kBins; ++k) {
          accumulator_[k].re += h[k].re * x[k].re - h[k].im * x[k].im;
          accumulator_[k].im += h[k].re * x[k].im + h[k].im * x[k].re;
        }
      }
    }
    accumulator_[0].im = 0.0F;
    accumulator_[kBlock].im = 0.0F;
    std::copy(accumulator_.begin(), accumulator_.end(), work_.begin());
    for (std::size_t k = kBins; k < kFft; ++k) {  // le spectre d'un signal réel est symétrique
      work_[k] = {accumulator_[kFft - k].re, -accumulator_[kFft - k].im};
    }
    fft_.inverse(work_);
    for (std::size_t i = 0; i < kBlock; ++i) out[i * 2 + ear] = work_[kBlock + i].re;
  }
}

void HrirConvolver::process(std::span<const float> in, std::vector<float>& out) {
  const auto n = static_cast<std::size_t>(channels_);
  pending_in_.insert(pending_in_.end(), in.begin(), in.end());
  std::array<float, kBlock * 2> block{};
  std::size_t used = 0;
  while (pending_in_.size() - used >= kBlock * n) {
    process_block({pending_in_.data() + used, kBlock * n}, block);
    ready_out_.insert(ready_out_.end(), block.begin(), block.end());
    used += kBlock * n;
  }
  pending_in_.erase(pending_in_.begin(), pending_in_.begin() + static_cast<std::ptrdiff_t>(used));
  const std::size_t wanted = in.size() / n * 2;
  out.assign(ready_out_.begin(), ready_out_.begin() + static_cast<std::ptrdiff_t>(wanted));
  ready_out_.erase(ready_out_.begin(), ready_out_.begin() + static_cast<std::ptrdiff_t>(wanted));
}

}  // namespace ajp::audio
