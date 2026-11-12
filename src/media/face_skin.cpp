#include "media/face_skin.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <optional>
#include <span>

#include "media/yuv_sample.h"

namespace ajp::media {
namespace {

// Les points d'une image : luma et chroma ramenées à leur plage (0 à 1, chroma centrée), comme
// measure_frame (shot_probe.cpp).
class Reader {
 public:
  Reader(const DecodedFrame& frame, const yuv::Layout& layout) : layout_(layout) {
    const bool full = frame.color.range == ColorRange::Full;
    const double scale = std::ldexp(1.0, layout.bits - 8);
    y_off_ = full ? 0.0 : 16.0 * scale;
    y_range_ = (full ? 255.0 : 219.0) * scale;
    c_range_ = (full ? 255.0 : 224.0) * scale;
    c_mid_ = 128.0 * scale;
    const bool bt601 = frame.color.matrix == ColorMatrix::Bt601;
    const bool bt2020 = frame.color.matrix == ColorMatrix::Bt2020Ncl;
    kr_ = bt601 ? 0.299 : bt2020 ? 0.2627 : 0.2126;
    kb_ = bt601 ? 0.114 : bt2020 ? 0.0593 : 0.0722;
    // Les plans lus une fois : host_plane est virtuel, et la réduction lit des millions de points.
    for (int i = 0; i < (layout.semi_planar ? 2 : 3); ++i) {
      planes_.at(static_cast<std::size_t>(i)) = frame.data->host_plane(i);
      strides_.at(static_cast<std::size_t>(i)) = frame.data->host_stride(i);
    }
  }
  double luma(int x, int y) const {
    return (yuv::sample(planes_[0], strides_[0], x, y, layout_) - y_off_) / y_range_;
  }
  // R'G'B' encodés (0 à 1).
  std::array<double, 3> rgb(int x, int y) const {
    const double yy = luma(x, y);
    double cb = 0.0;
    double cr = 0.0;
    const auto u_plane = planes_[1];
    const std::size_t u_stride = strides_[1];
    if (layout_.semi_planar) {
      cb = yuv::sample(u_plane, u_stride, x / 2, y / 2, layout_, 0, 2);
      cr = yuv::sample(u_plane, u_stride, x / 2, y / 2, layout_, 1, 2);
    } else {
      cb = yuv::sample(u_plane, u_stride, x / 2, y / 2, layout_);
      cr = yuv::sample(planes_[2], strides_[2], x / 2, y / 2, layout_);
    }
    cb = (cb - c_mid_) / c_range_;
    cr = (cr - c_mid_) / c_range_;
    const double r = yy + (2.0 * (1.0 - kr_) * cr);
    const double b = yy + (2.0 * (1.0 - kb_) * cb);
    const double g = (yy - (kr_ * r) - (kb_ * b)) / (1.0 - kr_ - kb_);
    return {std::clamp(r, 0.0, 1.0), std::clamp(g, 0.0, 1.0), std::clamp(b, 0.0, 1.0)};
  }

 private:
  yuv::Layout layout_;
  std::array<std::span<const std::byte>, 3> planes_{};
  std::array<std::size_t, 3> strides_{};
  double y_off_ = 0.0, y_range_ = 1.0, c_range_ = 1.0, c_mid_ = 0.0, kr_ = 0.2126, kb_ = 0.0722;
};

// Un canal encodé en lumière (l'échelle n'importe pas : la teinte et la saturation n'en dépendent pas).
double to_linear(double v, Transfer transfer) {
  if (transfer == Transfer::Pq) {
    const double p = std::pow(v, 1.0 / 78.84375);
    return std::pow(std::max(p - 0.8359375, 0.0) / (18.8515625 - (18.6875 * p)), 1.0 / 0.1593017578125);
  }
  if (transfer == Transfer::Hlg) {  // l'inverse de l'OETF, sans l'OOTF de l'écran
    constexpr double a = 0.17883277, b = 0.28466892, c = 0.55991073;
    return v <= 0.5 ? v * v / 3.0 : (std::exp((v - c) / a) + b) / 12.0;
  }
  return std::pow(v, 2.4);  // BT.1886, noir à 0
}

// La teinte d'un point en lumière BT.2020, comme le grade la voit (la source passée en BT.2020 par
// libplacebo).
std::optional<SkinTone> tone_at(const Reader& reader, const DecodedFrame& frame, int x, int y) {
  const auto e = reader.rgb(x, y);
  std::array<double, 3> l{};
  for (std::size_t i = 0; i < 3; ++i) l.at(i) = to_linear(e.at(i), frame.color.transfer);
  if (frame.color.primaries != Primaries::Bt2020) {  // BT.709 vers BT.2020 (BT.2087)
    l = {(0.6274 * l[0]) + (0.3293 * l[1]) + (0.0433 * l[2]),
         (0.0691 * l[0]) + (0.9195 * l[1]) + (0.0114 * l[2]),
         (0.0164 * l[0]) + (0.0880 * l[1]) + (0.8956 * l[2])};
  }
  return skin_tone_of(static_cast<float>(l[0]), static_cast<float>(l[1]), static_cast<float>(l[2]));
}

// La luma réduite à kFaceSearchHeight lignes (ou moins, une image plus petite garde sa taille), par moyenne.
GrayImage search_image(const Reader& reader, const DecodedFrame& frame) {
  const int height = std::min(frame.height, kFaceSearchHeight);
  const int width = std::max(1, static_cast<int>(std::lround(static_cast<double>(frame.width) * height /
                                                             static_cast<double>(frame.height))));
  GrayImage gray{.width = width, .height = height};
  gray.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    const int y0 = y * frame.height / height;
    const int y1 = std::max(y0 + 1, (y + 1) * frame.height / height);
    for (int x = 0; x < width; ++x) {
      const int x0 = x * frame.width / width;
      const int x1 = std::max(x0 + 1, (x + 1) * frame.width / width);
      // Au plus 4 × 4 points par case : une image 4K en a 13 × 13, que la moyenne n'a pas besoin de tous
      // lire.
      const int step_x = std::max(1, (x1 - x0) / 4);
      const int step_y = std::max(1, (y1 - y0) / 4);
      double sum = 0.0;
      int n = 0;
      for (int sy = y0; sy < y1; sy += step_y)
        for (int sx = x0; sx < x1; sx += step_x, ++n) sum += reader.luma(sx, sy);
      const double mean = sum / static_cast<double>(n);
      gray.pixels[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) +
                  static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(std::lround(std::clamp(mean, 0.0, 1.0) * 255.0));
    }
  }
  return gray;
}

// La palette d'un visage : une grille de 8 × 6 points sous les yeux (x de 30 à 70 % du cadre, y de 55 à
// 85 %), ni cheveux ni yeux ; le quart au moins doit être d'une teinte de peau (R ≥ V ≥ B) : la bouche, le
// cou, une mèche ou une ombre bleutée prennent souvent le reste.
std::optional<SkinTone> palette_of(const Reader& reader, const DecodedFrame& frame, const FaceBox& box) {
  constexpr int kCols = 8;
  constexpr int kRows = 6;
  std::vector<SkinTone> samples;
  for (int j = 0; j < kRows; ++j) {
    const float fy =
        box.y0 + ((box.y1 - box.y0) * (0.55F + (0.30F * (static_cast<float>(j) + 0.5F) / kRows)));
    for (int i = 0; i < kCols; ++i) {
      const float fx =
          box.x0 + ((box.x1 - box.x0) * (0.30F + (0.40F * (static_cast<float>(i) + 0.5F) / kCols)));
      const int x = std::clamp(static_cast<int>(fx * static_cast<float>(frame.width)), 0, frame.width - 1);
      const int y = std::clamp(static_cast<int>(fy * static_cast<float>(frame.height)), 0, frame.height - 1);
      if (const auto tone = tone_at(reader, frame, x, y)) samples.push_back(*tone);
    }
  }
  if (samples.size() * 4 < static_cast<std::size_t>(kCols * kRows)) return std::nullopt;
  return face_palette(samples);
}

}  // namespace

std::vector<SkinTone> frame_skin_tones(const DecodedFrame& frame, const FaceCascade& cascade) {
  std::vector<SkinTone> tones;
  const std::optional<yuv::Layout> layout = yuv::layout_of(frame.format);
  if (!layout || frame.data == nullptr || frame.data->memory() != FrameMemory::Host || frame.width < 2 ||
      frame.height < 2)
    return tones;
  const Reader reader(frame, *layout);
  for (const FaceBox& box : cascade.detect(search_image(reader, frame)))
    if (const auto tone = palette_of(reader, frame, box)) tones.push_back(*tone);
  return tones;
}

void FaceSearch::look(const DecodedFrame& frame, ShotFrame& measured) {
  if (cascade_ == nullptr) return;
  if (seen_++ % kFaceEvery == 0) measured.faces = frame_skin_tones(frame, *cascade_);
}

std::string FaceSearch::summary(const ShotTimeline& timeline) const {
  if (cascade_ == nullptr) return {};
  const auto shots = timeline.shots();
  const auto with = std::ranges::count_if(shots, [](const Shot& s) { return s.meta.skins.count > 0; });
  return std::format(", visages dans {} plans", with);
}

}  // namespace ajp::media
