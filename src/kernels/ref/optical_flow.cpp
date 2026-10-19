// Écrit pour ce dépôt (spec/optical_flow.h).
#include "kernels/ref/optical_flow.h"

#include <cmath>
#include <format>

#include "kernels/ref/check.h"

namespace ajp::kernels::ref {
namespace {

struct Vec {
  float x = 0.0F;
  float y = 0.0F;
};

Vec operator+(Vec a, Vec b) {
  return {a.x + b.x, a.y + b.y};
}
Vec operator-(Vec a) {
  return {-a.x, -a.y};
}
Vec operator*(float s, Vec a) {
  return {s * a.x, s * a.y};
}

// Lecture bilinéaire bornée d'un flot ; les plans ont été vérifiés par mid_motion.
Vec at(const FlowField& f, Vec p) {
  return {*sample(f.x, p.x, p.y), *sample(f.y, p.x, p.y)};
}

float confidence(float error) {
  const float r = error / kFlowConsistency;
  return 1.0F / (1.0F + r * r);
}

// Désaccord du déplacement v (de a vers b, vu de p) avec chaque image (ea, eb de la spec).
float disagreement_a(const FlowField& forward, Vec p, Vec v) {
  const Vec e = at(forward, p + (-0.5F) * v) + (-v);
  return std::hypot(e.x, e.y);
}
float disagreement_b(const FlowField& backward, Vec p, Vec v) {
  const Vec e = at(backward, p + 0.5F * v) + v;
  return std::hypot(e.x, e.y);
}

Result<void> check_field(const FlowField& f, std::string_view what) {
  if (auto ok = check_plane(f.x, what); !ok) return ok;
  if (auto ok = check_plane(f.y, what); !ok) return ok;
  return check_same_size(f.x, f.y, what);
}

}  // namespace

Result<FlowField> decode_flow(const Plane<std::int16_t>& x, const Plane<std::int16_t>& y, int grid) {
  if (grid != 1 && grid != 2 && grid != 4)
    return fail(ErrorCode::ParameterOutOfRange, std::format("flot : grille {} hors de 1, 2, 4", grid));
  if (auto ok = check_plane(x, "flot S10.5"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_plane(y, "flot S10.5"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_same_size(x, y, "flot S10.5"); !ok) return std::unexpected(ok.error());
  FlowField f{.x = PlaneF(x.width, x.height), .y = PlaneF(y.width, y.height)};
  for (std::size_t i = 0; i < x.data.size(); ++i) {
    f.x.data[i] = flow_to_grid(x.data[i], grid);
    f.y.data[i] = flow_to_grid(y.data[i], grid);
  }
  return f;
}

Result<Motion> mid_motion(const FlowField& forward, const FlowField& backward) {
  if (auto ok = check_field(forward, "flot avant"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_field(backward, "flot arrière"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_same_size(forward.x, backward.x, "flots avant et arrière"); !ok)
    return std::unexpected(ok.error());
  const int w = forward.x.width;
  const int h = forward.x.height;
  Motion motion{.flow = {PlaneF(w, h), PlaneF(w, h), PlaneF(w, h), PlaneF(w, h)},
                .mask = PlaneF(w, h),
                .area = Window{.x = 0, .y = 0, .width = w, .height = h}};
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      const Vec p{static_cast<float>(i), static_cast<float>(j)};
      Vec m = at(forward, p);
      Vec n = -at(backward, p);
      for (int k = 0; k < kMidFlowIterations; ++k) {
        m = at(forward, p + (-0.5F) * m);
        n = -at(backward, p + 0.5F * n);
      }
      const auto score = [&](Vec v) {
        return confidence(disagreement_a(forward, p, v)) * confidence(disagreement_b(backward, p, v));
      };
      const Vec mid = score(m) >= score(n) ? m : n;
      const float va = confidence(disagreement_a(forward, p, mid));
      const float vb = confidence(disagreement_b(backward, p, mid));
      motion.flow[0].at(i, j) = -0.5F * mid.x;
      motion.flow[1].at(i, j) = -0.5F * mid.y;
      motion.flow[2].at(i, j) = 0.5F * mid.x;
      motion.flow[3].at(i, j) = 0.5F * mid.y;
      motion.mask.at(i, j) = va / (va + vb);
    }
  }
  return motion;
}

}  // namespace ajp::kernels::ref
