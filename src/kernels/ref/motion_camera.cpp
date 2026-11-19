#include "kernels/ref/motion_camera.h"

#include <cmath>
#include <format>

#include "kernels/spec/motion_camera.h"

namespace kirari::kernels::ref {

Result<Motion> motion_camera(const std::array<float, 6>& camera, float scale, int plane_width,
                             int plane_height, const Window& area) {
  if (plane_width < 1 || plane_height < 1 || area.x < 0 || area.y < 0 || area.width < 1 || area.height < 1 ||
      area.x + area.width > plane_width || area.y + area.height > plane_height) {
    return fail(ErrorCode::DimensionOutOfRange,
                std::format("mouvement de la caméra : fenêtre {}x{} en ({}, {}) dans des plans {}x{}",
                            area.width, area.height, area.x, area.y, plane_width, plane_height));
  }
  if (!(scale > 0.0F) || !std::isfinite(scale)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("mouvement de la caméra : échelle {}", scale));
  }
  for (const float c : camera) {
    if (!std::isfinite(c)) return fail(ErrorCode::ParameterOutOfRange, "mouvement de la caméra : non fini");
  }
  Motion out{.flow = {PlaneF(plane_width, plane_height, 0.0F), PlaneF(plane_width, plane_height, 0.0F),
                      PlaneF(plane_width, plane_height, 0.0F), PlaneF(plane_width, plane_height, 0.0F)},
             .mask = PlaneF(plane_width, plane_height, 0.5F),
             .area = area};
  for (int y = area.y; y < area.y + area.height; ++y) {
    for (int x = area.x; x < area.x + area.width; ++x) {
      const CameraShift d = camera_shift(camera, scale, x - area.x, y - area.y);
      out.flow[0].at(x, y) = -0.5F * d.x;
      out.flow[1].at(x, y) = -0.5F * d.y;
      out.flow[2].at(x, y) = 0.5F * d.x;
      out.flow[3].at(x, y) = 0.5F * d.y;
    }
  }
  return out;
}

}  // namespace kirari::kernels::ref
