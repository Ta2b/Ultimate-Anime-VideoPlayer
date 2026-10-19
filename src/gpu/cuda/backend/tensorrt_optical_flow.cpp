// Flux optique matériel (nvof_session.h, optical_flow_kernels.h) dans le backend TensorRT :
// open_optical_flow et optical_flow de KernelOps, à la place de la variante mouvement de RIFE.
#include <format>
#include <string>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/optical_flow_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

bool fp16_planes(const Tensor& tensor, std::int64_t channels, int width, int height) {
  return tensor.type == TensorType::F16 && tensor.handle != 0 && tensor.shape.dims[0] == 1 &&
         tensor.shape.dims[1] == channels && tensor.shape.dims[2] == height && tensor.shape.dims[3] == width;
}

void* address(const Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

}  // namespace

std::expected<int, Refusal> TensorRtBackend::open_optical_flow(int width, int height) {
  if (auto ok = bind(); !ok) return std::unexpected(std::move(ok.error()));
  optical_flow_.reset();
  auto opened = NvofSession::open(stream_, width, height);
  if (!opened) return std::unexpected(std::move(opened.error()));
  optical_flow_ = std::move(*opened);
  log(BackendLogLevel::Info,
      std::format("flux optique matériel : {}x{}, grille {} ({}x{} vecteurs)", width, height,
                  optical_flow_->grid(), optical_flow_->cells_x(), optical_flow_->cells_y()));
  return optical_flow_->grid();
}

std::expected<void, Refusal> TensorRtBackend::optical_flow(const Tensor& a, const Tensor& b,
                                                           const Motion& motion) {
  if (optical_flow_ == nullptr)
    return refuse(RefusalCode::FieldMissing, "optical_flow : open_optical_flow d'abord");
  NvofSession& session = *optical_flow_;
  const int w = session.width();
  const int h = session.height();
  if (!fp16_planes(a, 3, w, h) || !fp16_planes(b, 3, w, h)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("optical_flow : deux images RGB FP16 de {}x{} attendues", w, h));
  }
  const int cx = session.cells_x();
  const int cy = session.cells_y();
  const kernels::Window& area = motion.area;
  if (!fp16_planes(motion.flow, 4, cx, cy) || !fp16_planes(motion.mask, 1, cx, cy) || area.x != 0 ||
      area.y != 0 || area.width != cx || area.height != cy) {
    return refuse(
        RefusalCode::ShapeInvalid,
        std::format("optical_flow : mouvement FP16 de {}x{} points, fenêtre entière, attendu", cx, cy));
  }
  using Direction = NvofSession::Direction;
  const NvofSession::Pitched a_forward = session.input(Direction::Forward);
  const NvofSession::Pitched a_backward = session.reference(Direction::Backward);
  const NvofSession::Pitched b_forward = session.reference(Direction::Forward);
  const NvofSession::Pitched b_backward = session.input(Direction::Backward);
  const NvofSession::Pitched forward = session.flow(Direction::Forward);
  const NvofSession::Pitched backward = session.flow(Direction::Backward);
  for (const NvofSession::Pitched& buffer :
       {a_forward, a_backward, b_forward, b_backward, forward, backward}) {
    if (buffer.data == nullptr)
      return refuse(RefusalCode::VendorCallFailed, "optical_flow : tampon du flux optique illisible");
  }
  if (auto ok = bind(); !ok) return ok;
  GpuTiming& timing = stage("flux optique");
  timing.begin(stream_);
  auto done = cuda_check(launch_luma8(Luma8Params{.rgb = address(a),
                                                  .width = w,
                                                  .height = h,
                                                  .first = a_forward.data,
                                                  .first_pitch = a_forward.pitch,
                                                  .second = a_backward.data,
                                                  .second_pitch = a_backward.pitch},
                                      stream_),
                         "luma8<<<>>> (a)");
  if (done) {
    done = cuda_check(launch_luma8(Luma8Params{.rgb = address(b),
                                               .width = w,
                                               .height = h,
                                               .first = b_forward.data,
                                               .first_pitch = b_forward.pitch,
                                               .second = b_backward.data,
                                               .second_pitch = b_backward.pitch},
                                   stream_),
                      "luma8<<<>>> (b)");
  }
  if (done) done = session.execute();
  if (done) {
    done = cuda_check(launch_mid_motion(MidMotionParams{.forward = forward.data,
                                                        .forward_pitch = forward.pitch,
                                                        .backward = backward.data,
                                                        .backward_pitch = backward.pitch,
                                                        .cells_x = cx,
                                                        .cells_y = cy,
                                                        .grid = session.grid(),
                                                        .flow = address(motion.flow),
                                                        .mask = address(motion.mask)},
                                        stream_),
                      "mid_motion<<<>>>");
  }
  timing.end(stream_);
  return done;
}

}  // namespace ajp::gpu::cuda
