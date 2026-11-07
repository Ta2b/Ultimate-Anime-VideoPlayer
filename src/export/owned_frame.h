// Une image FFmpeg possédée (P2.32) : av_frame_free à sa fin, sans en-tête de FFmpeg ici.
#pragma once

#include <memory>

struct AVFrame;

namespace ajp::exporter {

struct OwnedFrameFree {
  void operator()(AVFrame* frame) const noexcept;  // export_writer.cpp
};
using OwnedFrame = std::unique_ptr<AVFrame, OwnedFrameFree>;

}  // namespace ajp::exporter
