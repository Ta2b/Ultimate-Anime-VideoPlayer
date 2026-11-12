// Les pointeurs d'un paquet « .span8 » pour SpanFp8Net::create : un seul endroit pour le lecteur, la
// dérivation native et l'outil ajp_span_fp8_run, qui en avait une copie sans acc16 (A02-11).
#pragma once

#include <cstddef>

#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/span_fp8/package.h"

namespace ajp::gpu::cuda {

// Les pointeurs du paquet, pour SpanFp8Net::create et SpanFp8Reference::create : valables autant que lui.
inline SpanFp8Weights span_fp8_view(const span_fp8::Package& package) {
  const auto view = [](const span_fp8::Conv& conv) {
    return SpanFp8ConvView{.weights = conv.weights.data(), .bias = conv.bias.data()};
  };
  SpanFp8Weights weights;
  weights.blocks = static_cast<int>(package.blocks.size());
  weights.scales = package.scales.data();
  weights.first = view(package.first);
  for (std::size_t b = 0; b < package.blocks.size(); ++b) {
    for (std::size_t i = 0; i < 3; ++i) weights.conv[b][i] = view(package.blocks[b].conv[i]);
    weights.attention[b] = view(package.blocks[b].attention);
  }
  weights.concat = view(package.concat);
  weights.pixel = view(package.pixel);
  weights.acc16 = package.acc16;
  weights.channels = package.real_channels;
  return weights;
}

}  // namespace ajp::gpu::cuda
