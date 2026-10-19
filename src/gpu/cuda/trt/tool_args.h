// Lecture des arguments des outils ajp_trt_bench et ajp_trt_build.
#pragma once

#include <charconv>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <system_error>

#include "gpu/cuda/trt/types.h"

namespace ajp::gpu::trt::tool {

// Nombre écrit en entier dans text, sans espace ni reste.
template <typename T>
std::optional<T> parse_number(std::string_view text) {
  T value{};
  const char* first = std::to_address(text.begin());
  const char* last = std::to_address(text.end());
  const auto [end, error] = std::from_chars(first, last, value);
  if (text.empty() || error != std::errc{} || end != last) return std::nullopt;
  return value;
}

// « 1x3x720x1280 » : dimensions positives séparées par des x.
inline std::optional<Dims> parse_dims(std::string_view text) {
  Dims dims;
  while (!text.empty()) {
    const std::size_t x = text.find('x');
    const auto d = parse_number<std::int64_t>(text.substr(0, x));
    if (!d || *d < 1) return std::nullopt;
    dims.push_back(*d);
    text = x == std::string_view::npos ? std::string_view{} : text.substr(x + 1);
    if (x != std::string_view::npos && text.empty()) return std::nullopt;
  }
  if (dims.empty()) return std::nullopt;
  return dims;
}

}  // namespace ajp::gpu::trt::tool
