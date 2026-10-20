#include "gpu/vulkan/backend/net_weights.h"

#include <bit>
#include <cstring>

#include "kernels/ref/half.h"

namespace ajp::gpu::vk::net {
namespace {

// Valeurs rangées en float ou en binary16 (au plus proche, égalités au pair).
class Packer {
 public:
  Packer(std::size_t count, BlobType type)
      : type_(type), bytes_(count * (type == BlobType::F16 ? 2 : 4), std::byte{0}) {}

  void put(std::size_t index, float value) {
    if (type_ == BlobType::F16) {
      const std::uint16_t bits = kernels::ref::float_to_half(value);
      std::memcpy(bytes_.data() + (index * 2), &bits, 2);
    } else {
      std::memcpy(bytes_.data() + (index * 4), &value, 4);
    }
  }

  std::vector<std::byte> take() { return std::move(bytes_); }

 private:
  BlobType type_;
  std::vector<std::byte> bytes_;
};

std::size_t at(const ir::Dims& d, std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t e) {
  return static_cast<std::size_t>((((((a * d[1]) + b) * d[2]) + c) * d[3]) + e);
}

}  // namespace

std::vector<std::byte> conv_weights(const ir::Tensor& w, std::uint32_t groups, std::uint32_t lanes,
                                    BlobType type) {
  const ir::Dims& d = w.dims;  // [cout][cing][kh][kw]
  const std::int64_t cout_g = d[0] / groups;
  const std::int64_t blocks = (cout_g + lanes - 1) / lanes;
  const std::int64_t cing = d[1];
  Packer out(static_cast<std::size_t>(groups * blocks * d[2] * d[3] * cing * lanes), type);
  std::size_t pos = 0;
  for (std::int64_t g = 0; g < groups; ++g) {
    for (std::int64_t b = 0; b < blocks; ++b) {
      for (std::int64_t ky = 0; ky < d[2]; ++ky) {
        for (std::int64_t kx = 0; kx < d[3]; ++kx) {
          for (std::int64_t ci = 0; ci < cing; ++ci) {
            for (std::int64_t l = 0; l < lanes; ++l, ++pos) {
              const std::int64_t o = (b * lanes) + l;
              if (o < cout_g) out.put(pos, w.value.f[at(d, (g * cout_g) + o, ci, ky, kx)]);
            }
          }
        }
      }
    }
  }
  return out.take();
}

std::vector<std::byte> conv_transpose_weights(const ir::Tensor& w, std::uint32_t groups,
                                              std::uint32_t lanes) {
  const ir::Dims& d = w.dims;  // [cin][cout_g][kh][kw]
  const std::int64_t cing = d[0] / groups;
  const std::int64_t cout_g = d[1];
  const std::int64_t blocks = (cout_g + lanes - 1) / lanes;
  Packer out(static_cast<std::size_t>(groups * blocks * d[2] * d[3] * cing * lanes), BlobType::F32);
  std::size_t pos = 0;
  for (std::int64_t g = 0; g < groups; ++g) {
    for (std::int64_t b = 0; b < blocks; ++b) {
      for (std::int64_t ky = 0; ky < d[2]; ++ky) {
        for (std::int64_t kx = 0; kx < d[3]; ++kx) {
          for (std::int64_t ci = 0; ci < cing; ++ci) {
            for (std::int64_t l = 0; l < lanes; ++l, ++pos) {
              const std::int64_t o = (b * lanes) + l;
              if (o < cout_g) out.put(pos, w.value.f[at(d, (g * cing) + ci, o, ky, kx)]);
            }
          }
        }
      }
    }
  }
  return out.take();
}

std::vector<std::byte> matrix_weights(const ir::Tensor& w, std::uint32_t cin_padded,
                                      std::uint32_t cout_padded) {
  const ir::Dims& d = w.dims;  // [cout][cin][kh][kw]
  const std::size_t rows = static_cast<std::size_t>(d[2] * d[3]) * cin_padded;
  Packer out(rows * cout_padded, BlobType::F16);
  for (std::int64_t ky = 0; ky < d[2]; ++ky) {
    for (std::int64_t kx = 0; kx < d[3]; ++kx) {
      for (std::int64_t ci = 0; ci < d[1]; ++ci) {
        const std::size_t row =
            (static_cast<std::size_t>((ky * d[3]) + kx) * cin_padded) + static_cast<std::size_t>(ci);
        for (std::int64_t co = 0; co < d[0]; ++co) {
          out.put((row * cout_padded) + static_cast<std::size_t>(co), w.value.f[at(d, co, ci, ky, kx)]);
        }
      }
    }
  }
  return out.take();
}

std::vector<std::byte> channel_floats(const ir::Tensor* values, std::uint32_t count) {
  Packer out(count, BlobType::F32);
  if (values != nullptr) {
    for (std::size_t k = 0; k < values->value.f.size() && k < count; ++k) out.put(k, values->value.f[k]);
  }
  return out.take();
}

std::vector<std::byte> constant_blob(const ir::Tensor& constant) {
  const BlobType type = constant.type == ir::DataType::F16 ? BlobType::F16 : BlobType::F32;
  Packer out(constant.value.f.size(), type);
  for (std::size_t k = 0; k < constant.value.f.size(); ++k) out.put(k, constant.value.f[k]);
  return out.take();
}

}  // namespace ajp::gpu::vk::net
