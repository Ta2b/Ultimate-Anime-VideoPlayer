// Ce que l'exécuteur garde des images : des tenseurs recyclés et des contenus partagés.
//
// Un contenu réunit les états d'une même image (RGB de la chaîne, RGB préparé pour RIFE,
// RGB de sortie). Une sortie qui montre une entrée telle quelle (Source, REPEATED) partage
// son contenu : l'agrandissement d'une source sert à toutes ses reprises.
#pragma once

#include <cstddef>
#include <expected>
#include <format>
#include <memory>
#include <utility>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/ref_slots.h"
#include "gpu/tensor.h"

namespace ajp::gpu {

struct PooledTensor {
  std::unique_ptr<TensorBuffer> buffer;
};

using TensorSlots = RefSlots<PooledTensor, false>;
using TensorRef = TensorSlots::Ref;

inline const Tensor& tensor_of(const TensorRef& ref) {
  return ref->buffer->tensor();
}

// Tenseurs du backend, alloués au premier besoin puis repris quand leur type et leur forme
// conviennent. Un tenseur libre d'une autre forme cède sa place quand la réserve est pleine.
class TensorPool {
 public:
  TensorPool(Backend& backend, std::size_t capacity) : backend_(&backend), slots_(capacity) {}

  // Refus : celui de Backend::alloc_tensor, ou réserve pleine de tenseurs tous pris
  // (ShapeInvalid : l'exécuteur garde plus d'images que ses bornes ne le prévoient).
  std::expected<TensorRef, Refusal> acquire(TensorType type, const Shape& shape) {
    const auto same = [&](const PooledTensor& pooled) {
      return pooled.buffer->tensor().type == type && pooled.buffer->tensor().shape == shape;
    };
    if (auto reused = slots_.take_free(same)) return std::move(*reused);
    if (!slots_.can_put()) {
      return std::unexpected(
          Refusal{RefusalCode::ShapeInvalid, std::format("réserve de {} tenseurs pleine", slots_.created())});
    }
    auto buffer = backend_->alloc_tensor(type, shape);
    if (!buffer) return std::unexpected(std::move(buffer.error()));
    auto put = slots_.put(PooledTensor{std::move(*buffer)});  // can_put() : une place existe
    if (!put) return std::unexpected(Refusal{RefusalCode::ShapeInvalid, "réserve de tenseurs pleine"});
    return std::move(*put);
  }

  std::size_t created() const { return slots_.created(); }
  std::size_t in_use() const { return slots_.in_use(); }

 private:
  Backend* backend_;
  TensorSlots slots_;
};

struct Content {
  TensorRef low;   // RGB de la chaîne, à la résolution de la source
  TensorRef rife;  // RGB préparé pour RIFE, rempli au multiple de 64
  TensorRef high;  // RGB à la résolution de sortie
};

using ContentSlots = RefSlots<Content, true>;
using ContentRef = ContentSlots::Ref;

}  // namespace ajp::gpu
