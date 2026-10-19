// Vocabulaire du runtime TensorRT, sans en-tête TensorRT : ce que le moteur
// déclare de ses entrées et sorties, et leurs profils de formes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::gpu::trt {

// Types des tenseurs d'entrée et de sortie ; Other couvre FP8, INT4, FP4…,
// qu'aucun modèle livré n'expose et que le runtime refuse de lier.
enum class DataType : std::uint8_t { F32, F16, BF16, I8, U8, I32, I64, Bool, Other };

std::string_view name(DataType type);

// Octets d'un élément ; 0 pour Other.
std::size_t element_bytes(DataType type);

enum class TensorIo : std::uint8_t { Input, Output };

// Dimensions dans l'ordre du moteur (NCHW pour nos modèles) ; -1 marque une
// dimension que seul le profil borne.
using Dims = std::vector<std::int64_t>;

// Bornes d'une entrée pour un profil d'optimisation, dimension par dimension.
struct ShapeRange {
  Dims min;
  Dims opt;
  Dims max;
};

struct TensorDesc {
  std::string name;
  TensorIo io = TensorIo::Input;
  DataType type = DataType::F32;
  Dims dims;                         // telles que le moteur les déclare
  std::vector<ShapeRange> profiles;  // un par profil, pour les entrées seulement
};

// « 1x3x720x1280 », l'écriture de trtexec.
std::string dims_text(std::span<const std::int64_t> dims);

// Octets d'un tenseur entièrement connu ; refuse une dimension négative ou un
// produit qui déborde.
std::expected<std::size_t, Refusal> byte_count(DataType type, std::span<const std::int64_t> dims);

}  // namespace ajp::gpu::trt
