// Description des entrées et sorties d'un ICudaEngine et de leurs profils.
#include <format>
#include <string>

#include "gpu/cuda/trt/engine_impl.h"

namespace ajp::gpu::trt {
namespace {

DataType data_type(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return DataType::F32;
    case nvinfer1::DataType::kHALF:
      return DataType::F16;
    case nvinfer1::DataType::kBF16:
      return DataType::BF16;
    case nvinfer1::DataType::kINT8:
      return DataType::I8;
    case nvinfer1::DataType::kUINT8:
      return DataType::U8;
    case nvinfer1::DataType::kINT32:
      return DataType::I32;
    case nvinfer1::DataType::kINT64:
      return DataType::I64;
    case nvinfer1::DataType::kBOOL:
      return DataType::Bool;
    default:
      return DataType::Other;
  }
}

Dims dims_of(const nvinfer1::Dims& dims) {
  Dims out;
  for (int k = 0; k < dims.nbDims; ++k) out.push_back(dims.d[k]);
  return out;
}

Refusal refuse(RefusalCode code, const std::string& tensor, std::string_view why) {
  return Refusal{code, std::format("tenseur « {} » : {}", tensor, why)};
}

// Bornes du profil pour une entrée. Un moteur entièrement statique n'a pas de
// profil déclaré : TensorRT rend alors des dimensions invalides, et les bornes
// sont les formes du moteur.
std::expected<ShapeRange, Refusal> profile_range(const nvinfer1::ICudaEngine& engine, const TensorDesc& desc,
                                                 int profile) {
  using nvinfer1::OptProfileSelector;
  const char* name = desc.name.c_str();
  const nvinfer1::Dims min = engine.getProfileShape(name, profile, OptProfileSelector::kMIN);
  const nvinfer1::Dims opt = engine.getProfileShape(name, profile, OptProfileSelector::kOPT);
  const nvinfer1::Dims max = engine.getProfileShape(name, profile, OptProfileSelector::kMAX);
  const auto rank = static_cast<int>(desc.dims.size());
  if (min.nbDims == rank && opt.nbDims == rank && max.nbDims == rank) {
    return ShapeRange{.min = dims_of(min), .opt = dims_of(opt), .max = dims_of(max)};
  }
  for (const std::int64_t d : desc.dims) {
    if (d < 0) {
      return std::unexpected(refuse(RefusalCode::EngineInvalid, desc.name,
                                    std::format("dimension dynamique sans borne au profil {}", profile)));
    }
  }
  return ShapeRange{.min = desc.dims, .opt = desc.dims, .max = desc.dims};
}

}  // namespace

std::expected<std::vector<TensorDesc>, Refusal> describe_io(const nvinfer1::ICudaEngine& engine) {
  std::vector<TensorDesc> tensors;
  const int profiles = engine.getNbOptimizationProfiles();
  for (int k = 0; k < engine.getNbIOTensors(); ++k) {
    const char* raw_name = engine.getIOTensorName(k);
    if (raw_name == nullptr) {
      return std::unexpected(Refusal{RefusalCode::TensorRtFailed, std::format("getIOTensorName({}) nul", k)});
    }
    TensorDesc desc{.name = raw_name};
    const nvinfer1::TensorIOMode mode = engine.getTensorIOMode(raw_name);
    if (mode == nvinfer1::TensorIOMode::kNONE) {
      return std::unexpected(refuse(RefusalCode::TensorRtFailed, desc.name, "ni entrée ni sortie"));
    }
    desc.io = mode == nvinfer1::TensorIOMode::kINPUT ? TensorIo::Input : TensorIo::Output;
    desc.type = data_type(engine.getTensorDataType(raw_name));
    desc.dims = dims_of(engine.getTensorShape(raw_name));
    if (desc.type == DataType::Other) {
      return std::unexpected(refuse(RefusalCode::FormatUnsupported, desc.name,
                                    "type hors de F32, F16, BF16, "
                                    "I8, U8, I32, I64 et Bool"));
    }
    if (engine.isShapeInferenceIO(raw_name) ||
        engine.getTensorLocation(raw_name) != nvinfer1::TensorLocation::kDEVICE) {
      return std::unexpected(refuse(RefusalCode::FormatUnsupported, desc.name,
                                    "tenseur de forme lu sur l'hôte, non pris en charge"));
    }
    if (engine.getTensorFormat(raw_name) != nvinfer1::TensorFormat::kLINEAR) {
      return std::unexpected(
          refuse(RefusalCode::FormatUnsupported, desc.name, "format vectorisé au lieu de linéaire (NCHW)"));
    }
    if (desc.io == TensorIo::Input) {
      for (int p = 0; p < profiles; ++p) {
        auto range = profile_range(engine, desc, p);
        if (!range) return std::unexpected(range.error());
        desc.profiles.push_back(std::move(*range));
      }
    }
    tensors.push_back(std::move(desc));
  }
  return tensors;
}

}  // namespace ajp::gpu::trt
