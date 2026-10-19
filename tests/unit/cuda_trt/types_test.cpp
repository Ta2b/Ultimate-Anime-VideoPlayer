// Vocabulaire du runtime et refus CUDA : sans GPU, ces tests tournent partout
// où le module se construit.
#include "gpu/cuda/trt/types.h"

#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/cuda/trt/cuda_objects.h"
#include "gpu/cuda/trt/tool_args.h"

namespace trt = ajp::gpu::trt;
using ajp::RefusalCode;

TEST_CASE("écriture des formes à la manière de trtexec", "[cuda_trt]") {
  CHECK(trt::dims_text(std::vector<std::int64_t>{1, 3, 720, 1280}) == "1x3x720x1280");
  CHECK(trt::dims_text(std::vector<std::int64_t>{1, 3, -1, -1}) == "1x3x-1x-1");
  CHECK(trt::dims_text(std::vector<std::int64_t>{}).empty());
}

TEST_CASE("octets d'un tenseur : type, forme et débordement", "[cuda_trt]") {
  CHECK(trt::byte_count(trt::DataType::F16, std::vector<std::int64_t>{1, 3, 720, 1280}) == 5'529'600U);
  CHECK(trt::byte_count(trt::DataType::F32, std::vector<std::int64_t>{1, 11, 768, 1280}) == 43'253'760U);
  CHECK(trt::byte_count(trt::DataType::I64, std::vector<std::int64_t>{2, 0}) == 0U);
  CHECK(trt::byte_count(trt::DataType::F16, std::vector<std::int64_t>{1, 3, -1, 8}).error().code ==
        RefusalCode::ShapeInvalid);
  const std::int64_t huge = std::numeric_limits<std::int64_t>::max();
  CHECK(trt::byte_count(trt::DataType::F32, std::vector<std::int64_t>{huge, huge}).error().code ==
        RefusalCode::ShapeInvalid);
  CHECK(trt::byte_count(trt::DataType::Other, std::vector<std::int64_t>{4}).error().code ==
        RefusalCode::FormatUnsupported);
}

TEST_CASE("chaque type a un nom et une taille", "[cuda_trt]") {
  CHECK(trt::name(trt::DataType::F16) == "F16");
  CHECK(trt::name(trt::DataType::Other) == "Other");
  CHECK(trt::element_bytes(trt::DataType::BF16) == 2);
  CHECK(trt::element_bytes(trt::DataType::Bool) == 1);
  CHECK(trt::element_bytes(trt::DataType::Other) == 0);
}

TEST_CASE("une erreur CUDA devient un refus qui cite l'appel", "[cuda_trt]") {
  const ajp::Refusal refused = trt::cuda_refusal(cudaErrorMemoryAllocation, "cudaMalloc(64)");
  CHECK(refused.code == RefusalCode::CudaFailed);
  CHECK(refused.detail.starts_with("cudaMalloc(64) : cudaErrorMemoryAllocation"));
  CHECK(trt::cuda_check(cudaSuccess, "rien").has_value());
}

TEST_CASE("formes et nombres des arguments des outils", "[cuda_trt]") {
  using trt::tool::parse_dims;
  CHECK(parse_dims("1x3x720x1280") == trt::Dims{1, 3, 720, 1280});
  CHECK(parse_dims("16") == trt::Dims{16});
  for (const char* bad : {"", "x", "1x", "x1", "1xx2", "1x0", "1x-2", "1x3a", "1 x3"}) {
    INFO("« " << bad << " »");
    CHECK_FALSE(parse_dims(bad).has_value());
  }
  CHECK(trt::tool::parse_number<int>("200") == 200);
  CHECK_FALSE(trt::tool::parse_number<int>("").has_value());
  CHECK_FALSE(trt::tool::parse_number<int>("20 ").has_value());
  CHECK_FALSE(trt::tool::parse_number<int>("99999999999").has_value());
}
