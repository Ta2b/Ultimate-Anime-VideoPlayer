// Comparaison des constantes de maxine_abi.h aux en-têtes du SDK importé.
#include <cstddef>
#include <format>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "support.h"
#include "vendor_hdr/maxine_abi.h"
#include "vendor_hdr/maxine_abi_check.h"

using ajp::RefusalCode;
using namespace ajp::vendor_hdr;
using namespace ajp::vendor_hdr::test;

namespace {

// Remplace la ligne « #define name … » de abi_header_text().
std::string with_define(std::string_view name, std::string_view value) {
  std::string text = abi_header_text();
  const std::string head = std::format("#define {} ", name);
  const std::size_t at = text.find(head);
  REQUIRE(at != std::string::npos);
  const std::size_t end = text.find('\n', at);
  text.replace(at, end - at, head + std::string(value));
  return text;
}

std::string without_define(std::string_view name) {
  std::string text = abi_header_text();
  const std::size_t at = text.find(std::format("#define {} ", name));
  REQUIRE(at != std::string::npos);
  text.erase(at, text.find('\n', at) - at + 1);
  return text;
}

}  // namespace

TEST_CASE("ABI : #define, alias et commentaires", "[abi]") {
  const std::vector<std::string> headers{
      "#define NVCV_GPU 1 /* NVCV_CUDA */\n#define NVCV_CUDA NVCV_GPU\n",
      "// #define NVCV_U8 99\n#define NVCV_U8   1   // octet\n#define NVVFX_FX_TRUE_HDR \"TrueHDR\"\n",
      "#define NVVFX_INPUT_IMAGE NVVFX_INPUT_IMAGE_0\n#define NVVFX_INPUT_IMAGE_0 \"SrcImage0\"\n",
  };
  CHECK(find_definition(headers, "NVCV_GPU") == "1");
  CHECK(find_definition(headers, "NVCV_CUDA") == "1");
  CHECK(find_definition(headers, "NVCV_U8") == "1");
  CHECK(find_definition(headers, "NVVFX_FX_TRUE_HDR") == "\"TrueHDR\"");
  CHECK(find_definition(headers, "NVVFX_INPUT_IMAGE") == "\"SrcImage0\"");
  CHECK_FALSE(find_definition(headers, "NVCV_RGBA"));
}

TEST_CASE("ABI : énumérateurs explicites et implicites", "[abi]") {
  const std::vector<std::string> headers{
      "typedef enum {\n  NVCV_FORMAT_UNKNOWN = 0,\n  NVCV_Y = 1, NVCV_A, NVCV_YA,\n  NVCV_RGB, NVCV_BGR,\n"
      "  NVCV_RGBA, // 6\n  NVCV_BGRA,\n  NVCV_YUV420 = 0x0A,\n  NVCV_YUV422,\n  NVCV_YUV444,\n"
      "  NVCV_RGB10A2 = (NVCV_YUV444 + 1)\n} NvCVImage_PixelFormat;\n",
      "struct NotAnEnum { int NVCV_P32; };\n"
      "enum NvCV_Status { NVCV_SUCCESS = 0, NVCV_ERR_GENERAL = -1, NVCV_ERR_UNIMPLEMENTED = -2 };\n",
  };
  CHECK(find_definition(headers, "NVCV_A") == "2");
  CHECK(find_definition(headers, "NVCV_RGBA") == "6");
  CHECK(find_definition(headers, "NVCV_YUV444") == "12");
  CHECK(find_definition(headers, "NVCV_ERR_UNIMPLEMENTED") == "-2");
  // Une expression n'est pas évaluée : pas de valeur plutôt qu'une fausse.
  CHECK_FALSE(find_definition(headers, "NVCV_RGB10A2"));
  // Un membre de structure n'est pas un énumérateur.
  CHECK_FALSE(find_definition(headers, "NVCV_P32"));
}

TEST_CASE("ABI : les en-têtes attendus passent", "[abi]") {
  const std::vector<std::string> headers{abi_header_text()};
  CHECK(check_maxine_abi(headers));
}

TEST_CASE("ABI : une constante différente est nommée", "[abi]") {
  const std::vector<std::string> pixel{with_define("NVCV_RGB10A2", "14")};
  auto refused = check_maxine_abi(pixel);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::SdkRejected);
  CHECK(refused.error().detail ==
        "NVCV_RGB10A2 vaut 14 dans les en-têtes du SDK, ce lecteur attend 13 (maxine_abi.h)");

  // L'ancienne supposition de la vague 1, démentie par nvVFXTrueHDR.h:20.
  const std::vector<std::string> selector{with_define("NVVFXTRUEHDR_LUMINANCE", "\"MaxLuminance\"")};
  CHECK(check_maxine_abi(selector).error().detail ==
        "NVVFXTRUEHDR_LUMINANCE vaut \"MaxLuminance\" dans les en-têtes du SDK, ce lecteur attend "
        "\"Luminance\" (maxine_abi.h)");
}

TEST_CASE("ABI : une constante absente est refusée", "[abi]") {
  const std::vector<std::string> headers{without_define("NVVFX_CUDA_STREAM")};
  CHECK(check_maxine_abi(headers).error().detail ==
        "NVVFX_CUDA_STREAM absent des en-têtes du SDK : SDK d'une autre version ?");
}

TEST_CASE("ABI : en-têtes lus dans les dossiers du SDK", "[abi]") {
  TempDir dir;
  CHECK(check_maxine_abi(dir.path()).error().code == RefusalCode::SdkRejected);

  // Les constantes de la fonction TrueHDR vivent dans son propre dossier include/.
  std::string core = abi_header_text();
  std::string feature = "#pragma once\n";
  for (const char* name : {"NVVFXTRUEHDR_CONTRAST", "NVVFXTRUEHDR_SATURATION", "NVVFXTRUEHDR_MIDDLE_GRAY",
                           "NVVFXTRUEHDR_LUMINANCE"}) {
    const std::size_t at = core.find(std::format("#define {} ", name));
    const std::size_t end = core.find('\n', at) + 1;
    feature += core.substr(at, end - at);
    core.erase(at, end - at);
  }
  write_file(dir.path() / "include/nvVideoEffects.h", core);
  write_file(dir.path() / "include/README.txt", "#define NVCV_RGBA 99\n");  // pas un .h : ignoré
  CHECK_FALSE(check_maxine_abi(dir.path()));
  write_file(dir.path() / "features/nvvfxtruehdr/include/nvVFXTrueHDR.h", feature);
  CHECK(check_maxine_abi(dir.path()));
}

TEST_CASE("ABI : la structure NvCVImage tient en 64 octets", "[abi]") {
  CHECK(sizeof(maxine::NvCVImage) == 64);
  CHECK(offsetof(maxine::NvCVImage, pixel_bytes) == 20);
  CHECK(offsetof(maxine::NvCVImage, pixels) == 32);
  CHECK(offsetof(maxine::NvCVImage, buffer_bytes) == 56);
}
