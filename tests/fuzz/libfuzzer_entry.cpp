// Le point d'entrée libFuzzer d'une cible (KIRARI_FUZZ_TARGET, posé par tests/fuzz/CMakeLists.txt).
#include <cstddef>
#include <cstdint>

#include "fuzz_targets.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  return kirari::fuzz::KIRARI_FUZZ_TARGET(data, size);
}
