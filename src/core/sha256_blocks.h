// Le cœur de SHA-256 (FIPS 180-4) : la compression de blocs de 64 octets, commune à core::Sha256 et
// prepare::Sha256 (P4.10). Les instructions SHA des x86 quand le processeur les a (cpuid), sinon le code
// portable.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ajp::core {

void sha256_blocks(std::array<std::uint32_t, 8>& state, const std::uint8_t* data, std::size_t count);

}  // namespace ajp::core
