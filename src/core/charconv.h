// std::from_chars, flottants compris là où la bibliothèque standard ne les lit pas encore (libc++
// de Xcode 16, sans __cpp_lib_to_chars) : repli par un flux en locale « C », sans dépendre de la
// locale du processus. Mêmes règles qu'ailleurs pour l'appelant : il vérifie ec et ptr.
#pragma once

#include <charconv>
#include <cstddef>
#include <ios>
#include <locale>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>

namespace ajp::core {

template <class T>
std::from_chars_result from_chars(const char* first, const char* last, T& value) {
#if defined(__cpp_lib_to_chars)
  return std::from_chars(first, last, value);
#else
  if constexpr (!std::is_floating_point_v<T>) {
    return std::from_chars(first, last, value);
  } else {
    // Comme std::from_chars : ni blanc ni « + » en tête.
    if (first == last || *first == '+' || *first == ' ' || *first == '\t') {
      return {first, std::errc::invalid_argument};
    }
    std::istringstream in(std::string(first, last));
    in.imbue(std::locale::classic());
    T parsed{};
    in >> std::noskipws >> parsed;
    if (in.fail()) return {first, std::errc::invalid_argument};
    const std::ptrdiff_t read = in.eof() ? last - first : static_cast<std::ptrdiff_t>(in.tellg());
    value = parsed;
    return {first + read, std::errc{}};
  }
#endif
}

}  // namespace ajp::core
