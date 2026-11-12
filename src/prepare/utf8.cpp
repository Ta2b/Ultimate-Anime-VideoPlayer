#include "prepare/utf8.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "core/utf8_path.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <system_error>
#include <vector>
#include <windows.h>
#endif

namespace ajp::prepare {

std::string utf8_path(const std::filesystem::path& path) {
  return core::to_utf8(path);
}

std::filesystem::path path_from_utf8(std::string_view text) {
  return core::path_from_utf8(text);
}

#ifdef _WIN32
namespace {

// La page de code ANSI du SYSTÈME, celle que relisent les programmes sans manifeste UTF-8 (trtexec, curl,
// tar). Les exécutables d'ajp portent activeCodePage=UTF-8 : pour eux CP_ACP vaut 65001, quelle que soit la
// page du système (B34-01). 0 quand elle ne se lit pas.
UINT system_ansi_code_page() {
  DWORD page = 0;
  const int written =
      GetLocaleInfoEx(LOCALE_NAME_SYSTEM_DEFAULT, LOCALE_IDEFAULTANSICODEPAGE | LOCALE_RETURN_NUMBER,
                      reinterpret_cast<LPWSTR>(&page), static_cast<int>(sizeof(page) / sizeof(wchar_t)));
  return written > 0 ? static_cast<UINT>(page) : 0;
}

// Le texte s'écrit-il dans la page ANSI du système sans caractère de remplacement ?
bool ansi_representable(const std::wstring& text) {
  if (text.empty()) return true;
  UINT page = system_ansi_code_page();
  if (page == CP_UTF8)
    return true;  // « Utiliser Unicode UTF-8 » : tout s'écrit, et les drapeaux ci-dessous sont interdits
  if (page == 0) page = CP_ACP;  // illisible : la page du processus, comme avant
  if (page == CP_UTF8) return true;
  BOOL replaced = FALSE;
  const int size = WideCharToMultiByte(page, WC_NO_BEST_FIT_CHARS, text.c_str(),
                                       static_cast<int>(text.size()), nullptr, 0, nullptr, &replaced);
  return size > 0 && replaced == FALSE;
}

std::wstring short_name(const std::wstring& existing) {
  const DWORD size = GetShortPathNameW(existing.c_str(), nullptr, 0);
  if (size == 0) return {};
  std::vector<wchar_t> out(size);
  const DWORD written = GetShortPathNameW(existing.c_str(), out.data(), size);
  if (written == 0 || written >= size) return {};
  return {out.data(), written};
}

}  // namespace
#endif

std::optional<std::string> ansi_command_path(const std::filesystem::path& path) {
#ifdef _WIN32
  if (ansi_representable(path.native())) return core::to_utf8(path);
  // La plus longue partie qui existe prend son nom court ; ce qui reste (le fichier à écrire) est ajouté.
  std::filesystem::path existing = path;
  std::filesystem::path rest;
  std::error_code error;
  while (!existing.empty() && !std::filesystem::exists(existing, error)) {
    rest = rest.empty() ? existing.filename() : existing.filename() / rest;
    if (existing == existing.parent_path()) break;
    existing = existing.parent_path();
  }
  const std::wstring shortened = existing.empty() ? std::wstring() : short_name(existing.native());
  if (shortened.empty()) return std::nullopt;
  const std::filesystem::path candidate =
      rest.empty() ? std::filesystem::path(shortened) : std::filesystem::path(shortened) / rest;
  if (!ansi_representable(candidate.native())) return std::nullopt;
  return core::to_utf8(candidate);
#else
  return core::to_utf8(path);
#endif
}

std::string ansi_safe_path(const std::filesystem::path& path) {
  if (auto safe = ansi_command_path(path)) return std::move(*safe);
  return core::to_utf8(path);
}

bool is_utf8(std::string_view text) {
  std::size_t i = 0;
  while (i < text.size()) {
    const auto lead = static_cast<std::uint8_t>(text[i]);
    std::size_t length = 1;
    std::uint32_t point = lead;
    if (lead >= 0xF0 && lead <= 0xF4) {
      length = 4;
      point = lead & 0x07U;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      length = 3;
      point = lead & 0x0FU;
    } else if (lead >= 0xC2 && lead <= 0xDF) {
      length = 2;
      point = lead & 0x1FU;
    } else if (lead >= 0x80) {
      return false;  // octet de suite isolé, C0, C1 ou F5 à FF
    }
    if (text.size() - i < length) return false;
    for (std::size_t k = 1; k < length; ++k) {
      const auto next = static_cast<std::uint8_t>(text[i + k]);
      if ((next & 0xC0U) != 0x80U) return false;
      point = (point << 6U) | (next & 0x3FU);
    }
    const bool overlong = (length == 3 && point < 0x800) || (length == 4 && point < 0x10000);
    const bool surrogate = point >= 0xD800 && point <= 0xDFFF;
    if (overlong || surrogate || point > 0x10FFFF) return false;
    i += length;
  }
  return true;
}

}  // namespace ajp::prepare
