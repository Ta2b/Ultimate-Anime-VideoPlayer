// Les entrées d'un dossier, lues sans exception. Une boucle « for (… : directory_iterator(dir, error)) » lève
// quand l'incrément échoue (dossier qui disparaît pendant le parcours, droits) : l'error_code ne vaut que
// pour la construction. Ici l'incrément prend l'error_code, et une erreur arrête la liste en la gardant dans
// error.
#pragma once

#include <filesystem>
#include <system_error>
#include <vector>

namespace ajp::core {

template <class Iterator>
std::vector<std::filesystem::directory_entry> entries_of(Iterator it, std::error_code& error) {
  std::vector<std::filesystem::directory_entry> entries;
  for (; !error && it != Iterator(); it.increment(error)) entries.push_back(*it);
  return entries;
}

inline std::vector<std::filesystem::directory_entry> dir_entries(const std::filesystem::path& dir,
                                                                 std::error_code& error) {
  return entries_of(std::filesystem::directory_iterator(dir, error), error);
}

inline std::vector<std::filesystem::directory_entry> tree_entries(
    const std::filesystem::path& dir, std::error_code& error,
    std::filesystem::directory_options options = std::filesystem::directory_options::skip_permission_denied) {
  return entries_of(std::filesystem::recursive_directory_iterator(dir, options, error), error);
}

}  // namespace ajp::core
