// Lecture et écriture des fichiers bruts du banc DistilDRBA (drba_fusion_bench.cu).
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

inline std::vector<char> lire_fichier(const std::string& chemin) {
  std::ifstream f(chemin, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "fichier illisible : %s\n", chemin.c_str());
    std::exit(1);
  }
  const auto n = static_cast<std::size_t>(f.tellg());
  std::vector<char> d(n);
  f.seekg(0);
  f.read(d.data(), static_cast<std::streamsize>(n));
  return d;
}

// Un fichier d'entrée de la taille exacte attendue : trop court, le téléversement lirait au-delà ; trop long,
// il ignorerait des données en silence (C46-24).
inline std::vector<char> lire_taille(const std::string& chemin, std::size_t octets) {
  auto d = lire_fichier(chemin);
  if (d.size() != octets) {
    std::fprintf(stderr, "%s : %zu octets, %zu attendus\n", chemin.c_str(), d.size(), octets);
    std::exit(1);
  }
  return d;
}

inline void ecrire_fichier(const std::string& chemin, const void* data, std::size_t n) {
  std::ofstream f(chemin, std::ios::binary);
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
}
