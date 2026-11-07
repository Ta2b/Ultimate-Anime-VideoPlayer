// Un fichier projeté en mémoire, en lecture et copie à l'écriture (P4.19) : une police CJK de 20 à 60 Mo
// n'est plus lue entière à chaque lancement, FreeType ne touche que les pages des glyphes demandés.
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

namespace ajp::app::play {

class MappedFile {
 public:
  // Rien si le fichier ne s'ouvre pas, est vide ou dépasse max_bytes.
  static std::shared_ptr<MappedFile> open(const std::filesystem::path& path, std::size_t max_bytes);
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile();

  unsigned char* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  MappedFile() = default;
  unsigned char* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace ajp::app::play
