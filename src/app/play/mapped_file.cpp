#include "app/play/mapped_file.h"

#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ajp::app::play {

// La taille vient du fichier ouvert, pas du chemin (A18-14) : une police remplacée ou tronquée entre les deux
// ne laisse pas une taille plus grande que la projection.
std::shared_ptr<MappedFile> MappedFile::open(const std::filesystem::path& path, std::size_t max_bytes) {
  std::shared_ptr<MappedFile> mapped(new MappedFile());
#ifdef _WIN32
  HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) return nullptr;
  LARGE_INTEGER bytes{};
  if (!GetFileSizeEx(file, &bytes) || bytes.QuadPart <= 0 ||
      static_cast<std::uint64_t>(bytes.QuadPart) > max_bytes) {
    CloseHandle(file);
    return nullptr;
  }
  const auto size = static_cast<std::size_t>(bytes.QuadPart);
  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_WRITECOPY, 0, 0, nullptr);
  CloseHandle(file);
  if (mapping == nullptr) return nullptr;
  void* view = MapViewOfFile(mapping, FILE_MAP_COPY, 0, 0, size);
  CloseHandle(mapping);  // la vue garde la projection
  if (view == nullptr) return nullptr;
#else
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return nullptr;
  struct stat facts{};
  if (fstat(fd, &facts) != 0 || facts.st_size <= 0 || static_cast<std::uint64_t>(facts.st_size) > max_bytes) {
    ::close(fd);
    return nullptr;
  }
  const auto size = static_cast<std::size_t>(facts.st_size);
  void* view = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ::close(fd);  // la projection garde le fichier
  if (view == MAP_FAILED) return nullptr;
#endif
  mapped->data_ = static_cast<unsigned char*>(view);
  mapped->size_ = size;
  return mapped;
}

MappedFile::~MappedFile() {
  if (data_ == nullptr) return;
#ifdef _WIN32
  UnmapViewOfFile(data_);
#else
  munmap(data_, size_);
#endif
}

}  // namespace ajp::app::play
