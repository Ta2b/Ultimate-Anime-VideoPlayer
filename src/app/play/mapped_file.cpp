#include "app/play/mapped_file.h"

#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace ajp::app::play {

std::shared_ptr<MappedFile> MappedFile::open(const std::filesystem::path& path, std::size_t max_bytes) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size == 0 || size > max_bytes) return nullptr;
  std::shared_ptr<MappedFile> mapped(new MappedFile());
#ifdef _WIN32
  HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) return nullptr;
  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_WRITECOPY, 0, 0, nullptr);
  CloseHandle(file);
  if (mapping == nullptr) return nullptr;
  void* view = MapViewOfFile(mapping, FILE_MAP_COPY, 0, 0, 0);
  CloseHandle(mapping);  // la vue garde la projection
  if (view == nullptr) return nullptr;
#else
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return nullptr;
  void* view = mmap(nullptr, static_cast<std::size_t>(size), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ::close(fd);  // la projection garde le fichier
  if (view == MAP_FAILED) return nullptr;
#endif
  mapped->data_ = static_cast<unsigned char*>(view);
  mapped->size_ = static_cast<std::size_t>(size);
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
