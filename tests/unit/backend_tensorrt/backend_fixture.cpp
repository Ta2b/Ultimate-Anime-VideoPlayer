#include "backend_fixture.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ajp::test {

#if defined(_WIN32)
void* open_backend_library() {
  return LoadLibraryA(AJP_BACKEND_TENSORRT);
}

void close_backend_library(void* handle) {
  if (handle != nullptr) FreeLibrary(static_cast<HMODULE>(handle));
}

void* backend_symbol(void* handle, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
}
#else
void* open_backend_library() {
  return dlopen(AJP_BACKEND_TENSORRT, RTLD_NOW | RTLD_LOCAL);
}

void close_backend_library(void* handle) {
  if (handle != nullptr) dlclose(handle);
}

void* backend_symbol(void* handle, const char* name) {
  return dlsym(handle, name);
}
#endif

}  // namespace ajp::test
