// Bibliothèque de backend factice d'une autre version de l'interface, construite pour les tests
// et jamais rangée dans le dépôt : le chargeur doit la refuser avant ajp_backend_create.
#include <cstdint>

#if defined(_WIN32)
#define AJP_FAKE_EXPORT __declspec(dllexport)
#else
#define AJP_FAKE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" AJP_FAKE_EXPORT int ajp_backend_abi() noexcept {
  return 2;
}

extern "C" AJP_FAKE_EXPORT std::uint64_t ajp_backend_layout() noexcept {
  return 0;
}
