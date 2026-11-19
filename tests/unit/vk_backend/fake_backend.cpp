// Bibliothèque de backend factice d'une autre version de l'interface, construite pour les tests
// et jamais rangée dans le dépôt : le chargeur doit la refuser avant kirari_backend_create.
#include <cstdint>

#if defined(_WIN32)
#define KIRARI_FAKE_EXPORT __declspec(dllexport)
#else
#define KIRARI_FAKE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" KIRARI_FAKE_EXPORT int kirari_backend_abi() noexcept {
  return 2;
}

extern "C" KIRARI_FAKE_EXPORT std::uint64_t kirari_backend_layout() noexcept {
  return 0;
}
