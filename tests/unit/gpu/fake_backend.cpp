// Bibliothèques de backend factices pour les tests du chargeur, d'une seule source :
//   AJP_FAKE_OLD_ABI : ajp_backend_abi() rend 2 ;
//   AJP_FAKE_BAD_LAYOUT : la bonne version, une autre empreinte binaire ;
//   AJP_FAKE_NO_CREATE : ni ajp_backend_create ni ajp_backend_destroy ;
//   sinon : ajp_backend_create refuse l'appareil de rendu (DeviceMismatch).
#include <cstdint>

#include "gpu/backend.h"

extern "C" int ajp_backend_abi() noexcept {
#ifdef AJP_FAKE_OLD_ABI
  return 2;
#else
  return ajp::gpu::kBackendAbi;
#endif
}

extern "C" std::uint64_t ajp_backend_layout() noexcept {
#ifdef AJP_FAKE_BAD_LAYOUT
  return ajp::gpu::kBackendLayout ^ 1U;
#else
  return ajp::gpu::kBackendLayout;
#endif
}

#ifndef AJP_FAKE_NO_CREATE
extern "C" ajp::gpu::Backend* ajp_backend_create(int, const ajp::gpu::BackendInit* init) noexcept {
  if (init != nullptr && init->refusal != nullptr) {
    *init->refusal = {ajp::RefusalCode::DeviceMismatch, "backend factice : appareil de rendu refusé"};
  }
  return nullptr;
}

extern "C" void ajp_backend_destroy(ajp::gpu::Backend* backend) noexcept {  // NOLINT(misc-const-correctness)
  delete backend;
}
#endif
