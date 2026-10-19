// Compte les allocations par operator new du thread courant pendant la vie d'une sonde : un
// test vérifie ainsi qu'un appel du chemin de lecture n'alloue rien sur l'hôte. Le binaire de
// test remplace operator new et delete (host_allocations.cpp) ; hors d'une sonde, ils ne font
// qu'appeler malloc et free.
#pragma once

#include <cstdint>

namespace ajp::gpu::vk::test {

class HostAllocationProbe {
 public:
  HostAllocationProbe();
  HostAllocationProbe(const HostAllocationProbe&) = delete;
  HostAllocationProbe& operator=(const HostAllocationProbe&) = delete;
  ~HostAllocationProbe();

  std::uint64_t count() const;  // allocations depuis la création de la sonde
};

}  // namespace ajp::gpu::vk::test
