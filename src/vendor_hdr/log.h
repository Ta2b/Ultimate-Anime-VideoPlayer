// Canal de journal du module vendor_hdr.
#pragma once

#include <spdlog/logger.h>

namespace ajp::vendor_hdr {

// Cloné du journal par défaut au premier appel : mêmes sorties, nom « vendor_hdr ».
spdlog::logger& log();

}  // namespace ajp::vendor_hdr
