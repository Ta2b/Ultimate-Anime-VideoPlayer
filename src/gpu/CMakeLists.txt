# gpu : exécution du Plan, ring de sortie, flux de calcul. Thread gpu.
# Chaque sous-dossier (cuda/, vulkan/, metal/...) deviendra une bibliothèque
# chargée à l'exécution, installée seulement si l'utilisateur la choisit.
# ir/ est la représentation interne commune aux moteurs Vulkan et Metal.
add_subdirectory(ir)
if(AJP_WITH_CUDA)
  add_subdirectory(cuda)
endif()

ajp_add_module(gpu DEPS ajp::core ajp::kernels ajp::vendor_hdr)

# Socle de calcul Vulkan (vk_compute), sous vulkan/compute.
add_subdirectory(vulkan)
# Backends Vulkan dont les réseaux passent par le runtime du fabricant.
if(AJP_WITH_MIGRAPHX)
  add_subdirectory(migraphx)
endif()
if(AJP_WITH_OPENVINO)
  add_subdirectory(openvino)
endif()
# MIGraphX de Windows ML : le même module ajp_backend_migraphx, sous Windows, sans ROCm.
if(AJP_WITH_WINML)
  if(AJP_WITH_MIGRAPHX)
    message(FATAL_ERROR "AJP_WITH_WINML et AJP_WITH_MIGRAPHX construisent tous deux ajp_backend_migraphx")
  endif()
  add_subdirectory(winml)
endif()
# Les images de l'interpolation d'Apple (coreml/vt_pixels.h) : portables, vérifiées partout.
ajp_add_module(coreml_pixels SOURCES coreml/vt_pixels.cpp DEPS ajp::gpu ajp::kernels)
if(AJP_WITH_COREML)
  add_subdirectory(coreml)
endif()
