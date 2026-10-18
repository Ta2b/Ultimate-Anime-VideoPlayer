# vendor_hdr : interface TrueHdr, RTX Video SDK et Maxine VFX. Sans thread, appelé
# directement par gpu. Les SDK sont importés par l'utilisateur et ouverts à l'exécution :
# rien n'est lié à NVIDIA ici, seulement libdl.
ajp_add_module(vendor_hdr
  SOURCES
    binary_probe.cpp
    blockers.cpp
    dynamic_library.cpp
    log.cpp
    maxine_abi_check.cpp
    maxine_vfx.cpp
    open_true_hdr.cpp
    rtx_video.cpp
    sdk_copy.cpp
    sdk_layout.cpp
    sdk_manifest.cpp
    sdk_store.cpp
    sdk_version.cpp
    true_hdr.cpp
    true_hdr_stage.cpp
  DEPS ajp::core_types
  PRIVATE_DEPS ajp::dep::spdlog ajp::dep::toml ${CMAKE_DL_LIBS})
