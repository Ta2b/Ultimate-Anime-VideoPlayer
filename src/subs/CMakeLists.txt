# subs : rendu libass en atlas bornés. Thread subs.
# libass et spdlog restent privés : les en-têtes publics les déclarent seulement.
ajp_add_module(subs
  SOURCES
    ass_library.cpp atlas_packer.cpp atlas_pool.cpp srt.cpp srt_markup.cpp
    subs_thread.cpp subs_worker.cpp subtitle_renderer.cpp subtitle_track.cpp text_encoding.cpp vsfilter_color.cpp
  DEPS ajp::core_types
  PRIVATE_DEPS ajp::dep::ass ajp::core_log)
# Les encodages hérités des sous-titres externes (P3.74) : iconv hors de Windows (dans la libc sous Linux).
if(NOT WIN32)
  find_package(Iconv REQUIRED)
  target_link_libraries(ajp_subs PRIVATE Iconv::Iconv)
endif()
