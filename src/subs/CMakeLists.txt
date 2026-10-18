# subs : rendu libass en atlas bornés. Thread subs.
# libass et spdlog restent privés : les en-têtes publics les déclarent seulement.
ajp_add_module(subs
  SOURCES
    ass_library.cpp atlas_packer.cpp atlas_pool.cpp log.cpp srt.cpp srt_markup.cpp
    subs_thread.cpp subs_worker.cpp subtitle_renderer.cpp subtitle_track.cpp
  DEPS ajp::core_types
  PRIVATE_DEPS ajp::dep::ass ajp::dep::spdlog)
