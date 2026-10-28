# media : démultiplexage, décodage matériel, horodatage. Thread source.
# FFmpeg reste hors des en-têtes d'interface (source, demuxer, video_decoder,
# packet, track, selection, hw_api) : seuls les en-têtes internes et les .cpp l'incluent.
ajp_add_module(media
  SOURCES
    av_util.cpp bench_clip.cpp color.cpp stills.cpp cuda_driver.cpp demuxer.cpp frame_data.cpp hw_device.cpp
    language.cpp packet.cpp packet_reserve.cpp selection.cpp source.cpp source_loop.cpp
    track.cpp track_language.cpp video_decoder.cpp
  DEPS ajp::core_types
  PRIVATE_DEPS ajp::dep::ffmpeg ajp::core_log ajp::dep::vulkan ${CMAKE_DL_LIBS})

if(WIN32)
  target_link_libraries(ajp_media PRIVATE dxgi)
endif()

if(TARGET ajp::dep::va)
  target_sources(ajp_media PRIVATE vaapi_device.cpp)
  target_link_libraries(ajp_media PRIVATE ajp::dep::va)
  target_compile_definitions(ajp_media PRIVATE AJP_HAVE_VA=1)
endif()
