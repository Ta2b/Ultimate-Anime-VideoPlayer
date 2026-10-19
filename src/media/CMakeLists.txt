# media : démultiplexage, décodage matériel, horodatage. Thread source.
# FFmpeg reste hors des en-têtes d'interface (source, demuxer, video_decoder,
# packet, track, selection, hw_api) : seuls les en-têtes internes et les .cpp l'incluent.
ajp_add_module(media
  SOURCES
    av_util.cpp color.cpp cuda_driver.cpp demuxer.cpp frame_data.cpp hw_device.cpp
    language.cpp packet.cpp selection.cpp source.cpp source_loop.cpp
    track.cpp video_decoder.cpp
  DEPS ajp::core_types
  PRIVATE_DEPS ajp::dep::ffmpeg ajp::core_log ajp::dep::vulkan ${CMAKE_DL_LIBS})
