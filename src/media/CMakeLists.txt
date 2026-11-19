# media : démultiplexage, décodage matériel, horodatage. Thread source.
# FFmpeg reste hors des en-têtes d'interface (source, demuxer, video_decoder,
# packet, track, selection, hw_api) : seuls les en-têtes internes et les .cpp l'incluent.
kirari_add_module(media
  SOURCES
    av_util.cpp face_cascade.cpp face_skin.cpp faces.cpp field_alignment.cpp "${CMAKE_CURRENT_BINARY_DIR}/face_cascade_data.cpp" ffmpeg_log.cpp bench_clip.cpp color.cpp live_scenes.cpp probe_flow.cpp probe_flow_nvof.cpp shot_cache.cpp shot_cutter.cpp shot_rhythm.cpp shot_measure.cpp shot_probe.cpp shot_timeline.cpp stills.cpp cuda_driver.cpp demuxer.cpp frame_data.cpp hw_device.cpp
    language.cpp packet.cpp packet_reserve.cpp selection.cpp source.cpp source_loop.cpp
    tag_text.cpp track.cpp track_language.cpp video_decoder.cpp deinterlace.cpp mkv_chapters.cpp telecine.cpp
  DEPS kirari::core_types kirari::kernels
  PRIVATE_DEPS kirari::av kirari::dep::ffmpeg kirari::core kirari::core_log kirari::core_os kirari::dep::vulkan ${CMAKE_DL_LIBS})

# Le flux optique de la pré-analyse (probe_flow_nvof.cpp) : les en-têtes BSD de NVIDIA, et le <cuda.h> réduit
# de nvof_cuda/ à la place du toolkit, pour ce fichier seulement.
set_source_files_properties(probe_flow_nvof.cpp PROPERTIES INCLUDE_DIRECTORIES
  "${CMAKE_CURRENT_SOURCE_DIR}/nvof_cuda;${CMAKE_SOURCE_DIR}/third_party/nvidia-optical-flow")

# La cascade de visages (media/face_cascade.h), embarquée octet pour octet depuis sa copie intacte.
set(face_cascade_xml "${CMAKE_SOURCE_DIR}/third_party/lbpcascade_animeface/lbpcascade_animeface.xml")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/face_cascade_data.cpp"
  COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/face_cascade_data.cpp"
          "-DINPUT=${face_cascade_xml}" -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_cascade.cmake"
  DEPENDS "${face_cascade_xml}" "${CMAKE_CURRENT_SOURCE_DIR}/embed_cascade.cmake"
  COMMENT "Embarque la cascade de visages"
  VERBATIM)

if(WIN32)
  target_link_libraries(kirari_media PRIVATE dxgi)
endif()

if(TARGET kirari::dep::va)
  target_sources(kirari_media PRIVATE vaapi_device.cpp)
  target_link_libraries(kirari_media PRIVATE kirari::dep::va)
  target_compile_definitions(kirari_media PRIVATE KIRARI_HAVE_VA=1)
endif()
