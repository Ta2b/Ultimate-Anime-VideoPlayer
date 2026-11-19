# export : encodage matériel, mux MKV, MP4 ou MOV avec les pistes de la source copiées. Thread export.
# FFmpeg reste hors de l'en-tête d'interface (export_writer.h, encoder_choice.h).
kirari_add_module(export
  SOURCES av_handles.cpp container.cpp encoder_choice.cpp export_writer.cpp muxer.cpp video_encoder.cpp
  DEPS kirari::core_types kirari::kernels
  PRIVATE_DEPS kirari::av kirari::dep::ffmpeg kirari::dep::vulkan kirari::core_log)
