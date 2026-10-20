# export : encodage matériel, mux MKV avec les pistes de la source copiées. Thread export.
# FFmpeg reste hors de l'en-tête d'interface (export_writer.h, encoder_choice.h).
ajp_add_module(export
  SOURCES av_handles.cpp encoder_choice.cpp export_writer.cpp muxer.cpp video_encoder.cpp
  DEPS ajp::core_types
  PRIVATE_DEPS ajp::dep::ffmpeg ajp::dep::vulkan ajp::core_log)
