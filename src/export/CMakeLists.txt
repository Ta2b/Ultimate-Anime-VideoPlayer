# export : encodage, mux, métadonnées HDR10. Thread export.
# FFmpeg reste hors des en-têtes publics (export, export_thread, frame_source, encoder_settings,
# hdr10, progress, framemd5) : seuls les en-têtes internes et les .cpp l'incluent.
ajp_add_module(export
  SOURCES
    av_util.cpp encoder.cpp encoder_settings.cpp export_thread.cpp framemd5.cpp hdr10.cpp mux.cpp
    progress.cpp remux.cpp run_export.cpp side_data.cpp
  DEPS ajp::core ajp::render
  PRIVATE_DEPS ajp::dep::ffmpeg ajp::core_log)
