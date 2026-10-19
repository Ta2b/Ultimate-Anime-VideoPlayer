# audio : décodage audio, sortie SDL3, rééchantillonnage, horloge. Thread audio.
# FFmpeg, SDL3 et spdlog restent privés : les en-têtes publics n'en exposent rien.
ajp_add_module(audio
  SOURCES audio_thread.cpp converter.cpp decoder.cpp drift.cpp output.cpp pcm.cpp player.cpp
          sdl_sink.cpp timeline.cpp
  DEPS ajp::core_types
  PRIVATE_DEPS ajp::dep::ffmpeg ajp::dep::sdl3 ajp::core_log)
