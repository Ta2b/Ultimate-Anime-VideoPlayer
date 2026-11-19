# audio : décodage audio, sortie SDL3, rééchantillonnage, horloge. Thread audio.
# FFmpeg, SDL3 et spdlog restent privés : les en-têtes publics n'en exposent rien.
kirari_add_module(audio
  SOURCES audio_thread.cpp devices.cpp converter.cpp decoder.cpp drift.cpp output.cpp pcm.cpp player.cpp
          sdl_sink.cpp shaper.cpp timeline.cpp
  DEPS kirari::core_types
  PRIVATE_DEPS kirari::av kirari::dep::ffmpeg kirari::dep::sdl3 kirari::core_log kirari::core_os)
