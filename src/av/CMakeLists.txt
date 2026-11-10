# av : le texte d'un code AVERROR de FFmpeg (av_error.h), en-tête seul. Hors de core, qui ne lie pas FFmpeg
# (A20-12) ; media, audio, export et app, qui le lient déjà, le prennent d'ici.
ajp_add_module(av DEPS ajp::dep::ffmpeg)
