// C35-01 : les seuls conteneurs et décodeurs de FFmpeg qu'une vidéo ouverte par le lecteur peut atteindre. Le
// fichier, l'entrée la plus exposée (épisode tiré d'un torrent), était sondé contre tous les démultiplexeurs et
// décodeurs de FFmpeg, y compris ceux de formats qu'aucun anime n'utilise et où tombent la plupart des failles,
// et « concat », qui ouvre d'autres fichiers du dossier. Listes passées à FFmpeg (format_whitelist,
// codec_whitelist) : un nom absent fait refuser le fichier ou la piste, comme un format inconnu.
//
// Ajouter un nom ici, c'est ouvrir ce démultiplexeur ou ce décodeur à tout fichier : seulement pour un format
// que de vrais épisodes portent.
#pragma once

namespace ajp::av {

// Les noms de démultiplexeurs (AVInputFormat::name, dont chaque nom d'une liste comme « matroska,webm »).
inline constexpr const char* kAllowedDemuxers =
    "matroska,webm,mov,mp4,m4a,3gp,avi,mpegts,mpeg,flv,ogg,asf,ivf,obu,h264,hevc,m4v";

// Les noms de décodeurs (AVCodec::name) : vidéo (avec leurs variantes NVDEC « _cuvid » et la pochette jointe),
// son, sous-titres.
inline constexpr const char* kAllowedDecoders =
    "h264,hevc,av1,libdav1d,vp9,vp8,mpeg1video,mpeg2video,mpeg4,msmpeg4,msmpeg4v2,h263,vc1,wmv3,ffv1,prores,"
    "h264_cuvid,hevc_cuvid,av1_cuvid,vp9_cuvid,vp8_cuvid,mpeg1_cuvid,mpeg2_cuvid,mpeg4_cuvid,vc1_cuvid,"
    "mjpeg,png,"
    "aac,aac_fixed,aac_latm,ac3,ac3_fixed,eac3,dca,truehd,mlp,flac,opus,libopus,vorbis,mp3,mp3float,mp2,"
    "mp2float,mp1,mp1float,alac,wmav1,wmav2,wmapro,pcm_s16le,pcm_s16be,pcm_s24le,pcm_s24be,pcm_s32le,pcm_s32be,"
    "pcm_f32le,pcm_f32be,pcm_f64le,pcm_u8,pcm_bluray,pcm_dvd,"
    "ass,ssa,subrip,srt,text,webvtt,mov_text,hdmv_pgs_subtitle,dvdsub,dvbsub";

}  // namespace ajp::av
