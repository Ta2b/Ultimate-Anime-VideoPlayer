// Ce que le démultiplexeur remet au thread audio. Structures propres au module :
// audio ne dépend pas de media, app traduit ses paquets dans ce format.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ajp::audio {

// Paramètres du flux, recopiés d'AVCodecParameters par celui qui démultiplexe.
struct AudioCodecInfo {
  std::string codec;               // nom de descripteur FFmpeg : "aac", "flac", "pcm_s16le"…
  int sample_rate = 0;             // 0 : lu dans le flux
  int channels = 0;                // 0 : lu dans le flux
  std::uint64_t channel_mask = 0;  // masque AV_CH_* ; 0 : disposition par défaut
  int block_align = 0;
  int bits_per_coded_sample = 0;
  std::int64_t bit_rate = 0;
  std::vector<std::uint8_t> extradata;
};

// Donnée annexe d'un paquet (AVPacketSideDataType) : amorce et remplissage de l'encodeur
// (SKIP_SAMPLES), nouveau extradata, changement de paramètres.
struct AudioSideData {
  int type = 0;
  std::vector<std::uint8_t> bytes;
};

// Un paquet compressé, horodaté en microsecondes.
struct AudioPacket {
  std::vector<std::uint8_t> data;
  std::vector<AudioSideData> side_data;
  std::optional<std::int64_t> pts_us;  // absent : suit le paquet précédent
  std::uint32_t epoch = 0;             // époque du Flush ; un paquet d'une époque passée est jeté
  // Seek exact : les échantillons antérieurs à cet instant sont jetés après décodage.
  std::optional<std::int64_t> discard_before_us;
};

// Ouvre le décodeur d'une piste : au premier flux, puis à chaque changement de piste.
struct AudioTrack {
  AudioCodecInfo codec;
};

// Fin de la piste : le décodeur rend ses dernières trames, puis l'horloge
// continue sans l'audio au lieu de s'arrêter faute d'échantillons.
struct EndOfStream {
  std::uint32_t epoch = 0;
};

}  // namespace ajp::audio
