// framemd5 : une ligne par image (flux, dts, pts, durée, taille, MD5 des pixels), le texte
// même du muxer framemd5 de libavformat, comme « ffmpeg -i <fichier> -f framemd5 - ».
//
// La preuve de déterminisme du plan : deux exports identiques donnent le même texte, et une
// lecture sans pertes vidée par ajp bench --dump se compare à l'export par lui.
#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <string>

#include "core/types.h"
#include "export/frame_source.h"

namespace ajp::exporter {

// Décode le meilleur flux vidéo du fichier et hache chaque image décodée, plans tassés sans
// marge de ligne. Refus MediaUnreadable, DecoderUnavailable, DecodeFailed.
std::expected<std::string, Refusal> framemd5_of_file(const std::filesystem::path& path);

// Le même texte pour des images RGBA16 hôte (rawvideo rgba64, base de temps la microseconde).
class FrameMd5 {
 public:
  // Refus FieldOutOfRange pour une taille nulle ou négative.
  static std::expected<FrameMd5, Refusal> create(int width, int height);

  FrameMd5(FrameMd5&&) noexcept;
  FrameMd5& operator=(FrameMd5&&) noexcept;
  ~FrameMd5();

  // Refus InputMismatch : autre taille, tampon trop court, pts inconnu.
  std::expected<void, Refusal> add(const HostFrame& frame);

  // Le texte complet ; l'objet ne sert plus ensuite.
  std::expected<std::string, Refusal> finish();

 private:
  struct State;
  explicit FrameMd5(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

}  // namespace ajp::exporter
