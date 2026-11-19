// Garder la sortie d'un encodeur d'une passe à l'autre (TemporalFix, app/play/temporal_fix.h) : le graphe
// encode chacune de ses sept images (un lot de sept, convolutions image par image) avant de les aligner.
// Une image lue par sept passes de suite était encodée sept fois. La découpe en fait deux modèles :
//
// - l'encodeur, une image 1 × 3 × H × W → ses traits 1 × C × H × W, lancé une fois par image ingérée ;
// - la fenêtre, le graphe d'origine où la sortie de l'encodeur est remplacée par la concaténation (axe 0) de
//   frames entrées de traits gardés, une par image.
//
// Idée reprise de libaji (the-database/animejanai-inference, src/native/tfix.cpp, commit 1fea490 : « chaque
// image est encodée une fois »), écrite ici sur nos propres découpes ONNX ; aucune ligne reprise.
//
// Demi-taille (app/play/temporal_fix_model.h) : l'encodeur moyenne d'abord son image 2 × 2, et ses traits à
// demi-taille sont rangés en DepthToSpace (C / 4 canaux à pleine taille) ; la fenêtre les remet en place
// par SpaceToDepth. Toutes les entrées gardent ainsi les dimensions H et W des images : le profil de
// construction les donne à toutes (ModelSource::twins).
#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace kirari::prepare {

struct EncoderCache {
  std::string_view source = "/Reshape_1_output_0";       // l'entrée de l'encodeur : N·frames × 3 × H × W
  std::string_view encoded = "/encode/Resize_output_0";  // sa sortie : N·frames × channels × H × W
  std::string_view input = "input";                      // l'entrée du modèle de l'encodeur
  std::string_view features = "features";                // les entrées de traits de la fenêtre
  int frames = 7;
  int channels = 4;
  bool half = false;
};

// Le nom de l'entrée de traits k de la fenêtre : « <features>_<k> ».
std::string feature_input_name(std::string_view features, int k);
// Les canaux d'une entrée de traits : channels, ou channels / 4 en demi-taille (DepthToSpace).
int feature_channels(const EncoderCache& cache);

// L'encodeur seul : entrée cache.input (1 × 3 × H × W, H et W libres), sortie les traits (rangés en
// DepthToSpace en demi-taille). Refus : ModelUnreadable, FieldMissing (source ou encoded absents du graphe),
// FieldOutOfRange (channels pas multiple de 4 en demi-taille), ceux d'extract_subgraph.
std::expected<std::vector<std::byte>, Refusal> cached_encoder(std::span<const std::byte> model,
                                                              const EncoderCache& cache);

// La fenêtre : model (déjà coupé en images, onnx_frames.h) dont les entrées restent, suivies de frames
// entrées de traits (1 × feature_channels × H × W) ; l'encodeur n'y est plus. Mêmes refus.
std::expected<std::vector<std::byte>, Refusal> with_cached_encoder(std::span<const std::byte> model,
                                                                   const EncoderCache& cache);

}  // namespace kirari::prepare
