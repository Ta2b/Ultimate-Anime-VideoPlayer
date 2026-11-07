// Les cibles de fuzz : chacune lit des octets quelconques et vérifie ses propriétés
// (fuzz_check.h). replay_main.cpp les appelle par leur nom ; libfuzzer_entry.cpp en fait la
// LLVMFuzzerTestOneInput d'un exécutable libFuzzer (AJP_FUZZ_TARGET).
#pragma once

#include <cstddef>
#include <cstdint>

namespace ajp::fuzz {

int settings(const std::uint8_t* data, std::size_t size);       // settings.toml
int play_settings(const std::uint8_t* data, std::size_t size);  // play.toml
int edid(const std::uint8_t* data, std::size_t size);           // EDID de l'écran
int build_log(const std::uint8_t* data, std::size_t size);      // *.build.log du cache
int resume(const std::uint8_t* data, std::size_t size);         // resume.tsv
int osd_script(const std::uint8_t* data, std::size_t size);     // texte de l'OSD en ASS
// fuzz_parsers.cpp
int onnx(const std::uint8_t* data, std::size_t size);      // modèle ONNX (protobuf)
int srt(const std::uint8_t* data, std::size_t size);       // SRT, encodage compris
int cube(const std::uint8_t* data, std::size_t size);      // LUT .cube
int inflate(const std::uint8_t* data, std::size_t size);   // deflate brut
int manifest(const std::uint8_t* data, std::size_t size);  // manifeste de mise à jour
int options(const std::uint8_t* data, std::size_t size);   // ligne de commande

}  // namespace ajp::fuzz
