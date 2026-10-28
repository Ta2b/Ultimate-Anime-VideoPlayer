// Le choix de l'encodeur d'un export : les encodeurs FFmpeg à essayer, dans l'ordre, pour un codec,
// une qualité et le GPU du rendu, avec leurs options. Logique pure, sans FFmpeg : video_encoder.cpp
// ouvre le premier qui accepte (un encodeur absent de cette construction de FFmpeg est sauté).
//
// L'ordre suit le GPU du rendu : NVENC sur une carte NVIDIA, AMF puis VA-API sur une AMD, QSV puis
// VA-API sur une Intel, VideoToolbox sur un Mac ; Media Foundation (Windows) et Vulkan Video
// viennent ensuite, pour tout GPU. Les qualités visent un rendu visuellement identique à la lecture
// en direct : quantification constante basse, 10 bits sauf en H.264.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ajp::exporter {

enum class Codec : std::uint8_t {
  Hevc,      // HEVC Main 10 : le défaut, lu partout
  Av1,       // AV1 10 bits : plus compact, encodeur matériel récent (RTX 40, RDNA 3, Arc)
  H264,      // H.264 8 bits : compatibilité
  Lossless,  // FFV1 10 bits sans perte, encodé par le processeur : fichiers très lourds
};
inline constexpr int kCodecCount = 4;

enum class Quality : std::uint8_t { VeryHigh, High, Standard };
inline constexpr int kQualityCount = 3;

// Disposition des plans que l'encodeur lit en mémoire hôte.
enum class Layout : std::uint8_t {
  P010,       // Y, puis U et V entrelacés ; 10 bits dans les poids forts de 16
  Nv12,       // la même en 8 bits
  Yuv420p10,  // Y, U et V séparés ; 10 bits dans les poids faibles de 16
};

// Comment l'encodeur reçoit ses images.
enum class Upload : std::uint8_t {
  Host,    // images en mémoire hôte, copiées par l'encodeur lui-même
  Cuda,    // mémoire CUDA : le rendu y copie ses plans sur le GPU, NVENC les lit sans l'hôte ; une
           // image hôte y est chargée (av_hwframe_transfer_data)
  Vaapi,   // surfaces VA-API, chargées depuis l'hôte (av_hwframe_transfer_data)
  Vulkan,  // images Vulkan Video, chargées de même
};

enum class GpuVendor : std::uint8_t { Nvidia, Amd, Intel, Apple, Other };
// Le constructeur d'un identifiant PCI (VkPhysicalDeviceProperties::vendorID).
GpuVendor vendor_of(std::uint32_t pci_vendor);

struct EncoderOption {
  std::string key;  // option de l'encodeur ou de AVCodecContext (av_opt_set)
  std::string value;
  bool operator==(const EncoderOption&) const = default;
};

struct EncoderCandidate {
  std::string name;   // « hevc_nvenc »
  std::string label;  // « NVENC », pour l'interface et le journal
  Upload upload = Upload::Host;
  bool ten_bit = true;  // P010 (ou YUV 4:2:0 10 bits) ; faux : NV12
  std::vector<EncoderOption> options;
  // Les options d'une génération récente (Ada, Blackwell), chacune un jeu ajouté à options et
  // essayé avant lui, le plus exigeant d'abord : un GPU plus ancien les refuse à l'ouverture.
  std::vector<std::vector<EncoderOption>> recent;
  // Le jeu réduit, essayé quand le GPU refuse une option du premier (B-frames, AQ temporelle d'une
  // génération plus ancienne) : le contrôle de débit et la quantification seulement.
  std::vector<EncoderOption> fallback;
};

// Les encodeurs à essayer, le meilleur d'abord. Toujours au moins un.
std::vector<EncoderCandidate> encoder_candidates(Codec codec, Quality quality, GpuVendor vendor);

std::string_view label(Codec codec);      // « HEVC 10 bits »
std::string_view label(Quality quality);  // « Très haute »

}  // namespace ajp::exporter
