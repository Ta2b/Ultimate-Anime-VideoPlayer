#include "export/encoder_choice.h"

#include <array>
#include <cstddef>
#include <format>
#include <utility>

namespace ajp::exporter {
namespace {

enum class Family : std::uint8_t { Nvenc, Amf, Qsv, VideoToolbox, Vaapi, MediaFoundation, Vulkan };

constexpr std::size_t index(Quality quality) {
  return static_cast<std::size_t>(quality);
}

// La quantification de chaque qualité, par codec : HEVC et H.264 sur l'échelle 0 à 51, AV1 sur
// l'échelle de son encodeur (0 à 63 pour NVENC, 0 à 255 pour AMF, VA-API et Vulkan).
constexpr std::array<int, kQualityCount> kHevcQp{17, 20, 24};
constexpr std::array<int, kQualityCount> kH264Qp{17, 20, 23};
constexpr std::array<int, kQualityCount> kAv1NvencCq{24, 28, 33};
constexpr std::array<int, kQualityCount> kAv1QIndex{72, 92, 112};

int qp_of(Codec codec, Quality quality) {
  switch (codec) {
    case Codec::H264:
      return kH264Qp[index(quality)];
    case Codec::Av1:
      return kAv1QIndex[index(quality)];
    default:
      return kHevcQp[index(quality)];
  }
}

std::string_view prefix(Codec codec) {
  switch (codec) {
    case Codec::Av1:
      return "av1";
    case Codec::H264:
      return "h264";
    default:
      return "hevc";
  }
}

std::vector<EncoderOption> nvenc_rate(Codec codec, Quality quality) {
  constexpr std::array<std::string_view, kQualityCount> kPresets{"p6", "p5", "p4"};
  const int cq = codec == Codec::Av1 ? kAv1NvencCq[index(quality)] : qp_of(codec, quality);
  return {{"preset", std::string(kPresets[index(quality)])},
          {"tune", "hq"},
          {"rc", "vbr"},
          {"cq", std::to_string(cq)},
          {"b", "0"}};
}

EncoderCandidate nvenc(Codec codec, Quality quality) {
  EncoderCandidate c{
      .name = std::format("{}_nvenc", prefix(codec)), .label = "NVENC", .upload = Upload::Cuda};
  c.fallback = nvenc_rate(codec, quality);
  c.options = c.fallback;
  c.options.insert(c.options.end(), {{"spatial-aq", "1"},
                                     {"temporal-aq", "1"},
                                     {"aq-strength", "8"},
                                     {"rc-lookahead", "32"},
                                     {"bf", "3"},
                                     {"b_ref_mode", "middle"}});
  if (quality == Quality::VeryHigh) c.options.push_back({"multipass", "qres"});
  // Ada et après : l'anticipation à 3 niveaux, et en HEVC et AV1 le filtre temporel (4 B-frames
  // au moins) ; Blackwell, en très haute qualité : le réglage UHQ, qui les règle lui-même.
  std::vector<EncoderOption> ada{{"lookahead_level", "3"}};
  if (codec != Codec::H264) ada.insert(ada.end(), {{"bf", "4"}, {"tf_level", "4"}});
  if (codec != Codec::H264 && quality == Quality::VeryHigh) {
    std::vector<EncoderOption> uhq = ada;
    uhq.push_back({"tune", "uhq"});
    c.recent.push_back(std::move(uhq));
  }
  c.recent.push_back(std::move(ada));
  return c;
}

EncoderCandidate amf(Codec codec, Quality quality) {
  const int qp = qp_of(codec, quality);
  EncoderCandidate c{.name = std::format("{}_amf", prefix(codec)), .label = "AMF"};
  c.fallback = {{"rc", "cqp"}, {"qp_i", std::to_string(qp)}, {"qp_p", std::to_string(qp + 2)}};
  c.options = c.fallback;
  c.options.insert(c.options.begin(), {{"usage", "transcoding"}, {"quality", "quality"}});
  return c;
}

EncoderCandidate qsv(Codec codec, Quality quality) {
  constexpr std::array<std::string_view, kQualityCount> kPresets{"veryslow", "slower", "slow"};
  EncoderCandidate c{.name = std::format("{}_qsv", prefix(codec)), .label = "Quick Sync"};
  c.fallback = {{"global_quality", std::to_string(qp_of(codec, quality))}};  // ICQ
  c.options = c.fallback;
  c.options.push_back({"preset", std::string(kPresets[index(quality)])});
  return c;
}

EncoderCandidate videotoolbox(Codec codec, Quality quality) {
  constexpr std::array<int, kQualityCount> kQuality{80, 70, 60};  // sur 100
  constexpr std::array<std::string_view, kQualityCount> kBitrate{"60M", "40M", "25M"};
  constexpr int kQp2Lambda = 118;  // FF_QP2LAMBDA : global_quality / 118 / 100 = qualité de 0 à 1
  EncoderCandidate c{.name = std::format("{}_videotoolbox", prefix(codec)), .label = "VideoToolbox"};
  c.options = {{"flags", "+qscale"},
               {"global_quality", std::to_string(kQuality[index(quality)] * kQp2Lambda)},
               {"realtime", "0"}};
  // Un Mac Intel sans qualité constante en HEVC : un débit haut à la place.
  c.fallback = {{"b", std::string(kBitrate[index(quality)])}};
  return c;
}

EncoderCandidate vaapi(Codec codec, Quality quality) {
  EncoderCandidate c{
      .name = std::format("{}_vaapi", prefix(codec)), .label = "VA-API", .upload = Upload::Vaapi};
  c.fallback = {{"rc_mode", "CQP"}, {"qp", std::to_string(qp_of(codec, quality))}};
  c.options = c.fallback;
  return c;
}

EncoderCandidate media_foundation(Codec codec, Quality quality) {
  constexpr std::array<int, kQualityCount> kQuality{90, 80, 70};  // AVEncCommonQuality, sur 100
  constexpr std::array<std::string_view, kQualityCount> kBitrate{"60M", "40M", "25M"};
  EncoderCandidate c{.name = std::format("{}_mf", prefix(codec)), .label = "Media Foundation"};
  c.options = {{"hw_encoding", "1"},
               {"rate_control", "quality"},
               {"global_quality", std::to_string(kQuality[index(quality)])}};
  c.fallback = {
      {"hw_encoding", "1"}, {"rate_control", "pc_vbr"}, {"b", std::string(kBitrate[index(quality)])}};
  return c;
}

EncoderCandidate vulkan(Codec codec, Quality quality) {
  EncoderCandidate c{
      .name = std::format("{}_vulkan", prefix(codec)), .label = "Vulkan Video", .upload = Upload::Vulkan};
  c.fallback = {{"rc_mode", "cqp"}, {"qp", std::to_string(qp_of(codec, quality))}};
  c.options = c.fallback;
  c.options.insert(c.options.end(), {{"tune", "hq"}, {"usage", "transcode"}});
  return c;
}

// Les familles dans l'ordre d'essai pour un GPU : la sienne d'abord, puis celles de tout GPU.
std::vector<Family> order_for(GpuVendor vendor) {
  using enum Family;
  switch (vendor) {
    case GpuVendor::Nvidia:
      return {Nvenc, Vaapi, MediaFoundation, Vulkan, Qsv, Amf, VideoToolbox};
    case GpuVendor::Amd:
      return {Amf, Vaapi, MediaFoundation, Vulkan, Nvenc, Qsv, VideoToolbox};
    case GpuVendor::Intel:
      return {Qsv, Vaapi, MediaFoundation, Vulkan, Nvenc, Amf, VideoToolbox};
    case GpuVendor::Apple:
      return {VideoToolbox, Vulkan};
    case GpuVendor::Other:
      break;
  }
  return {VideoToolbox, Nvenc, Amf, Qsv, Vaapi, MediaFoundation, Vulkan};
}

}  // namespace

GpuVendor vendor_of(std::uint32_t pci_vendor) {
  switch (pci_vendor) {
    case 0x10DE:
      return GpuVendor::Nvidia;
    case 0x1002:
    case 0x1022:
      return GpuVendor::Amd;
    case 0x8086:
      return GpuVendor::Intel;
    case 0x106B:
      return GpuVendor::Apple;
    default:
      return GpuVendor::Other;
  }
}

std::vector<EncoderCandidate> encoder_candidates(Codec codec, Quality quality, GpuVendor vendor) {
  if (codec == Codec::Lossless) {
    // FFV1 version 3 : découpé en tranches encodées en parallèle, chacune avec sa somme de contrôle.
    EncoderCandidate ffv1{.name = "ffv1", .label = "FFV1"};
    ffv1.options = {{"level", "3"},   {"slices", "24"}, {"slicecrc", "1"},
                    {"context", "1"}, {"g", "1"},       {"threads", "auto"}};
    ffv1.fallback = {{"level", "3"}, {"threads", "auto"}};
    return {std::move(ffv1)};
  }
  std::vector<EncoderCandidate> out;
  for (const Family family : order_for(vendor)) {
    switch (family) {
      case Family::Nvenc:
        out.push_back(nvenc(codec, quality));
        break;
      case Family::Amf:
        out.push_back(amf(codec, quality));
        break;
      case Family::Qsv:
        if (codec != Codec::Av1) out.push_back(qsv(codec, quality));  // échelle ICQ de l'AV1 incertaine
        break;
      case Family::VideoToolbox:
        if (codec != Codec::Av1) out.push_back(videotoolbox(codec, quality));  // pas d'encodeur AV1 Apple
        break;
      case Family::Vaapi:
        out.push_back(vaapi(codec, quality));
        break;
      case Family::MediaFoundation:
        out.push_back(media_foundation(codec, quality));
        break;
      case Family::Vulkan:
        out.push_back(vulkan(codec, quality));
        break;
    }
  }
  if (codec == Codec::H264) {
    for (EncoderCandidate& c : out) c.ten_bit = false;
  }
  return out;
}

std::string_view label(Codec codec) {
  switch (codec) {
    case Codec::Hevc:
      return "HEVC 10 bits";
    case Codec::Av1:
      return "AV1 10 bits";
    case Codec::H264:
      return "H.264 8 bits";
    case Codec::Lossless:
      return "FFV1 4:2:0 10 bits, sans perte du codec";
  }
  return "?";
}

std::string_view label(Quality quality) {
  switch (quality) {
    case Quality::VeryHigh:
      return "Très haute";
    case Quality::High:
      return "Haute";
    case Quality::Standard:
      return "Standard";
  }
  return "?";
}

}  // namespace ajp::exporter
