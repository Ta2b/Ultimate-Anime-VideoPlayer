#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include "media/hw_device.h"

extern "C" {
#include <va/va.h>

#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vaapi.h>
}

namespace ajp::media {
namespace {

struct RenderNode {
  std::string path;  // /dev/dri/renderD128
  bool intel_igpu = false;
};

std::string read_line(const std::filesystem::path& file) {
  std::ifstream in(file);
  std::string line;
  std::getline(in, line);
  return line;
}

// Les nœuds de rendu dans l'ordre du noyau ; l'iGPU Intel est toujours 0000:00:02.0.
std::vector<RenderNode> render_nodes() {
  std::vector<RenderNode> nodes;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
    const std::string name = entry.path().filename().string();
    if (!name.starts_with("renderD")) continue;
    const std::filesystem::path device = std::filesystem::canonical(entry.path() / "device", ec);
    const bool intel = read_line(entry.path() / "device" / "vendor") == "0x8086";
    nodes.push_back(
        {.path = "/dev/dri/" + name, .intel_igpu = intel && !ec && device.filename() == "0000:00:02.0"});
  }
  std::ranges::sort(nodes, {}, &RenderNode::path);
  return nodes;
}

std::vector<VAProfile> profiles_of(AVCodecID codec) {
  switch (codec) {
    case AV_CODEC_ID_H264:
      return {VAProfileH264ConstrainedBaseline, VAProfileH264Main, VAProfileH264High};
    case AV_CODEC_ID_HEVC:
      return {VAProfileHEVCMain, VAProfileHEVCMain10};
    case AV_CODEC_ID_AV1:
      return {VAProfileAV1Profile0};
    case AV_CODEC_ID_VP9:
      return {VAProfileVP9Profile0, VAProfileVP9Profile2};
    case AV_CODEC_ID_MPEG2VIDEO:
      return {VAProfileMPEG2Main};
    default:
      return {};
  }
}

// Le pilote du nœud a-t-il un point d'entrée de décodage (VLD) pour un profil du codec ? Un
// codec hors de la table est laissé à FFmpeg, qui retombe en logiciel au besoin.
bool decodes(VADisplay display, AVCodecID codec) {
  const std::vector<VAProfile> wanted = profiles_of(codec);
  if (wanted.empty()) return true;
  std::vector<VAProfile> listed(static_cast<std::size_t>(std::max(vaMaxNumProfiles(display), 1)));
  int count = 0;
  if (vaQueryConfigProfiles(display, listed.data(), &count) != VA_STATUS_SUCCESS) return false;
  listed.resize(static_cast<std::size_t>(count));
  std::vector<VAEntrypoint> entries(static_cast<std::size_t>(std::max(vaMaxNumEntrypoints(display), 1)));
  for (const VAProfile profile : wanted) {
    if (!std::ranges::contains(listed, profile)) continue;
    int n = 0;
    if (vaQueryConfigEntrypoints(display, profile, entries.data(), &n) != VA_STATUS_SUCCESS) continue;
    if (std::ranges::contains(std::span(entries).first(static_cast<std::size_t>(n)), VAEntrypointVLD))
      return true;
  }
  return false;
}

}  // namespace

std::expected<BufferPtr, Refusal> open_vaapi_device(AVCodecID codec, bool prefer_igpu) {
  std::vector<RenderNode> nodes = render_nodes();
  if (prefer_igpu) std::ranges::stable_partition(nodes, &RenderNode::intel_igpu);
  std::string tried;
  for (const RenderNode& node : nodes) {
    AVBufferRef* ref = nullptr;
    if (av_hwdevice_ctx_create(&ref, AV_HWDEVICE_TYPE_VAAPI, node.path.c_str(), nullptr, 0) < 0) continue;
    BufferPtr device(ref);
    const auto& ctx = *reinterpret_cast<const AVHWDeviceContext*>(device->data);
    if (decodes(static_cast<const AVVAAPIDeviceContext*>(ctx.hwctx)->display, codec)) return device;
    tried += tried.empty() ? node.path : ", " + node.path;
  }
  if (tried.empty())
    return refuse(RefusalCode::HwDecodeUnavailable, "vaapi : aucun nœud de rendu ne s'ouvre");
  return refuse(RefusalCode::HwDecodeUnavailable,
                std::format("vaapi : {} ne décode pas {}", tried, avcodec_get_name(codec)));
}

}  // namespace ajp::media
