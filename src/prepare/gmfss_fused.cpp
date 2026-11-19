#include "prepare/gmfss_fused.h"

#include <charconv>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <system_error>
#include <utility>

#include "core/file_sync.h"
#include "core/utf8_path.h"
#include "prepare/derived_files.h"
#include "prepare/model_source.h"
#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"
#include "prepare/sha256.h"

namespace kirari::prepare {
namespace {

constexpr std::string_view kHeader = "kirari-gmfss-fused";
constexpr std::size_t kPieces = 2;

std::optional<std::string> read_size(std::string_view rest, GmfssFusedManifest& manifest) {
  const std::size_t gap = rest.find(' ');
  const std::string_view hs = rest.substr(0, gap);
  const std::string_view ws = gap == std::string_view::npos ? std::string_view() : rest.substr(gap + 1);
  const auto h = std::from_chars(hs.data(), hs.data() + hs.size(), manifest.height);
  const auto w = std::from_chars(ws.data(), ws.data() + ws.size(), manifest.width);
  if (h.ec != std::errc{} || w.ec != std::errc{} || h.ptr != hs.data() + hs.size() ||
      w.ptr != ws.data() + ws.size())
    return "taille illisible";
  return std::nullopt;
}

// Une ligne « clé reste » du manifeste ; rien : lue.
std::optional<std::string> read_field(std::string_view key, std::string_view rest,
                                      GmfssFusedManifest& manifest, bool& header, bool& size) {
  if (key == kHeader) {
    if (rest != "1") return "version " + std::string(rest);
    header = true;
  } else if (key == "size") {
    if (auto problem = read_size(rest, manifest)) return problem;
    size = true;
  } else if (key == "piece") {
    if (!is_piece_file_name(rest)) return "morceau « " + std::string(rest) + " »";
    manifest.pieces.emplace_back(rest);
  } else if (key == "sizes") {
    auto sizes = parse_sizes(rest);
    if (!sizes) return "tailles illisibles";
    manifest.sizes = std::move(*sizes);
  } else if (key == "digests") {
    auto digests = parse_digests(rest);
    if (!digests) return "empreintes illisibles";
    manifest.digests = std::move(*digests);
  } else {
    return "champ « " + std::string(key) + " » inconnu";
  }
  return std::nullopt;
}

// Les noyaux prennent une fusion dont les deux côtés sont des multiples de 32 (le quart du GridNet en entier
// de pixels, les plans remplis de GMFSS étant des multiples de 64), et au plus 4K.
bool size_valid(int height, int width) {
  return height >= 32 && width >= 32 && height % 32 == 0 && width % 32 == 0 && height <= 2176 &&
         width <= 3840;
}

}  // namespace

std::string format_gmfss_manifest(const GmfssFusedManifest& manifest) {
  std::string text = std::format("{} 1\n", kHeader);
  text += std::format("size {} {}\n", manifest.height, manifest.width);
  for (const std::string& piece : manifest.pieces) text += std::format("piece {}\n", piece);
  text += format_sizes(manifest.sizes);
  text += format_digests(manifest.digests);
  return text;
}

std::expected<GmfssFusedManifest, Refusal> parse_gmfss_manifest(std::string_view text) {
  const auto bad = [](std::string detail) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange, "GMFSS natif : manifeste : " + std::move(detail)});
  };
  GmfssFusedManifest manifest;
  bool header = false;
  bool size = false;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    const std::string_view line = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view() : text.substr(end + 1);
    if (line.empty()) continue;
    const std::size_t space = line.find(' ');
    const std::string_view rest =
        space == std::string_view::npos ? std::string_view() : line.substr(space + 1);
    if (auto problem = read_field(line.substr(0, space), rest, manifest, header, size))
      return bad(std::move(*problem));
  }
  if (!header || !size) return bad("en-tête ou taille absents");
  if (!size_valid(manifest.height, manifest.width)) return bad("taille que les noyaux ne prennent pas");
  if (manifest.pieces.size() != kPieces) return bad("la fusion et l'IFNet attendus");
  if (manifest.sizes.size() != kPieces || manifest.digests.size() != kPieces)
    return bad("une taille et une empreinte par morceau attendues");
  return manifest;
}

bool is_gmfss_fused_manifest(const std::filesystem::path& path) {
  return path.extension() == kGmfssFusedExtension;
}

std::expected<std::vector<std::byte>, Refusal> gmfss_ifnet(std::span<const std::byte> fusion, int height,
                                                           int width) {
  const std::vector<SplitTensor> inputs{
      {.name = "half0", .dims = {1, 3, height, width}},
      {.name = "half1", .dims = {1, 3, height, width}},
      {.name = "time", .dims = {1, 1, 1, 1}},
  };
  const std::vector<SplitTensor> outputs{{.name = std::string(kGmfssRifeTensor)}};
  return extract_subgraph(fusion, inputs, outputs, "gmfss_ifnet");
}

std::expected<std::filesystem::path, Refusal> derive_gmfss_fused(const std::filesystem::path& fusion,
                                                                 int height, int width) {
  if (!size_valid(height, width))
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("GMFSS natif : fusion de {}x{}, des multiples de 32 attendus", width, height));
  auto sha = sha256_file_known(fusion);
  if (!sha) return std::unexpected(std::move(sha.error()));
  const std::filesystem::path dir = fusion.parent_path();
  const std::string stem = std::format("{}-{}-{}x{}-{}", core::to_utf8(fusion.stem()), kGmfssFusedDerivation,
                                       height, width, sha->substr(0, 16));
  std::filesystem::path manifest_path =
      dir / core::path_from_utf8(std::format("{}{}", stem, kGmfssFusedExtension));
  // Déjà dérivé : le manifeste est écrit en dernier, sa présence dit que l'IFNet y est.
  if (std::ifstream known(manifest_path); known) {
    const std::string text{std::istreambuf_iterator<char>(known), std::istreambuf_iterator<char>()};
    if (const auto parsed = parse_gmfss_manifest(text);
        parsed && parsed->height == height && parsed->width == width && pieces_intact(dir, *parsed))
      return manifest_path;
  }
  auto bytes = read_onnx_bytes(fusion);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto ifnet = gmfss_ifnet(*bytes, height, width);
  if (!ifnet) return std::unexpected(std::move(ifnet.error()));
  const std::string file = std::format("{}-ifnet.onnx", stem);
  const std::string digest = sha256_hex(byte_text(*ifnet));
  if (!file_matches(dir / core::path_from_utf8(file), ifnet->size(), digest)) {
    if (auto written = core::write_atomic(dir / core::path_from_utf8(file), *ifnet); !written)
      return std::unexpected(std::move(written.error()));
  }
  const GmfssFusedManifest manifest{.height = height,
                                    .width = width,
                                    .pieces = {core::to_utf8(fusion.filename()), file},
                                    .sizes = {bytes->size(), ifnet->size()},
                                    .digests = {*sha, digest}};
  const std::string text = format_gmfss_manifest(manifest);
  const auto* raw = reinterpret_cast<const std::byte*>(text.data());
  if (auto written = core::write_atomic(manifest_path, std::span(raw, text.size())); !written)
    return std::unexpected(std::move(written.error()));
  return manifest_path;
}

std::expected<std::vector<std::byte>, Refusal> read_gmfss_piece(const std::filesystem::path& dir,
                                                                const std::string& piece,
                                                                const std::string& digest) {
  const std::filesystem::path file = dir / core::path_from_utf8(piece);
  auto sha = sha256_file_known(file);
  if (!sha) return std::unexpected(std::move(sha.error()));
  if (*sha != digest)
    return refuse(RefusalCode::ModelUnreadable, std::format("GMFSS natif : {} a changé", piece));
  return read_onnx_bytes(file);
}

}  // namespace kirari::prepare
