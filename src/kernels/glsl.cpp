#include "kernels/glsl.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <string_view>
#include <utility>

#include "kernels/spec.h"

namespace ajp::kernels {
namespace {

constexpr std::array kFormats{Format::Nv12,    Format::P010,      Format::P016,
                              Format::Yuv420p, Format::Yuv420p10, Format::Yuv444p16};
constexpr std::array kMatrices{Matrix::Bt601, Matrix::Bt709, Matrix::Bt2020Ncl};
constexpr std::array kSitings{Siting::Left, Siting::Center, Siting::TopLeft};

std::string_view glsl_name(Format f) {
  switch (f) {
    case Format::Nv12:
      return "NV12";
    case Format::P010:
      return "P010";
    case Format::P016:
      return "P016";
    case Format::Yuv420p:
      return "YUV420P";
    case Format::Yuv420p10:
      return "YUV420P10";
    case Format::Yuv444p16:
      return "YUV444P16";
  }
  return "UNKNOWN";
}

std::string_view glsl_name(Matrix m) {
  switch (m) {
    case Matrix::Bt601:
      return "BT601";
    case Matrix::Bt709:
      return "BT709";
    case Matrix::Bt2020Ncl:
      return "BT2020";
  }
  return "UNKNOWN";
}

std::string_view glsl_name(Siting s) {
  switch (s) {
    case Siting::Left:
      return "LEFT";
    case Siting::Center:
      return "CENTER";
    case Siting::TopLeft:
      return "TOPLEFT";
  }
  return "UNKNOWN";
}

std::string_view glsl_name(Range r) {
  return r == Range::Limited ? "LIMITED" : "FULL";
}

// Les constantes d'un sujet, ajoutées dans l'ordre de l'en-tête.
class Table {
 public:
  void real(std::string name, float value, std::string comment = {}) {
    add(std::move(name), GlslType::Float, {value}, 0, std::move(comment));
  }
  void integer(std::string name, std::int64_t value, std::string comment = {}) {
    add(std::move(name), GlslType::Int, {}, value, std::move(comment));
  }
  void unsigned_integer(std::string name, std::uint32_t value, std::string comment = {}) {
    add(std::move(name), GlslType::Uint, {}, value, std::move(comment));
  }
  void vec2(std::string name, float x, float y, std::string comment = {}) {
    add(std::move(name), GlslType::Vec2, {x, y}, 0, std::move(comment));
  }
  void vec4(std::string name, const std::array<float, 4>& v, std::string comment = {}) {
    add(std::move(name), GlslType::Vec4, {v.begin(), v.end()}, 0, std::move(comment));
  }
  // GLSL range une mat3 colonne par colonne.
  void mat3(std::string name, const Mat3& m, std::string comment = {}) {
    std::vector<float> columns;
    for (int c = 0; c < 3; ++c) {
      for (int r = 0; r < 3; ++r) columns.push_back(static_cast<float>(m.at(r, c)));
    }
    add(std::move(name), GlslType::Mat3, std::move(columns), 0, std::move(comment));
  }

  std::vector<GlslConstant> take() { return std::move(constants_); }

 private:
  void add(std::string name, GlslType type, std::vector<float> values, std::int64_t value,
           std::string comment) {
    constants_.push_back({.name = "AJP_" + std::move(name),
                          .type = type,
                          .values = std::move(values),
                          .integer = value,
                          .comment = std::move(comment)});
  }

  std::vector<GlslConstant> constants_;
};

void add_color(Table& t) {
  for (const Matrix m : kMatrices) {
    const Coefficients k = coefficients(m);
    t.real(std::format("KR_{}", glsl_name(m)), k.kr, std::format("Luma {} : kr et kb", glsl_name(m)));
    t.real(std::format("KB_{}", glsl_name(m)), k.kb);
  }
  for (const Format f : kFormats) {
    const std::string_view n = glsl_name(f);
    for (const Range r : {Range::Limited, Range::Full}) {
      const YuvScale s = yuv_scale(f, r);
      t.vec4(
          std::format("YUV_SCALE_{}_{}", n, glsl_name(r)), {s.yoff, s.yscale, s.coff, s.cscale},
          r == Range::Limited ? std::format("{} : yoff, yscale, coff, cscale en codes du conteneur", n) : "");
    }
    const Quantizer q = quantizer(f);
    t.vec2(std::format("QUANT_{}", n), q.qdiv, q.qmax, "code = clamp(roundEven(v / qdiv), 0, qmax) * qdiv");
    t.unsigned_integer(std::format("MAX_CODE_{}", n), max_code(f));
    t.unsigned_integer(std::format("LEVEL_UNIT_{}", n), level_unit(f), "Codes pour un niveau 8 bits");
  }
  t.real("UNORM16_MAX", kUnorm16Max, "RGBA16 : uint(clamp(v, 0, 1) * 65535 + 0.5), NaN donne 0");
  t.unsigned_integer("OPAQUE_ALPHA16", kOpaqueAlpha16);
  t.integer("SD_HEIGHT_LIMIT", kSdHeightLimit, "Matrice non étiquetée : BT.601 sous cette hauteur");
}

void add_resample(Table& t) {
  t.real("SPLINE36_SUPPORT", static_cast<float>(support(Filter::Spline36)),
         "Supports des noyaux, en pixels source");
  t.real("BILINEAR_SUPPORT", static_cast<float>(support(Filter::Bilinear)));
  for (const Siting s : kSitings) {
    const ChromaShift up = chroma_shift_up(s);
    const ChromaShift down = chroma_shift_down(s);
    t.vec2(std::format("CHROMA_SHIFT_UP_{}", glsl_name(s)), static_cast<float>(up.x),
           static_cast<float>(up.y),
           std::format("Siting {} : décalages de pos(i), 4:2:0 vers luma puis l'inverse", glsl_name(s)));
    t.vec2(std::format("CHROMA_SHIFT_DOWN_{}", glsl_name(s)), static_cast<float>(down.x),
           static_cast<float>(down.y));
  }
  t.integer("MAX_DIMENSION", kMaxDimension);
}

void add_rife_warp(Table& t) {
  t.integer("RIFE_ALIGN", kRifeAlign, "RIFE : dimensions remplies au multiple de 64, bords centrés pairs");
  t.integer("RIFE_INPUT_CHANNELS", kRifeInputChannels);
  t.integer("RIFE_CHANNEL_A", kRifeChannelA);
  t.integer("RIFE_CHANNEL_B", kRifeChannelB);
  t.integer("RIFE_CHANNEL_TIME", kRifeChannelTime);
  t.integer("RIFE_CHANNEL_MESH_X", kRifeChannelMeshX);
  t.integer("RIFE_CHANNEL_MESH_Y", kRifeChannelMeshY);
  t.integer("RIFE_CHANNEL_STEP_X", kRifeChannelStepX);
  t.integer("RIFE_CHANNEL_STEP_Y", kRifeChannelStepY);
  for (const Format f : kFormats) {
    t.unsigned_integer(std::format("RIFE_PAD_LUMA_{}", glsl_name(f)), rife_pad_luma(f),
                       f == kFormats[0] ? "Bords noir studio, quelle que soit la plage" : "");
    t.unsigned_integer(std::format("RIFE_PAD_CHROMA_{}", glsl_name(f)), rife_pad_chroma(f));
  }
  t.real("BLEND_LAW_A", kBlendLawA, "logit(m) = A logit(mm) + B ln((1 - t) / t) + C");
  t.real("BLEND_LAW_B", kBlendLawB);
  t.real("BLEND_LAW_C", kBlendLawC);
  t.real("MASK_EPSILON", kMaskEpsilon, "mm = clamp(masque, epsilon, 1 - epsilon)");
  t.real("SINGLE_RUN_MOTION_720", kSingleRunMotion720, "Pixels de mouvement à 720p");
  t.integer("POSE_ITERATIONS", kPoseIterations, "Garder les poses (spec/poses.h)");
  t.real("POSE_EPSILON", kPoseEpsilon);
  t.real("POSE_TIME_LIMIT", kPoseTimeLimit);
  t.real("GMFSS_NORMALIZE_EPSILON", kGmfssNormalizeEpsilon, "GMFSS : somme / (poids + epsilon)");
  t.real("GMFSS_METRIC_LIMIT", kGmfssMetricLimit, "GMFSS : la métrique z est bornée à [-limite, limite]");
}

void add_transfer(Table& t) {
  t.real("PQ_M1", pq::kM1, "PQ (SMPTE ST 2084)");
  t.real("PQ_M2", pq::kM2);
  t.real("PQ_C1", pq::kC1);
  t.real("PQ_C2", pq::kC2);
  t.real("PQ_C3", pq::kC3);
  t.real("PQ_PEAK_NITS", pq::kPeakNits);
  t.real("HLG_A", hlg::kA, "HLG (BT.2100)");
  t.real("HLG_B", hlg::kB);
  t.real("HLG_C", hlg::kC);
  t.real("HLG_KNEE", hlg::kKnee);
  t.real("HLG_GAMMA_AT_1000", hlg::kGammaAt1000);
  t.real("HLG_GAMMA_SLOPE", hlg::kGammaSlope);
  t.real("SRGB_ENCODED_KNEE", srgb::kEncodedKnee, "sRGB (IEC 61966-2-1)");
  t.real("SRGB_LINEAR_KNEE", srgb::kLinearKnee);
  t.real("SRGB_SLOPE", srgb::kSlope);
  t.real("SRGB_OFFSET", srgb::kOffset);
  t.real("SRGB_GAMMA", srgb::kGamma);
  t.real("BT1886_GAMMA", bt1886::kGamma, "BT.1886");
  t.mat3("BT709_TO_BT2020", kBt709ToBt2020,
         "Primaires en lumière linéaire : rgb2020 = AJP_BT709_TO_BT2020 * rgb709");
  t.mat3("BT2020_TO_BT709", kBt2020ToBt709);
}

void add_detect(Table& t) {
  t.real("SCENE_CUT_THRESHOLD", static_cast<float>(kSceneCutThreshold),
         "Coupe si somme / (pw * ph) > seuil (comparé en double sur l'hôte)");
  for (const Format f : kFormats) t.real(std::format("SCENE_NORM_{}", glsl_name(f)), scene_norm(f));
  t.integer("HELD_BLOCK", kHeldBlock, "Dessins tenus : tenu si pire * 100 <= seuil * 256");
  t.unsigned_integer("HELD_THRESHOLD", kHeldThreshold);
  for (const Format f : kFormats)
    t.unsigned_integer(std::format("HELD_SCALE_{}", glsl_name(f)), held_scale(f));
}

std::string_view type_name(GlslType type) {
  switch (type) {
    case GlslType::Float:
      return "float";
    case GlslType::Int:
      return "int";
    case GlslType::Uint:
      return "uint";
    case GlslType::Vec2:
      return "vec2";
    case GlslType::Vec4:
      return "vec4";
    case GlslType::Mat3:
      return "mat3";
  }
  return "float";
}

// AJP_ suivi de majuscules, chiffres et soulignés : un identifiant GLSL sans collision.
bool valid_name(std::string_view name) {
  constexpr std::string_view kPrefix = "AJP_";
  return name.size() > kPrefix.size() && name.starts_with(kPrefix) &&
         name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == std::string_view::npos;
}

// Un int ou un uint GLSL tient sur 32 bits. −2³¹ est exclu : « -2147483648 » applique le moins
// à 2147483648, un littéral trop grand pour un int.
bool integer_fits(const GlslConstant& c) {
  constexpr std::int64_t kIntMax = std::numeric_limits<std::int32_t>::max();
  switch (c.type) {
    case GlslType::Int:
      return c.integer >= -kIntMax && c.integer <= kIntMax;
    case GlslType::Uint:
      return c.integer >= 0 && c.integer <= std::numeric_limits<std::uint32_t>::max();
    case GlslType::Float:
    case GlslType::Vec2:
    case GlslType::Vec4:
    case GlslType::Mat3:
      return true;
  }
  return false;
}

}  // namespace

std::vector<GlslConstant> glsl_constants() {
  Table t;
  add_color(t);
  add_resample(t);
  add_rife_warp(t);
  add_transfer(t);
  add_detect(t);
  return t.take();
}

Result<std::string> glsl_literal(float value) {
  if (!std::isfinite(value)) {
    return fail(ErrorCode::InvalidConstant, std::format("littéral {} sans équivalent GLSL", value));
  }
  std::string text = std::format("{}", value);
  if (text.find_first_of(".e") == std::string::npos) text += ".0";
  return text;
}

Result<std::string> glsl_declaration(const GlslConstant& c) {
  if (!valid_name(c.name)) {
    return fail(ErrorCode::InvalidConstant, std::format("nom « {} » hors de AJP_[A-Z0-9_]+", c.name));
  }
  if (c.values.size() != glsl_float_count(c.type)) {
    return fail(ErrorCode::InvalidConstant,
                std::format("{} : {} valeurs pour un {}", c.name, c.values.size(), type_name(c.type)));
  }
  if (!integer_fits(c)) {
    return fail(ErrorCode::InvalidConstant,
                std::format("{} : {} hors d'un {}", c.name, c.integer, type_name(c.type)));
  }
  std::string value;
  if (c.type == GlslType::Int) value = std::format("{}", c.integer);
  if (c.type == GlslType::Uint) value = std::format("{}u", c.integer);
  for (std::size_t i = 0; i < c.values.size(); ++i) {
    const auto literal = glsl_literal(c.values[i]);
    if (!literal)
      return fail(ErrorCode::InvalidConstant, std::format("{} : {}", c.name, literal.error().detail));
    value += (i == 0 ? "" : ", ") + *literal;
  }
  if (c.values.size() > 1) value = std::format("{}({})", type_name(c.type), value);
  return std::format("const {} {} = {};", type_name(c.type), c.name, value);
}

Result<std::string> glsl_spec_header() {
  std::string text = std::format(
      "// Généré depuis kernels/spec.h (version {}) par ajp_kernels_glsl. Ne pas modifier :\n"
      "// les shaders reprennent ici les nombres exacts de la spécification.\n"
      "#ifndef AJP_KERNELS_SPEC_GLSL\n"
      "#define AJP_KERNELS_SPEC_GLSL\n\n"
      "#define AJP_KERNELS_SPEC_VERSION {}\n",
      kSpecVersion, kSpecVersion);
  for (const GlslConstant& c : glsl_constants()) {
    const auto line = glsl_declaration(c);
    if (!line) return std::unexpected(line.error());
    if (!c.comment.empty()) text += std::format("\n// {}\n", c.comment);
    text += *line + "\n";
  }
  text += "\n#endif  // AJP_KERNELS_SPEC_GLSL\n";
  return text;
}

}  // namespace ajp::kernels
