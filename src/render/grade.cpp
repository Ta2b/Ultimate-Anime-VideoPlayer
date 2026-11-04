#include "render/grade.h"

#include <algorithm>
#include <array>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "render/shaders.h"

namespace ajp::render {
namespace {

// Un //!PARAM et le champ de GradeParams qui le règle.
struct ParamBinding {
  std::string_view name;
  double GradeParams::* field;
};

constexpr std::array kGradeParams = {
    ParamBinding{"peak", &GradeParams::peak},
    ParamBinding{"black_lo", &GradeParams::black_lo},
    ParamBinding{"black_hi", &GradeParams::black_hi},
    ParamBinding{"paper", &GradeParams::paper},
    ParamBinding{"in_max", &GradeParams::in_max},
    ParamBinding{"shadows", &GradeParams::shadows},
    ParamBinding{"shadow_knee", &GradeParams::shadow_knee},
    ParamBinding{"area_gain", &GradeParams::area_gain},
    ParamBinding{"spec_gain", &GradeParams::spec_gain},
    ParamBinding{"vibrance", &GradeParams::vibrance},
    ParamBinding{"abl", &GradeParams::abl},
    ParamBinding{"guard", &GradeParams::guard},
    ParamBinding{"expand", &GradeParams::expand},
    ParamBinding{"abl_lo", &GradeParams::abl_lo},
    ParamBinding{"abl_hi", &GradeParams::abl_hi},
    ParamBinding{"gamut_p3", &GradeParams::gamut_p3},
    ParamBinding{"emulate", &GradeParams::emulate},
    ParamBinding{"adapt", &GradeParams::adapt},
    ParamBinding{"sdr_white", &GradeParams::sdr_white},
    ParamBinding{"em_base", &GradeParams::em_base},
    ParamBinding{"em_pow", &GradeParams::em_pow},
    ParamBinding{"em_lo", &GradeParams::em_lo},
    ParamBinding{"em_hi", &GradeParams::em_hi},
    ParamBinding{"em_sat", &GradeParams::em_sat},
    ParamBinding{"sdr_out", &GradeParams::sdr_out},
    ParamBinding{"sdr_tm", &GradeParams::sdr_tm},
    ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
    ParamBinding{"punch", &GradeParams::punch},
    ParamBinding{"chroma", &GradeParams::chroma},
    ParamBinding{"pivot", &GradeParams::pivot},
    ParamBinding{"ad_on", &GradeParams::ad_on},
    ParamBinding{"ad_lo", &GradeParams::ad_lo},
    ParamBinding{"ad_hi", &GradeParams::ad_hi},
    ParamBinding{"ad_shadows", &GradeParams::ad_shadows},
    ParamBinding{"ad_vibrance", &GradeParams::ad_vibrance},
    ParamBinding{"ad_em_base", &GradeParams::ad_em_base},
    ParamBinding{"ad_em_pow", &GradeParams::ad_em_pow},
    ParamBinding{"ad_sky_vibrance", &GradeParams::ad_sky_vibrance},
    ParamBinding{"gl_force", &GradeParams::gl_force},
    ParamBinding{"ar_series_ref", &GradeParams::ar_series_ref},
    ParamBinding{"ar_ratio_min", &GradeParams::ar_ratio_min},
    ParamBinding{"ar_ratio_max", &GradeParams::ar_ratio_max},
    ParamBinding{"ar_bright_cap", &GradeParams::ar_bright_cap},
    ParamBinding{"ar_blue_lo", &GradeParams::ar_blue_lo},
    ParamBinding{"ar_blue_hi", &GradeParams::ar_blue_hi},
    ParamBinding{"ar_blue_gate", &GradeParams::ar_blue_gate},
    ParamBinding{"ar_chroma_lo", &GradeParams::ar_chroma_lo},
    ParamBinding{"ar_chroma_hi", &GradeParams::ar_chroma_hi},
    ParamBinding{"ar_series_lo", &GradeParams::ar_series_lo},
    ParamBinding{"ar_series_hi", &GradeParams::ar_series_hi},
    ParamBinding{"ar_glow_lo", &GradeParams::ar_glow_lo},
    ParamBinding{"ar_glow_hi", &GradeParams::ar_glow_hi},
    ParamBinding{"ar_scene_lo", &GradeParams::ar_scene_lo},
    ParamBinding{"ar_scene_hi", &GradeParams::ar_scene_hi},
    ParamBinding{"gl_shadows", &GradeParams::gl_shadows},
    ParamBinding{"gl_vibrance", &GradeParams::gl_vibrance},
    ParamBinding{"gl_em_base", &GradeParams::gl_em_base},
    ParamBinding{"gl_em_pow", &GradeParams::gl_em_pow},
    ParamBinding{"sh_on", &GradeParams::sh_on},
    ParamBinding{"sh_t", &GradeParams::sh_t},
    ParamBinding{"sh_glow", &GradeParams::sh_glow},
    ParamBinding{"sh_sky", &GradeParams::sh_sky},
    ParamBinding{"sh_series", &GradeParams::sh_series},
    ParamBinding{"fs_apl", &GradeParams::fs_apl},
    ParamBinding{"fs_bright", &GradeParams::fs_bright},
    ParamBinding{"an_lift", &GradeParams::an_lift},
    ParamBinding{"an_punch", &GradeParams::an_punch},
    ParamBinding{"an_chroma", &GradeParams::an_chroma},
    ParamBinding{"an_spec", &GradeParams::an_spec},
};

constexpr std::array kLookParams = {
    ParamBinding{"look_mix", &GradeParams::look_mix},
    ParamBinding{"look_mix2", &GradeParams::look_mix2},
    ParamBinding{"look_mix3", &GradeParams::look_mix3},
    ParamBinding{"look_mix4", &GradeParams::look_mix4},
    ParamBinding{"look_mix5", &GradeParams::look_mix5},
    ParamBinding{"look_mix6", &GradeParams::look_mix6},
    ParamBinding{"look_mix7", &GradeParams::look_mix7},
    ParamBinding{"look_mix8", &GradeParams::look_mix8},
    ParamBinding{"look_mix9", &GradeParams::look_mix9},
    ParamBinding{"look_mix10", &GradeParams::look_mix10},
    ParamBinding{"look_mix11", &GradeParams::look_mix11},
    ParamBinding{"look_mix12", &GradeParams::look_mix12},
    ParamBinding{"look_mix13", &GradeParams::look_mix13},
    ParamBinding{"look_mix14", &GradeParams::look_mix14},
    ParamBinding{"look_mix15", &GradeParams::look_mix15},
    ParamBinding{"look_white", &GradeParams::look_white},
    ParamBinding{"look_pq", &GradeParams::look_pq},
    ParamBinding{"look_clean", &GradeParams::look_clean},
    ParamBinding{"sdr_out", &GradeParams::sdr_out},
    ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
    ParamBinding{"peak", &GradeParams::peak},
};

// Sortie SDR de la méthode anime (GradeParams::sdr_tm) : la première passe met l'image SDR de la cible en PQ
// BT.2020 (son blanc à sdr_white), toutes les passes suivantes travaillent comme vers un écran HDR ; la
// dernière (anime-hdr-black.hook) la ramène au SDR.
constexpr std::string_view kSdrToPq = R"(
//!HOOK OUTPUT
//!BIND HOOKED
//!WHEN sdr_tm
//!DESC AnimeJaNai HDR anime - image SDR en PQ (sortie SDR)

vec4 hook() {
    vec4 px = HOOKED_tex(HOOKED_pos);
    const mat3 to2020 = mat3(0.6274040, 0.0690970, 0.0163916,
                             0.3292820, 0.9195400, 0.0880132,
                             0.0433136, 0.0113612, 0.8955950);
    vec3 n = to2020 * (pow(clamp(px.rgb, 0.0, 1.0), vec3(sdr_gamma)) * sdr_white);
    vec3 q = pow(clamp(n / 10000.0, 0.0, 1.0), vec3(0.1593017578125));
    px.rgb = pow((0.8359375 + 18.8515625 * q) / (1.0 + 18.6875 * q), vec3(78.84375));
    return px;
}
)";

// Le nom du paramètre ou de la texture du `slot`-ième emplacement (0 : le premier, sans numéro).
std::string slot_name(std::string_view base, std::size_t slot) {
  return slot == 0 ? std::string(base) : std::format("{}{}", base, slot + 1);
}

// Le texte du hook du look : les emplacements (paramètres, textures, condition, chaîne) sont écrits pour
// kLookSlots LUT à partir des marques //@LOOK_...@ du fichier.
std::string look_source(std::size_t slots = kLookSlots) {
  std::string params;
  std::string binds;
  std::string when = "//!WHEN look_mix";
  std::string chain = "vec3 ajl_chain(vec3 c) {\n";
  for (std::size_t i = 0; i < slots; ++i) {
    const std::string mix = slot_name("look_mix", i);
    const std::string size = slot_name("look_size", i);
    const std::string tex = slot_name("AJG_LUT", i);
    if (i > 0)
      params += std::format(
          "//!PARAM {}\n//!DESC Intensité de la LUT {} (0 = aucune)\n//!TYPE float\n//!MINIMUM 0.0\n"
          "//!MAXIMUM 1.0\n0.0\n\n",
          mix, i + 1);
    params += std::format(
        "//!PARAM {}\n//!DESC Points par axe de la LUT {}\n//!TYPE float\n//!MINIMUM 2.0\n"
        "//!MAXIMUM 65.0\n2.0\n\n",
        size, i + 1);
    binds += std::format("//!BIND {}\n", tex);
    if (i > 0) when += std::format(" {} +", mix);
    chain += std::format(
        "    if ({0} > 0.0) {{\n        vec3 pos = (clamp(c, 0.0, 1.0) * ({1} - 1.0) + 0.5) / {1};\n"
        "        c = mix(c, clamp({2}_tex(pos).rgb, 0.0, 1.0), {0});\n    }}\n",
        mix, size, tex);
  }
  when += " 0 >";
  chain += "    return c;\n}\n";
  std::string text(shaders::anime_hdr_look);
  const auto replace = [&text](std::string_view mark, const std::string& with) {
    const std::size_t at = text.find(mark);
    if (at != std::string::npos) text.replace(at, mark.size(), with);
  };
  replace("//@LOOK_PARAMS@", params);
  if (!binds.empty()) binds.pop_back();
  replace("//@LOOK_BINDS@", binds);
  replace("//@LOOK_WHEN@", when);
  replace("//@LOOK_CHAIN@", chain);
  return text;
}

// Le look lu dans la passe du grade (GradeBody::Fused) : ses //!PARAM absents de `grade_text`, et ses
// fonctions jusqu'à ajl_look, pour un seul emplacement (la pile composée).
struct FusedLook {
  std::string params;
  std::string funcs;
};
FusedLook fused_look(std::string_view grade_text) {
  const std::string look = look_source(1);
  const std::size_t hook_at = look.find("//!HOOK OUTPUT");
  FusedLook out;
  constexpr std::string_view kParam = "//!PARAM ";
  std::size_t at = look.find(kParam);
  while (at != std::string::npos && at < hook_at) {
    const std::size_t next = look.find(kParam, at + 1);
    const std::size_t end = next == std::string::npos ? hook_at : std::min(next, hook_at);
    const std::string block = look.substr(at, end - at);
    const std::string name = block.substr(kParam.size(), block.find('\n') - kParam.size());
    if (grade_text.find(std::string(kParam) + name + "\n") == std::string_view::npos) out.params += block;
    at = next;
  }
  const std::size_t body = look.find('\n', look.find("//!DESC", hook_at)) + 1;
  out.funcs = look.substr(body, look.find("//@LOOK_END@") - body);
  return out;
}

// Le corps fusionné avec ses fonctions (anime-hdr-funcs.glsl) à la place de la marque //@AJG_FUNCS@ ; `look`
// : le look lu dans la même passe, après l'encodage (GradeBody::Fused), sinon sa passe à part.
// Le texte de anime-hdr-black.hook entre `from` et la marque `end` : les fonctions du vrai noir ou de KWin.
std::string black_funcs(std::string_view from, std::string_view end) {
  const std::string_view text = shaders::anime_hdr_black;
  const std::size_t at = text.find(from);
  return std::string(text.substr(at, text.find(end) - at));
}

std::string with_funcs(std::string_view body, const FusedLook* look) {
  std::string text(body);
  const bool fused = look != nullptr;
  // GradeBody::FusedSplit : ni le max local du vrai noir dans le grade, ni rien de ce qu'il lit.
  if (!fused) {
    constexpr std::string_view kBegin = "//@AJG_LOCALMAX_BEGIN@";
    constexpr std::string_view kEnd = "//@AJG_LOCALMAX_END@";
    const std::size_t from = text.find(kBegin);
    const std::size_t to = text.find(kEnd);
    if (from != std::string::npos && to != std::string::npos) text.erase(from, to + kEnd.size() - from);
  }
  const auto replace = [&text](std::string_view mark, std::string_view with) {
    for (std::size_t at = text.find(mark); at != std::string::npos; at = text.find(mark, at + with.size()))
      text.replace(at, mark.size(), with);
  };
  replace("//@AJG_FUNCS@", shaders::anime_hdr_funcs);
  replace("//@AJG_LOOK_BINDS@", fused ? "//!BIND AJG_LUT" : "");
  replace("//@AJG_LOOK_FUNCS@", fused ? look->funcs : std::string());
  // La passe du grade lit aussi le max local du vrai noir.
  replace("//@AJG_GRADE_BINDS@", fused ? "//!BIND AJG_LUT\n//!BIND AJG_LOCALMAX" : "");
  replace("//@AJG_BLACK_FUNCS@", fused ? black_funcs("float ajb_pq_to_nits", "//@BLACK_END@") +
                                             black_funcs("const mat3 AJK_TO_LMS", "//@KWIN_END@")
                                       : std::string());
  replace("//@AJG_LOOK@", fused ? "if (look_mix > 0.0) px.rgb = ajl_look(px.rgb);" : "");
  replace("//@AJG_LOOK_E@", fused ? "if (look_mix > 0.0) e = ajl_look(e);" : "");
  replace("//@AJG_BLACK@", fused ? "if (black_hi > black_lo) px.rgb = ajb_black(px.rgb, "
                                   "AJG_LOCALMAX_tex(AJG_LOCALMAX_pos).x);\n    if (kwin_on > 0.5) px.rgb = "
                                   "ajk_kwin(px.rgb);"
                                 : "");
  return text;
}

// Les //!PARAM du look lus dans la passe du grade fusionné (l'emplacement unique de la pile composée).
constexpr std::array kFusedLookParams = {
    ParamBinding{"look_mix", &GradeParams::look_mix},
    ParamBinding{"look_white", &GradeParams::look_white},
    ParamBinding{"look_pq", &GradeParams::look_pq},
};

constexpr std::array kBlackParams = {
    ParamBinding{"black_lo", &GradeParams::black_lo}, ParamBinding{"black_hi", &GradeParams::black_hi},
    ParamBinding{"sdr_out", &GradeParams::sdr_out},   ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
    ParamBinding{"peak", &GradeParams::peak},         ParamBinding{"sdr_tm", &GradeParams::sdr_tm},
    ParamBinding{"tm_white", &GradeParams::tm_white}, ParamBinding{"tm_top", &GradeParams::tm_top},
};

pl_var_data* find_param(const pl_hook& hook, std::string_view name) {
  for (int i = 0; i < hook.num_parameters; ++i) {
    const pl_hook_par& par = hook.parameters[i];
    if (par.type == PL_VAR_FLOAT && name == par.name) return par.data;
  }
  return nullptr;
}

struct ParsedHook {
  const pl_hook* hook = nullptr;
  std::vector<std::pair<pl_var_data*, double GradeParams::*>> params;
};

std::expected<ParsedHook, Refusal> parse(pl_gpu gpu, std::string_view text, std::string_view file,
                                         std::span<const ParamBinding> bindings,
                                         std::span<const ParamBinding> more = {}) {
  ParsedHook out;
  out.hook = pl_mpv_user_shader_parse(gpu, text.data(), text.size());
  if (out.hook == nullptr) {
    return std::unexpected(
        Refusal{RefusalCode::VendorCallFailed, std::format("pl_mpv_user_shader_parse ({})", file)});
  }
  std::vector<ParamBinding> all(bindings.begin(), bindings.end());
  all.insert(all.end(), more.begin(), more.end());
  for (const ParamBinding& binding : all) {
    pl_var_data* data = find_param(*out.hook, binding.name);
    if (data == nullptr) {
      pl_mpv_user_shader_destroy(&out.hook);
      return std::unexpected(
          Refusal{RefusalCode::FieldMissing, std::format("//!PARAM {} absent de {}", binding.name, file)});
    }
    out.params.emplace_back(data, binding.field);
  }
  return out;
}

// L'identité 2x2x2 d'un emplacement sans LUT.
const CubeLut& neutral_lut() {
  static const CubeLut kNeutral = [] {
    CubeLut neutral;
    neutral.size = 2;
    neutral.rgb = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
    return neutral;
  }();
  return kNeutral;
}

// Le hook du look : le bloc de la texture de la LUT (ou d'une identité 2x2x2), puis le texte embarqué.
std::expected<ParsedHook, Refusal> parse_look(pl_gpu gpu,
                                              const std::array<const CubeLut*, kLookSlots>& luts) {
  const CubeLut& kNeutral = neutral_lut();
  // Le corps d'une texture court jusqu'à la prochaine directive //! : le bloc va donc après le texte du hook.
  std::string text = look_source();
  text += '\n';
  for (std::size_t i = 0; i < kLookSlots; ++i)
    text += lut_texture_block(luts[i] != nullptr ? *luts[i] : kNeutral, slot_name("AJG_LUT", i));
  auto parsed = parse(gpu, text, "anime-hdr-look.hook", kLookParams);
  if (!parsed) return parsed;
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    const std::string size = slot_name("look_size", i);
    if (find_param(*parsed->hook, size) == nullptr) {
      pl_mpv_user_shader_destroy(&parsed->hook);
      return std::unexpected(
          Refusal{RefusalCode::FieldMissing, std::format("//!PARAM {} absent de anime-hdr-look.hook", size)});
    }
  }
  return parsed;
}

}  // namespace

// Le hook du grade fusionné avec la texture du look (la pile composée, ou l'identité) derrière son texte.
std::expected<ParsedHook, Refusal> parse_fused(pl_gpu gpu, const std::string& text, const CubeLut& lut) {
  std::string full = text;
  full += '\n';
  full += lut_texture_block(lut, "AJG_LUT");
  auto parsed = parse(gpu, full, "anime-hdr-fused", kGradeParams, kFusedLookParams);
  if (!parsed) return parsed;
  if (find_param(*parsed->hook, "look_size") == nullptr) {
    pl_mpv_user_shader_destroy(&parsed->hook);
    return std::unexpected(
        Refusal{RefusalCode::FieldMissing, "//!PARAM look_size absent de anime-hdr-fused"});
  }
  return parsed;
}

struct Grade::Impl {
  GradeBody body = GradeBody::Fused;
  std::string grade_text;  // Fused : le texte du grade sans la texture du look, pour le refaire avec la pile
  float video_fps = 24.0F;
  std::optional<Grade::KwinMatch> kwin;  // set_kwin_match, réécrit dans un hook refait
  void put_kwin();
  std::size_t hook_count = 3;
  ParsedHook grade;
  ParsedHook look;
  ParsedHook black;
  const pl_hook* calibration = nullptr;
  std::array<const pl_hook*, 3> hooks{};
  GradeParams params;
  pl_gpu gpu = nullptr;
  // Les LUT du look (copies : celles de set_looks ne vivent que le temps de l'appel), et l'intensité de
  // chacune pour laquelle la pile a été composée en une LUT (composed) ; rien de composé : jamais.
  std::array<std::optional<CubeLut>, kLookSlots> sources;
  std::optional<std::array<double, kLookSlots>> composed_for;
  bool composed = false;  // la texture du look est la pile composée (sinon une identité)
  CubeLut stack;          // la pile composée lue par le hook quand composed
  bool skin_mask = true;

  // La pile composée pour les intensités de `wanted`, refaite si elles changent ; le hook du look est
  // alors reconstruit, et ses paramètres sont à réécrire.
  void compose(const GradeParams& wanted);

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    pl_mpv_user_shader_destroy(&grade.hook);
    pl_mpv_user_shader_destroy(&look.hook);
    pl_mpv_user_shader_destroy(&black.hook);
    pl_mpv_user_shader_destroy(&calibration);
  }
};

constexpr std::array<double GradeParams::*, kLookSlots> kLookMixes{
    &GradeParams::look_mix,   &GradeParams::look_mix2,  &GradeParams::look_mix3,  &GradeParams::look_mix4,
    &GradeParams::look_mix5,  &GradeParams::look_mix6,  &GradeParams::look_mix7,  &GradeParams::look_mix8,
    &GradeParams::look_mix9,  &GradeParams::look_mix10, &GradeParams::look_mix11, &GradeParams::look_mix12,
    &GradeParams::look_mix13, &GradeParams::look_mix14, &GradeParams::look_mix15};

void Grade::Impl::compose(const GradeParams& wanted) {
  std::array<double, kLookSlots> mixes{};
  std::array<const CubeLut*, kLookSlots> luts{};
  bool active = false;
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    luts[i] = sources[i].has_value() ? &sources[i].value() : nullptr;
    mixes[i] = luts[i] != nullptr ? wanted.*kLookMixes[i] : 0.0;
    active = active || mixes[i] > 0.0;
  }
  if (composed_for == mixes) return;
  composed_for = mixes;
  stack = active ? compose_luts(luts, mixes) : CubeLut{};
  std::array<const CubeLut*, kLookSlots> slots{};
  if (active) slots[0] = &stack;
  if (body == GradeBody::Fused) {
    // La pile se lit dans la passe du grade : son hook est refait avec la nouvelle texture.
    auto rebuilt = parse_fused(gpu, grade_text, active ? stack : neutral_lut());
    if (!rebuilt) {
      composed = false;
      return;
    }
    pl_mpv_user_shader_destroy(&grade.hook);
    grade = std::move(*rebuilt);
    hooks[0] = grade.hook;
    if (pl_var_data* data = find_param(*grade.hook, "video_fps")) data->f = video_fps;
    put_kwin();
    composed = active;
    return;
  }
  auto rebuilt = parse_look(gpu, slots);
  if (!rebuilt) {  // la même texture se lisait déjà à set_looks : improbable ; l'image passe alors sans look
    composed = false;
    return;
  }
  pl_mpv_user_shader_destroy(&look.hook);
  look = std::move(*rebuilt);
  hooks[1] = look.hook;
  composed = active;
}

std::expected<Grade, Refusal> Grade::create(pl_gpu gpu, GradeBody body, bool skin_mask) {
  auto impl = std::make_unique<Impl>();
  impl->gpu = gpu;
  impl->body = body;
  impl->skin_mask = skin_mask;
  const bool legacy = body == GradeBody::Legacy;
  std::string text(shaders::anime_hdr_params);
  const bool anime = body == GradeBody::Anime || body == GradeBody::AnimeGuided;
  if (anime) {
    // Les deux premières passes (carte locale, zone) du grade en trois passes, puis la passe « anime ».
    const std::string_view fused = shaders::anime_hdr_fused;
    text += kSdrToPq;
    text += fused.substr(0, fused.find("//@AJG_LOCALMAX_BEGIN@"));
    text += shaders::anime_hdr_anime;
    if (body == GradeBody::AnimeGuided) {
      constexpr std::string_view kLocal = "float an_local(float y_pq, float plain) {";
      if (const auto at = text.find(kLocal); at != std::string::npos)
        text.insert(
            at + kLocal.size(),
            "\n    return an_local_guided(y_pq, plain);\n}\nfloat an_local_plain(float y_pq, float plain) {");
    }
  } else if (legacy) {
    text += shaders::anime_hdr_grade;
  } else if (body == GradeBody::Fused) {
    const FusedLook look = fused_look(text);
    text += look.params;
    text += with_funcs(shaders::anime_hdr_fused, &look);
  } else {
    text += with_funcs(shaders::anime_hdr_fused, nullptr);
  }
  if (!skin_mask) {
    // Le masque peau (damier sur le grain d'une peau pâle, mesuré à l'écran) est coupé sauf --skin-mask.
    constexpr std::string_view kHead = "float skin_area(vec2 pos) {";
    if (const auto at = text.find(kHead); at != std::string::npos)
      text.insert(at + kHead.size(), "\n    return 0.0;\n}\nfloat skin_area_off(vec2 pos) {");
  }
  auto grade =
      body == GradeBody::Fused
          ? parse_fused(gpu, text, neutral_lut())
          : parse(gpu, text, legacy ? "anime-hdr-grade" : (anime ? "anime-hdr-anime" : "anime-hdr-fused"),
                  kGradeParams);
  if (!grade) return std::unexpected(grade.error());
  if (body == GradeBody::Fused) impl->grade_text = std::move(text);
  impl->grade = std::move(*grade);
  auto look = parse_look(gpu, {});
  if (!look) return std::unexpected(look.error());
  impl->look = std::move(*look);
  auto black = parse(gpu, shaders::anime_hdr_black, "anime-hdr-black.hook", kBlackParams);
  if (!black) return std::unexpected(black.error());
  impl->black = std::move(*black);
  auto calibration = parse(gpu, shaders::anime_hdr_calibration, "anime-hdr-calibration.hook", {});
  if (!calibration) return std::unexpected(calibration.error());
  impl->calibration = calibration->hook;
  // Fused : le look, le vrai noir et KWin sont dans la passe du grade, leurs passes à part ne sont plus
  // posées.
  if (body == GradeBody::Fused) {
    impl->hooks = {impl->grade.hook, nullptr, nullptr};
    impl->hook_count = 1;
  } else {
    impl->hooks = {impl->grade.hook, impl->look.hook, impl->black.hook};
  }
  Grade out(std::move(impl));
  out.set_params(GradeParams{});
  return out;
}

std::expected<void, Refusal> Grade::set_looks(const std::array<const CubeLut*, kLookSlots>& luts) {
  // Chaque LUT doit se lire seule (le refus nomme la fautive) ; le hook, lui, ne lit que la pile composée.
  auto look = parse_look(impl_->gpu, luts);
  if (!look) return std::unexpected(look.error());
  pl_mpv_user_shader_destroy(&look->hook);  // la lecture seule importe : le hook posé lit la pile composée
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    if (luts[i] != nullptr)
      impl_->sources[i] = *luts[i];
    else
      impl_->sources[i].reset();
  }
  impl_->composed_for.reset();
  set_params(impl_->params);
  return {};
}

Grade::Grade(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Grade::Grade(Grade&&) noexcept = default;
Grade& Grade::operator=(Grade&&) noexcept = default;
Grade::~Grade() = default;

void Grade::set_params(const GradeParams& params) {
  impl_->params = params;
  impl_->compose(params);
  for (const ParsedHook* hook : {&impl_->grade, &impl_->look, &impl_->black}) {
    for (const auto& [data, field] : hook->params) data->f = static_cast<float>(params.*field);
  }
  // Le hook ne lit que la pile composée, en premier emplacement, à pleine intensité (les intensités y sont
  // déjà) ; les autres emplacements restent à 0, sautés.
  const pl_hook& look_hook = impl_->body == GradeBody::Fused ? *impl_->grade.hook : *impl_->look.hook;
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    const bool stack = i == 0 && impl_->composed;
    if (pl_var_data* size = find_param(look_hook, slot_name("look_size", i)))
      size->f = static_cast<float>(stack ? kLutMaxSize : 2);
    if (pl_var_data* mix = find_param(look_hook, slot_name("look_mix", i))) mix->f = stack ? 1.0F : 0.0F;
  }
}

void Grade::set_video_fps(double fps) {
  impl_->video_fps = static_cast<float>(std::clamp(fps, 1.0, 1000.0));
  if (pl_var_data* data = find_param(*impl_->grade.hook, "video_fps")) data->f = impl_->video_fps;
}

void Grade::set_kwin_match(std::optional<KwinMatch> match) {
  impl_->kwin = match;
  impl_->put_kwin();
}

void Grade::Impl::put_kwin() {
  const auto put = [this](std::string_view name, double value) {
    for (const pl_hook* hook : {black.hook, grade.hook})
      if (pl_var_data* data = find_param(*hook, name)) data->f = static_cast<float>(value);
  };
  put("kwin_on", kwin ? 1.0 : 0.0);
  if (kwin) {
    put("kwin_ref", std::clamp(kwin->reference, 50.0, 1000.0));
    put("kwin_peak", std::clamp(kwin->peak, 100.0, 10000.0));
  }
}

std::optional<float> Grade::param(std::string_view name) const {
  if (const pl_var_data* data = find_param(*impl_->grade.hook, name)) return data->f;
  return std::nullopt;
}

std::optional<float> Grade::param_for(std::string_view name, const GradeParams& params,
                                      std::optional<KwinMatch> kwin) const {
  const pl_var_data* data = find_param(*impl_->grade.hook, name);
  if (data == nullptr) return std::nullopt;
  if (name == "kwin_on") return kwin ? 1.0F : 0.0F;
  if (name == "kwin_ref")
    return kwin ? static_cast<float>(std::clamp(kwin->reference, 50.0, 1000.0)) : data->f;
  if (name == "kwin_peak") return kwin ? static_cast<float>(std::clamp(kwin->peak, 100.0, 10000.0)) : data->f;
  if (impl_->body == GradeBody::Fused && name == "look_mix") return look_stack_for(params) ? 1.0F : 0.0F;
  for (const auto& [bound, field] : impl_->grade.params) {
    if (bound == data) return static_cast<float>(params.*field);
  }
  return data->f;
}

std::optional<CubeLut> Grade::look_stack_for(const GradeParams& params) const {
  std::array<double, kLookSlots> mixes{};
  std::array<const CubeLut*, kLookSlots> luts{};
  bool active = false;
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    luts[i] = impl_->sources[i].has_value() ? &impl_->sources[i].value() : nullptr;
    mixes[i] = luts[i] != nullptr ? params.*kLookMixes[i] : 0.0;
    active = active || mixes[i] > 0.0;
  }
  if (!active) return std::nullopt;
  if (impl_->composed && impl_->composed_for == mixes) return impl_->stack;
  return compose_luts(luts, mixes);
}

const CubeLut* Grade::look_stack() const {
  return impl_->composed ? &impl_->stack : nullptr;
}

bool Grade::skin_mask() const {
  return impl_->skin_mask;
}

const GradeParams& Grade::params() const {
  return impl_->params;
}

std::span<const pl_hook* const> Grade::hooks() const {
  return std::span<const pl_hook* const>(impl_->hooks.data(), impl_->hook_count);
}

const pl_hook* Grade::calibration() const {
  return impl_->calibration;
}

}  // namespace ajp::render
