// FrameExport, le grade HDR anime de l'export : un grade à lui (l'écran garde le sien), rendu dans une
// image RVB à la taille de sortie, puis cette image encodée dans les plans YUV de l'encodeur.
//
// Pourquoi deux rendus : libplacebo lance les hooks OUTPUT après pl_shader_encode_color (pass_output_target,
// renderer.c). Sur l'écran, la cible est en RVB et l'encodage ne change rien ; sur une cible YCbCr, les passes
// du grade recevaient Y, Cb et Cr et les traitaient comme R, V et B. Un fond gris-bleu (53, 65, 80) sortait
// brun-orangé, Cb à 303 sur 10 bits au lieu de plus de 512, des cheveux lavande sortaient magenta.
#include <format>
#include <utility>

#include "app/play/placebo_export.h"
#include "app/play/placebo_impl.h"
#include "core/log.h"
#include "render/display_model.h"

namespace kirari::app::play {

// Le grade de l'export : ses paramètres, sa pile de LUT, la cadence exportée, et son image RVB.
// En PQ, la dernière passe compresse vers le pic du fichier comme la lecture vers celui de l'écran, au blanc
// de référence du contenu (203 nits : le PQ passe tel quel, comme sous Windows).
std::expected<void, Refusal> FrameExport::open_grade(pl_gpu gpu) {
  if (!setup_.grade) return {};
  auto grade = render::Grade::create(gpu);
  if (!grade) return std::unexpected(std::move(grade.error()));
  if (auto set = grade->set_looks(setup_.looks); !set) return std::unexpected(std::move(set.error()));
  const render::GradeParams& params = *setup_.grade;
  grade->set_params(params);
  grade->set_video_fps(setup_.video_fps);
  if (setup_.grade_pq) {
    grade->set_kwin_match(
        render::Grade::KwinMatch{.reference = render::kContentReferenceNits, .peak = params.peak});
    graded_color_ = pl_color_space_hdr10;
    graded_color_.hdr.max_luma = static_cast<float>(params.peak);
    graded_color_.hdr.max_cll = static_cast<float>(params.peak);
  } else {
    graded_color_ = pl_color_space_bt709;  // BT.1886 : le grade SDR y prend l'exposant 2,4
  }
  // 16 bits normalisés : bornés à [0, 1] comme l'image d'écran que le grade vise, et bien plus fins que les
  // 10 bits du fichier. En flottant 16 bits sinon (un GPU sans rendu dans rgba16).
  constexpr auto kNeeded = static_cast<pl_fmt_caps>(PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_SAMPLEABLE);
  pl_fmt format = pl_find_named_fmt(gpu, "rgba16");
  if (format == nullptr || (format->caps & kNeeded) != kNeeded) format = pl_find_named_fmt(gpu, "rgba16f");
  if (format == nullptr || (format->caps & kNeeded) != kNeeded)
    return refuse(RefusalCode::FeatureUnavailable, "export : ni rgba16 ni rgba16f rendables pour le grade");
  pl_tex_params tex{};
  tex.w = setup_.width;
  tex.h = setup_.height;
  tex.format = format;
  tex.renderable = true;
  tex.sampleable = true;
  graded_ = pl_tex_create(gpu, &tex);
  if (graded_ == nullptr) return refuse(RefusalCode::GpuMemoryExhausted, "export : image du grade");
  grade_.emplace(std::move(*grade));
  log_channel("export")->info("export : grade HDR anime, sortie {} (pic {:.0f} nits), image {}",
                              setup_.grade_pq ? "HDR10" : "SDR BT.709", params.peak, format->name);
  return {};
}

bool FrameExport::render_graded(const pl_frame& image, const pl_frame& target, const pl_render_params& params) {
  pl_frame rgb{};
  rgb.num_planes = 1;
  rgb.planes[0].texture = graded_;
  rgb.planes[0].components = 3;
  for (int c = 0; c < 3; ++c) rgb.planes[0].component_mapping[c] = c;
  rgb.repr = pl_color_repr_rgb;
  rgb.color = graded_color_;
  rgb.crop = target.crop;
  // Premier rendu : la mise à l'échelle, Anime4K et le grade, comme sur l'écran. Pas de tramage : l'image
  // garde 16 bits, le tramage vient au second rendu, vers la profondeur du fichier.
  pl_render_params first = params;
  first.dither_params = nullptr;
  if (!pl_render_image(renderer_, &image, &rgb, &first)) return false;
  // Second rendu : RVB vers YUV, même espace de couleur, même taille ; ni hook, ni analyse de pic (pas de
  // tone mapping entre deux espaces égaux), ni débandage. Le sous-échantillonnage de la chroma et le tramage
  // restent ceux de params.
  pl_render_params second = params;
  second.hooks = nullptr;
  second.num_hooks = 0;
  second.peak_detect_params = nullptr;
  second.deband_params = nullptr;
  return pl_render_image(renderer_, &rgb, &target, &second);
}

}  // namespace kirari::app::play
