#include "render/grade.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "render/shaders.h"

namespace ajp::render {
namespace {

// Un //!PARAM et le champ de GradeParams qui le règle : un champ simple, ou l'intensité d'une LUT de la pile
// (look_mixes[slot]).
struct ParamBinding {
  std::string_view name;
  double GradeParams::* field = nullptr;
  std::size_t slot = 0;
};

double value_of(const GradeParams& params, const ParamBinding& binding) {
  return binding.field != nullptr ? params.*binding.field : params.look_mixes.at(binding.slot);
}

constexpr std::array kGradeParams = {
    ParamBinding{"peak", &GradeParams::peak},
    ParamBinding{"black_lo", &GradeParams::black_lo},
    ParamBinding{"black_hi", &GradeParams::black_hi},
    ParamBinding{"paper", &GradeParams::paper},
    ParamBinding{"shadows", &GradeParams::shadows},
    ParamBinding{"shadow_knee", &GradeParams::shadow_knee},
    ParamBinding{"spec_gain", &GradeParams::spec_gain},
    ParamBinding{"adapt", &GradeParams::adapt},
    ParamBinding{"sdr_white", &GradeParams::sdr_white},
    ParamBinding{"sdr_out", &GradeParams::sdr_out},
    ParamBinding{"sdr_tm", &GradeParams::sdr_tm},
    ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
    ParamBinding{"chroma", &GradeParams::chroma},
    ParamBinding{"pivot", &GradeParams::pivot},
    ParamBinding{"fs_apl", &GradeParams::fs_apl},
    ParamBinding{"fs_series", &GradeParams::fs_series},
    ParamBinding{"an_lift", &GradeParams::an_lift},
    ParamBinding{"an_punch", &GradeParams::an_punch},
    ParamBinding{"an_chroma", &GradeParams::an_chroma},
    ParamBinding{"an_spec", &GradeParams::an_spec},
};

constexpr std::array kLookParams = {
    ParamBinding{.name = "look_mix", .slot = 0},    ParamBinding{.name = "look_mix2", .slot = 1},
    ParamBinding{.name = "look_mix3", .slot = 2},   ParamBinding{.name = "look_mix4", .slot = 3},
    ParamBinding{.name = "look_mix5", .slot = 4},   ParamBinding{.name = "look_mix6", .slot = 5},
    ParamBinding{.name = "look_mix7", .slot = 6},   ParamBinding{.name = "look_mix8", .slot = 7},
    ParamBinding{.name = "look_mix9", .slot = 8},   ParamBinding{.name = "look_mix10", .slot = 9},
    ParamBinding{.name = "look_mix11", .slot = 10}, ParamBinding{.name = "look_mix12", .slot = 11},
    ParamBinding{.name = "look_mix13", .slot = 12}, ParamBinding{.name = "look_mix14", .slot = 13},
    ParamBinding{.name = "look_mix15", .slot = 14}, ParamBinding{"look_white", &GradeParams::look_white},
    ParamBinding{"look_pq", &GradeParams::look_pq}, ParamBinding{"look_clean", &GradeParams::look_clean},
    ParamBinding{"sdr_out", &GradeParams::sdr_out}, ParamBinding{"sdr_gamma", &GradeParams::sdr_gamma},
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
  std::vector<std::pair<pl_var_data*, ParamBinding>> params;
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
    out.params.emplace_back(data, binding);
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

// Le texte du hook du look : le texte embarqué, puis le bloc de la texture de chaque LUT (ou d'une identité
// 2x2x2). Sans GPU : 4,4 Mo et une dizaine de millisecondes pour une pile de 65 points.
std::string look_text(const std::array<const CubeLut*, kLookSlots>& luts) {
  const CubeLut& kNeutral = neutral_lut();
  // Le corps d'une texture court jusqu'à la prochaine directive //! : le bloc va donc après le texte du hook.
  std::string text = look_source();
  text += '\n';
  for (std::size_t i = 0; i < kLookSlots; ++i)
    text += lut_texture_block(luts[i] != nullptr ? *luts[i] : kNeutral, slot_name("AJG_LUT", i));
  return text;
}

// Le hook du look lu par libplacebo, qui crée la texture de chaque LUT sur le pl_gpu.
std::expected<ParsedHook, Refusal> parse_look_text(pl_gpu gpu, const std::string& text) {
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

std::expected<ParsedHook, Refusal> parse_look(pl_gpu gpu,
                                              const std::array<const CubeLut*, kLookSlots>& luts) {
  return parse_look_text(gpu, look_text(luts));
}

}  // namespace

struct Grade::Impl {
  float video_fps = 24.0F;
  std::optional<Grade::KwinMatch> kwin;  // set_kwin_match, réécrit dans un hook refait
  void put_kwin();
  ParsedHook grade;
  ParsedHook look;
  ParsedHook black;
  const pl_hook* calibration = nullptr;
  std::array<const pl_hook*, 3> hooks{};
  GradeParams params;
  pl_gpu gpu = nullptr;
  // Les LUT du look (copies : celles de set_looks ne vivent que le temps de l'appel), et l'intensité de
  // chacune pour laquelle la pile a été composée en une LUT (composed) ; rien de composé : jamais.
  std::array<std::shared_ptr<const CubeLut>, kLookSlots> sources;
  std::optional<std::array<double, kLookSlots>> composed_for;
  bool composed = false;  // la texture du look est la pile composée (sinon une identité)
  CubeLut stack;          // la pile composée lue par le hook quand composed

  // La pile composée pour les intensités de `wanted`, refaite si elles changent ; le hook du look est
  // alors reconstruit, et ses paramètres sont à réécrire.
  void compose(const GradeParams& wanted);
  void write_params();

  // P4.9 : la composition en tâche de fond (set_background). Le thread ne garde que la dernière demande.
  struct Job {
    std::array<double, kLookSlots> mixes{};
    std::array<std::shared_ptr<const CubeLut>, kLookSlots> luts;
  };
  struct Composed {
    std::array<double, kLookSlots> mixes{};
    bool active = false;
    CubeLut stack;
    std::expected<ParsedHook, Refusal> hook;
  };
  bool background = false;
  std::optional<std::array<double, kLookSlots>> requested_for;
  std::mutex job_mutex;
  std::mutex gpu_mutex;  // tenu par le thread pendant qu'il lit un hook sur le pl_gpu (Grade::gpu_guard)
  std::condition_variable_any job_ready;
  std::optional<Job> job;
  std::optional<Composed> done;
  std::jthread worker;  // arrêté avant la destruction des hooks (~Impl)
  void run(const std::stop_token& stop);

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    worker = {};  // demande l'arrêt et attend le thread
    if (done && done->hook) pl_mpv_user_shader_destroy(&done->hook->hook);
    pl_mpv_user_shader_destroy(&grade.hook);
    pl_mpv_user_shader_destroy(&look.hook);
    pl_mpv_user_shader_destroy(&black.hook);
    pl_mpv_user_shader_destroy(&calibration);
  }
};

void Grade::Impl::run(const std::stop_token& stop) {
  while (true) {
    Job next;
    {
      std::unique_lock lock(job_mutex);
      if (!job_ready.wait(lock, stop, [this] { return job.has_value(); }) || !job) return;
      next = std::move(*job);
      job.reset();
    }
    Composed out{.mixes = next.mixes};
    std::array<const CubeLut*, kLookSlots> luts{};
    for (std::size_t i = 0; i < kLookSlots; ++i) {
      luts[i] = next.luts[i].get();
      out.active = out.active || (luts[i] != nullptr && next.mixes[i] > 0.0);
    }
    if (out.active) out.stack = compose_luts(luts, next.mixes);
    std::array<const CubeLut*, kLookSlots> slots{};
    if (out.active) slots[0] = &out.stack;
    const std::string text = look_text(slots);  // hors du verrou : le texte ne touche pas le GPU
    {
      const std::scoped_lock on_gpu(gpu_mutex);  // gpu_guard : jamais pendant que la fenêtre présente
      out.hook = parse_look_text(gpu, text);
    }
    const std::scoped_lock lock(job_mutex);
    if (done && done->hook) pl_mpv_user_shader_destroy(&done->hook->hook);  // jamais adopté, dépassé
    done = std::move(out);
  }
}

void Grade::Impl::compose(const GradeParams& wanted) {
  std::array<double, kLookSlots> mixes{};
  std::array<const CubeLut*, kLookSlots> luts{};
  bool active = false;
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    luts[i] = sources[i].get();
    mixes[i] = luts[i] != nullptr ? wanted.look_mixes[i] : 0.0;
    active = active || mixes[i] > 0.0;
  }
  if (background) {
    if (requested_for ? *requested_for == mixes : composed_for == mixes) return;
    requested_for = mixes;
    {
      const std::scoped_lock lock(job_mutex);
      job = Job{.mixes = mixes, .luts = sources};
    }
    if (!worker.joinable()) worker = std::jthread([this](const std::stop_token& stop) { run(stop); });
    job_ready.notify_one();
    return;
  }
  if (composed_for == mixes) return;
  composed_for = mixes;
  stack = active ? compose_luts(luts, mixes) : CubeLut{};
  std::array<const CubeLut*, kLookSlots> slots{};
  if (active) slots[0] = &stack;
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

std::expected<Grade, Refusal> Grade::create(pl_gpu gpu) {
  auto impl = std::make_unique<Impl>();
  impl->gpu = gpu;
  // Les paramètres, la mise en PQ d'une sortie SDR, puis les trois passes (carte locale, zone, passe anime).
  std::string text(shaders::anime_hdr_params);
  text += kSdrToPq;
  text += shaders::anime_hdr_anime;
  auto grade = parse(gpu, text, "anime-hdr-anime", kGradeParams);
  if (!grade) return std::unexpected(grade.error());
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
  impl->hooks = {impl->grade.hook, impl->look.hook, impl->black.hook};
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
      impl_->sources[i] = std::make_shared<const CubeLut>(*luts[i]);
    else
      impl_->sources[i].reset();
  }
  impl_->composed_for.reset();
  impl_->requested_for.reset();
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
  impl_->write_params();
}

void Grade::Impl::write_params() {
  for (const ParsedHook* hook : {&grade, &look, &black}) {
    for (const auto& [data, binding] : hook->params) data->f = static_cast<float>(value_of(params, binding));
  }
  // Le hook ne lit que la pile composée, en premier emplacement, à pleine intensité (les intensités y sont
  // déjà) ; les autres emplacements restent à 0, sautés.
  const pl_hook& look_hook = *look.hook;
  for (std::size_t i = 0; i < kLookSlots; ++i) {
    const bool is_stack = i == 0 && composed;
    if (pl_var_data* size = find_param(look_hook, slot_name("look_size", i)))
      size->f = static_cast<float>(is_stack ? kLutMaxSize : 2);
    if (pl_var_data* mix = find_param(look_hook, slot_name("look_mix", i))) mix->f = is_stack ? 1.0F : 0.0F;
  }
}

void Grade::set_background(bool on) {
  impl_->background = on;
}

std::unique_lock<std::mutex> Grade::gpu_guard() {
  return std::unique_lock(impl_->gpu_mutex);
}

bool Grade::adopt() {
  std::optional<Impl::Composed> ready;
  {
    const std::scoped_lock lock(impl_->job_mutex);
    ready = std::exchange(impl_->done, std::nullopt);
  }
  if (!ready) return false;
  Impl& s = *impl_;
  if (s.requested_for == ready->mixes) s.requested_for.reset();
  s.composed_for = ready->mixes;
  if (!ready->hook) {  // la même texture se lisait déjà à set_looks : improbable ; l'image passe sans look
    s.composed = false;
  } else {
    pl_mpv_user_shader_destroy(&s.look.hook);
    s.look = std::move(*ready->hook);
    s.hooks[1] = s.look.hook;
    s.stack = std::move(ready->stack);
    s.composed = ready->active;
  }
  s.write_params();
  return true;
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

const CubeLut* Grade::look_stack() const {
  return impl_->composed ? &impl_->stack : nullptr;
}

const GradeParams& Grade::params() const {
  return impl_->params;
}

std::span<const pl_hook* const> Grade::hooks() const {
  return std::span<const pl_hook* const>(impl_->hooks.data(), impl_->hooks.size());
}

const pl_hook* Grade::calibration() const {
  return impl_->calibration;
}

}  // namespace ajp::render
