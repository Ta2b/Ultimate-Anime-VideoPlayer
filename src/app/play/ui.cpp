#include "app/play/ui.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_video.h>
#include <imgui.h>
#if AJP_WITH_FONTCONFIG
#include <fontconfig/fontconfig.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

#include "app/play/placebo_imgui.h"
#include "app/play/placebo_screen.h"
#include "core/log.h"
#ifdef AJP_WITH_FREETYPE
#include "imgui_freetype.h"
#endif
#include "core/thread.h"

namespace ajp::app::play {
namespace {

constexpr float kFontSize = 17.0F;        // pixels à l'échelle 1, multipliés par l'échelle de l'écran
constexpr int kMaxFontBytes = 256 << 20;  // ImGui prend la taille en int

// Le fichier de police que fontconfig choisit pour pattern (« sans-serif », « sans-serif:lang=ja ») ;
// seulement TrueType ou OpenType, ce que lit ImGui. need_cjk : la police doit avoir de quoi écrire le
// japonais (あ et 日) ; fontconfig rend sinon la plus proche, parfois une police latine sans aucun
// idéogramme.
std::optional<FontFile> system_font(const char* pattern, bool need_cjk = false) {
#if AJP_WITH_FONTCONFIG
  if (FcInit() == FcFalse) return std::nullopt;
  FcPattern* wanted = FcNameParse(reinterpret_cast<const FcChar8*>(pattern));
  if (wanted == nullptr) return std::nullopt;
  FcConfigSubstitute(nullptr, wanted, FcMatchPattern);
  FcDefaultSubstitute(wanted);
  FcResult result = FcResultNoMatch;
  FcPattern* match = FcFontMatch(nullptr, wanted, &result);
  std::optional<FontFile> found;
  FcChar8* file = nullptr;
  int index = 0;
  if (match != nullptr && FcPatternGetString(match, FC_FILE, 0, &file) == FcResultMatch) {
    FcPatternGetInteger(match, FC_INDEX, 0, &index);
    FcCharSet* charset = nullptr;
    const bool covers = !need_cjk || (FcPatternGetCharSet(match, FC_CHARSET, 0, &charset) == FcResultMatch &&
                                      FcCharSetHasChar(charset, 0x3042) != FcFalse &&
                                      FcCharSetHasChar(charset, 0x65E5) != FcFalse);
    std::string path(reinterpret_cast<const char*>(file));
    std::string ext = std::filesystem::path(path).extension().string();
    std::ranges::transform(ext, ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (covers && (ext == ".ttf" || ext == ".otf" || ext == ".ttc" || ext == ".otc"))
      found = FontFile{.path = std::move(path), .index = index};
  }
  if (match != nullptr) FcPatternDestroy(match);
  FcPatternDestroy(wanted);
  return found;
#else
  static_cast<void>(pattern);
  static_cast<void>(need_cjk);
  return std::nullopt;
#endif
}

// Le fichier entier dans font.data ; false s'il ne se lit pas.
bool read_font(const std::filesystem::path& path, FontFile& font) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size == 0 || size > static_cast<std::uintmax_t>(kMaxFontBytes)) return false;
  std::ifstream file(path, std::ios::binary);
  font.data.resize(static_cast<std::size_t>(size));
  return file && file.read(reinterpret_cast<char*>(font.data.data()), static_cast<std::streamsize>(size));
}

#ifdef _WIN32
// Le dossier des polices de Windows (%WINDIR%\Fonts).
std::filesystem::path windows_fonts() {
  std::array<wchar_t, MAX_PATH> dir{};
  const UINT length = GetWindowsDirectoryW(dir.data(), static_cast<UINT>(dir.size()));
  const std::filesystem::path root = length > 0 && length < dir.size()
                                         ? std::filesystem::path(dir.data())
                                         : std::filesystem::path(L"C:\\Windows");
  return root / L"Fonts";
}

// La première de ces polices du dossier de Windows qui se lit (index : celle de la collection).
std::optional<FontFile> first_font(const std::filesystem::path& dir,
                                   std::initializer_list<std::pair<const wchar_t*, int>> files) {
  for (const auto& [name, index] : files) {
    const std::filesystem::path path = dir / name;
    const std::u8string utf8 = path.u8string();
    FontFile font{.path = std::string(utf8.begin(), utf8.end()), .index = index};
    if (read_font(path, font)) return font;
  }
  return std::nullopt;
}

// Sans fontconfig, les polices que Windows 10 et 11 installent toujours : Segoe UI (latin
// accentué, grec, cyrillique), Yu Gothic pour le japonais des titres de pistes (Meiryo ou MS
// Gothic, polices supplémentaires, à défaut), Segoe UI Symbol pour les symboles qui manquent aux
// deux (▶, filets).
std::vector<FontFile> search_fonts() {
  std::vector<FontFile> fonts;
  const std::filesystem::path dir = windows_fonts();
  std::optional<FontFile> main = first_font(dir, {{L"segoeui.ttf", 0}, {L"arial.ttf", 0}});
  if (!main) return fonts;
  fonts.push_back(std::move(*main));
  if (std::optional<FontFile> japanese = first_font(
          dir, {{L"YuGothM.ttc", 0}, {L"YuGothR.ttc", 0}, {L"meiryo.ttc", 0}, {L"msgothic.ttc", 0}}))
    fonts.push_back(std::move(*japanese));
  if (std::optional<FontFile> symbols = first_font(dir, {{L"seguisym.ttf", 0}}))
    fonts.push_back(std::move(*symbols));
  return fonts;
}
#else
// Police du bureau (sans-serif), complétée par une police japonaise pour les titres de pistes.
std::vector<FontFile> search_fonts() {
  std::vector<FontFile> fonts;
  std::optional<FontFile> main = system_font("sans-serif");
  if (!main || !read_font(std::filesystem::path(main->path), *main)) return fonts;
  fonts.push_back(std::move(*main));
  // Titres japonais (et chinois, coréens) : la police CJK que fontconfig donne pour la langue, sinon les
  // familles courantes des distributions (Fedora : google-noto-sans-cjk-fonts ; Debian : fonts-noto-cjk ;
  // Arch : noto-fonts-cjk).
  for (const char* pattern : {"sans-serif:lang=ja", "Noto Sans CJK JP", "Noto Sans JP", "Source Han Sans",
                              "IPAexGothic", "TakaoGothic", "VL Gothic", "WenQuanYi Zen Hei",
                              "Droid Sans Fallback", "sans-serif:lang=zh-cn", "sans-serif:lang=ko"}) {
    std::optional<FontFile> cjk = system_font(pattern, true);
    if (!cjk || cjk->path == fonts.front().path || !read_font(std::filesystem::path(cjk->path), *cjk))
      continue;
    log_channel("app")->info("interface : police CJK {} ({})", cjk->path, pattern);
    fonts.push_back(std::move(*cjk));
    return fonts;
  }
  log_channel("app")->warn(
      "interface : aucune police CJK trouvée, les titres japonais s'afficheront en carrés (Noto Sans CJK)");
  return fonts;
}
#endif

bool add_font(ImFontAtlas& fonts, FontFile& file, bool merge) {
  ImFontConfig config;
  config.MergeMode = merge;
  // fontconfig range dans l'index d'une police variable le numéro de l'instance nommée (bits 16 et plus) :
  // seule la face compte.
  config.FontNo = static_cast<ImU32>(std::max(file.index, 0)) & 0xFFFFU;
  config.FontDataOwnedByAtlas = false;  // file.data vit dans PlayerUi
  return fonts.AddFontFromMemoryTTF(file.data.data(), static_cast<int>(file.data.size()), kFontSize,
                                    &config) != nullptr;
}

// Les polices trouvées par FontSearch ; sinon la police vectorielle d'ImGui. ImGui 1.92 rastérise
// à la taille demandée.
void load_fonts(ImFontAtlas& atlas, std::vector<FontFile>& fonts) {
  const auto log = log_channel("app");
#ifdef AJP_WITH_FREETYPE
  // FreeType lit les polices que stb_truetype ne lit pas : OpenType CFF2 et variables (Noto Sans CJK de
  // Fedora).
  atlas.SetFontLoader(ImGuiFreeType::GetFontLoader());
#endif
  if (fonts.empty() || !add_font(atlas, fonts.front(), false)) {
    atlas.AddFontDefaultVector();
    log->info("interface : police vectorielle d'ImGui");
    return;
  }
  log->info("interface : police {}", fonts.front().path);
  for (std::size_t i = 1; i < fonts.size(); ++i) {
    if (add_font(atlas, fonts[i], true)) {
      log->info("interface : police de repli {} chargée", fonts[i].path);
    } else {
      log->warn("interface : police de repli {} refusée par le chargeur de polices (format non lu)",
                fonts[i].path);
    }
  }
}

// Thème sombre et sobre, accent bleu clair (celui de la barre de progression).
void apply_theme(ImGuiStyle& style) {
  ImGui::StyleColorsDark(&style);
  style.WindowRounding = 12.0F;
  style.PopupRounding = 10.0F;
  style.ChildRounding = 8.0F;
  style.FrameRounding = 6.0F;
  style.GrabRounding = 6.0F;
  style.TabRounding = 6.0F;
  style.ScrollbarRounding = 8.0F;
  style.WindowBorderSize = 0.0F;
  style.PopupBorderSize = 1.0F;
  style.WindowPadding = {14.0F, 12.0F};
  style.FramePadding = {10.0F, 6.0F};
  style.ItemSpacing = {10.0F, 8.0F};
  style.ItemInnerSpacing = {8.0F, 6.0F};
  const ImVec4 accent{0.42F, 0.66F, 1.0F, 1.0F};
  const auto with = [](ImVec4 c, float a) { return ImVec4(c.x, c.y, c.z, a); };
  ImVec4* colors = style.Colors;
  colors[ImGuiCol_Text] = {0.95F, 0.95F, 0.97F, 1.0F};
  colors[ImGuiCol_TextDisabled] = {0.56F, 0.57F, 0.62F, 1.0F};
  colors[ImGuiCol_WindowBg] = {0.07F, 0.07F, 0.09F, 0.97F};
  colors[ImGuiCol_PopupBg] = {0.09F, 0.09F, 0.11F, 0.96F};
  colors[ImGuiCol_Border] = {1.0F, 1.0F, 1.0F, 0.08F};
  colors[ImGuiCol_FrameBg] = {1.0F, 1.0F, 1.0F, 0.07F};
  colors[ImGuiCol_FrameBgHovered] = {1.0F, 1.0F, 1.0F, 0.12F};
  colors[ImGuiCol_FrameBgActive] = {1.0F, 1.0F, 1.0F, 0.16F};
  colors[ImGuiCol_Button] = {1.0F, 1.0F, 1.0F, 0.08F};
  colors[ImGuiCol_ButtonHovered] = {1.0F, 1.0F, 1.0F, 0.15F};
  colors[ImGuiCol_ButtonActive] = with(accent, 0.6F);
  colors[ImGuiCol_Header] = with(accent, 0.28F);
  colors[ImGuiCol_HeaderHovered] = with(accent, 0.42F);
  colors[ImGuiCol_HeaderActive] = with(accent, 0.6F);
  colors[ImGuiCol_CheckMark] = accent;
  colors[ImGuiCol_SliderGrab] = accent;
  colors[ImGuiCol_SliderGrabActive] = {0.6F, 0.78F, 1.0F, 1.0F};
  colors[ImGuiCol_Separator] = {1.0F, 1.0F, 1.0F, 0.1F};
  colors[ImGuiCol_Tab] = {1.0F, 1.0F, 1.0F, 0.06F};
  colors[ImGuiCol_TabHovered] = with(accent, 0.42F);
  colors[ImGuiCol_TabSelected] = with(accent, 0.32F);
  colors[ImGuiCol_TitleBg] = colors[ImGuiCol_WindowBg];
  colors[ImGuiCol_TitleBgActive] = colors[ImGuiCol_WindowBg];
}

int mouse_button(Uint8 button) {
  switch (button) {
    case SDL_BUTTON_LEFT:
      return ImGuiMouseButton_Left;
    case SDL_BUTTON_RIGHT:
      return ImGuiMouseButton_Right;
    case SDL_BUTTON_MIDDLE:
      return ImGuiMouseButton_Middle;
    default:
      return -1;
  }
}

// Les touches dont une saisie dans l'interface a besoin.
ImGuiKey imgui_key(SDL_Keycode key) {
  if (key >= SDLK_A && key <= SDLK_Z)
    return static_cast<ImGuiKey>(ImGuiKey_A + static_cast<int>(key - SDLK_A));
  if (key >= SDLK_0 && key <= SDLK_9)
    return static_cast<ImGuiKey>(ImGuiKey_0 + static_cast<int>(key - SDLK_0));
  switch (key) {
    case SDLK_TAB:
      return ImGuiKey_Tab;
    case SDLK_LEFT:
      return ImGuiKey_LeftArrow;
    case SDLK_RIGHT:
      return ImGuiKey_RightArrow;
    case SDLK_UP:
      return ImGuiKey_UpArrow;
    case SDLK_DOWN:
      return ImGuiKey_DownArrow;
    case SDLK_HOME:
      return ImGuiKey_Home;
    case SDLK_END:
      return ImGuiKey_End;
    case SDLK_DELETE:
      return ImGuiKey_Delete;
    case SDLK_BACKSPACE:
      return ImGuiKey_Backspace;
    case SDLK_RETURN:
      return ImGuiKey_Enter;
    case SDLK_KP_ENTER:
      return ImGuiKey_KeypadEnter;
    case SDLK_ESCAPE:
      return ImGuiKey_Escape;
    default:
      return ImGuiKey_None;
  }
}

}  // namespace

FontSearch FontSearch::start() {
  FontSearch search;
  auto thread = start_thread("polices", ThreadReport([](const Refusal& refusal) {
                               log_channel("app")->warn("polices : {}", refusal.detail);
                             }),
                             [found = search.found_.get()](const std::stop_token&, const ThreadReport&) {
                               *found = search_fonts();
                             });
  if (thread) {
    search.thread_ = std::move(*thread);
  } else {
    log_channel("app")->warn("polices : {} ; cherchées à l'ouverture de l'interface", thread.error().detail);
  }
  return search;
}

std::vector<FontFile> FontSearch::take() {
  if (thread_.joinable()) {
    thread_.join();
  } else if (found_->empty()) {
    *found_ = search_fonts();
  }
  return std::move(*found_);
}

PlayerUi::PlayerUi(SDL_Window* window, std::string title) : window_(window) {
  look_.title = std::move(title);
}

PlayerUi::~PlayerUi() {
  if (context_ == nullptr) return;
  ImGuiIO& io = ImGui::GetIO();
  io.BackendRendererName = nullptr;
  io.BackendPlatformName = nullptr;
  io.BackendFlags = ImGuiBackendFlags_None;
  ImGui::DestroyContext(context_);
}

std::expected<std::unique_ptr<PlayerUi>, Refusal> PlayerUi::create(SDL_Window* window, std::string title,
                                                                   FontSearch& fonts) {
  if (!IMGUI_CHECKVERSION()) return std::unexpected(Refusal{RefusalCode::LibraryUnavailable, "Dear ImGui"});
  auto ui = std::unique_ptr<PlayerUi>(new PlayerUi(window, std::move(title)));
  ui->context_ = ImGui::CreateContext();
  if (ui->context_ == nullptr)
    return std::unexpected(Refusal{RefusalCode::LibraryUnavailable, "ImGui::CreateContext"});
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.LogFilename = nullptr;
  io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
  io.BackendPlatformName = "ajp_play_sdl3";
  ImGuiLayer::configure(io);
  ui->fonts_ = fonts.take();
  load_fonts(*io.Fonts, ui->fonts_);
  ui->apply_scale();
  return ui;
}

void PlayerUi::apply_scale() {
  float scale = SDL_GetWindowDisplayScale(window_);
  if (!(scale > 0.0F)) scale = 1.0F;
  ImGuiStyle& style = ImGui::GetStyle();
  style = ImGuiStyle();
  apply_theme(style);
  style.ScaleAllSizes(scale);
  style.FontSizeBase = kFontSize;
  style.FontScaleDpi = scale;
  look_.scale = scale;
}

bool PlayerUi::over_ui() const {
  return (controls_.drawn() || home_) && ImGui::GetIO().WantCaptureMouse;
}

void PlayerUi::begin_frame(std::int64_t now_us) {
  ImGuiIO& io = ImGui::GetIO();
  int width = 0;
  int height = 0;
  SDL_GetWindowSizeInPixels(window_, &width, &height);
  io.DisplaySize = {static_cast<float>(std::max(width, 1)), static_cast<float>(std::max(height, 1))};
  io.DisplayFramebufferScale = {1.0F, 1.0F};
  io.DeltaTime = last_frame_us_ > 0 ? std::max(1e-4F, static_cast<float>(now_us - last_frame_us_) * 1e-6F)
                                    : 1.0F / 60.0F;
  last_frame_us_ = now_us;
  ImGui::NewFrame();
}

void PlayerUi::follow_text_input() {
  const bool wanted = ImGui::GetIO().WantTextInput;
  if (wanted == text_input_) return;
  text_input_ = wanted;
  if (text_input_) {
    SDL_StartTextInput(window_);
  } else {
    SDL_StopTextInput(window_);
  }
}

bool PlayerUi::handle(const SDL_Event& event, std::int64_t now_us, std::vector<Input>& actions) {
  ImGuiIO& io = ImGui::GetIO();
  // Les coordonnées de SDL sont en unités de fenêtre ; l'interface travaille en pixels.
  const float density = std::max(SDL_GetWindowPixelDensity(window_), 1.0F / 8.0F);
  switch (event.type) {
    case SDL_EVENT_MOUSE_MOTION:
      io.AddMousePosEvent(event.motion.x * density, event.motion.y * density);
      controls_.activity(now_us);
      input_pending_ = true;
      return false;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
      const bool down = event.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
      const int button = mouse_button(event.button.button);
      io.AddMousePosEvent(event.button.x * density, event.button.y * density);
      if (button >= 0) io.AddMouseButtonEvent(button, down);
      controls_.activity(now_us);
      input_pending_ = true;
      // Sur l'image ou sur l'interface ? Décidé à l'image ImGui qui voit ce clic (frame).
      if (down && (button == ImGuiMouseButton_Left || button == ImGuiMouseButton_Right))
        clicks_.push_back(Click{.button = button, .count = event.button.clicks, .at_us = now_us});
      return false;
    }
    case SDL_EVENT_MOUSE_WHEEL: {
      controls_.activity(now_us);
      input_pending_ = true;
      if (over_ui() && (panel_.is_open() || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId))) {
        io.AddMouseWheelEvent(event.wheel.x, event.wheel.y);  // défilement d'un menu
        return false;
      }
      // Molette : volume, un pas par cran (les pavés tactiles envoient des fractions de cran).
      wheel_ += event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -event.wheel.y : event.wheel.y;
      while (std::abs(wheel_) >= 1.0F) {
        const float sign = wheel_ > 0.0F ? 1.0F : -1.0F;
        actions.emplace_back(VolumeStep{.delta = sign * kVolumeStep});
        wheel_ -= sign;
      }
      return false;
    }
    case SDL_EVENT_WINDOW_MOUSE_LEAVE:
      io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
      controls_.pointer_left(now_us);
      input_pending_ = true;
      return false;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
    case SDL_EVENT_WINDOW_FOCUS_LOST:
      io.AddFocusEvent(event.type == SDL_EVENT_WINDOW_FOCUS_GAINED);
      return false;
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
      apply_scale();
      input_pending_ = true;
      return false;
    case SDL_EVENT_TEXT_INPUT:
      if (!text_input_) return false;
      io.AddInputCharactersUTF8(event.text.text);
      input_pending_ = true;
      return true;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
      // Les modificateurs arrivent toujours à ImGui (Ctrl + clic sur un curseur ouvre sa saisie,
      // et un Ctrl relâché hors saisie ne reste pas enfoncé) ; les touches, seulement en saisie.
      const SDL_Keymod mod = event.key.mod;
      io.AddKeyEvent(ImGuiMod_Ctrl, (mod & SDL_KMOD_CTRL) != 0);
      io.AddKeyEvent(ImGuiMod_Shift, (mod & SDL_KMOD_SHIFT) != 0);
      io.AddKeyEvent(ImGuiMod_Alt, (mod & SDL_KMOD_ALT) != 0);
      io.AddKeyEvent(ImGuiMod_Super, (mod & SDL_KMOD_GUI) != 0);
      if (!text_input_) return false;
      if (const ImGuiKey key = imgui_key(event.key.key); key != ImGuiKey_None)
        io.AddKeyEvent(key, event.type == SDL_EVENT_KEY_DOWN);
      input_pending_ = true;
      return true;
    }
    default:
      return false;
  }
}

UiFrame PlayerUi::frame(std::int64_t now_us, const ControlsView& view, std::vector<Input>& actions) {
  if (controls_.pending_click(now_us)) actions.emplace_back(Command{cmd::TogglePause{}});
  const bool hide_cursor = controls_.cursor_hidden(now_us, view.paused);
  if (hide_cursor != cursor_hidden_) {
    if (hide_cursor) {
      SDL_HideCursor();
    } else {
      SDL_ShowCursor();
    }
    cursor_hidden_ = hide_cursor;
  }
  // ImGui place certains éléments (onglets, fenêtre qui s'ouvre) une image après les avoir reçus :
  // après une entrée, deux images de plus, même en pause sans autre événement.
  if (input_pending_) last_input_us_ = now_us;
  const bool tooltip_due = hover_waiting_ && now_us - last_input_us_ < kTooltipWindowUs;
  if (!controls_.needs_frame(now_us, view, input_pending_ || settle_ > 0 || tooltip_due))
    return UiFrame{.draw = draw_, .changed = false, .bottom = bottom_, .top = top_};
  settle_ = input_pending_ ? kSettleFrames : std::max(settle_ - 1, 0);
  input_pending_ = false;
  home_ = false;

  const ImGuiIO& io = ImGui::GetIO();
  begin_frame(now_us);
  // Les clics que l'interface ne prend pas vont à l'image : simple clic pause, double plein écran,
  // clic droit pause.
  const bool captured = io.WantCaptureMouse;
  for (const Click& click : clicks_) {
    if (captured) continue;
    if (click.button == ImGuiMouseButton_Right) {
      actions.emplace_back(Command{cmd::TogglePause{}});
    } else if (controls_.click(click.at_us, click.count)) {
      actions.emplace_back(ToggleFullscreen{});
    }
  }
  clicks_.clear();
  bool drawn = draw_controls(controls_, view, look_, now_us, actions);
  if (panel_.is_open()) {
    const SettingsPanel::Result result = panel_.draw(running_, choices_, look_.scale, actions);
    if (result.live) actions.emplace_back(ApplySettings{.settings = *result.live});
    if (result.reopen) actions.emplace_back(ApplySettings{.settings = *result.reopen, .reopen = true});
    look_.keep_visible = panel_.is_open();  // fermé par sa croix
    drawn = true;
  }
  hover_waiting_ = ImGui::IsAnyItemHovered();
  ImGui::Render();
  controls_.framed(now_us, view, drawn);
  follow_text_input();
  draw_ = drawn ? ImGui::GetDrawData() : nullptr;
  const bool bar = controls_.shown(now_us, view.paused);
  bottom_ = bar ? static_cast<int>(std::ceil(controls_height(look_.scale))) : 0;
  top_ = bar ? static_cast<int>(std::ceil(title_height(look_.scale))) : 0;
  return UiFrame{.draw = draw_, .changed = true, .bottom = bottom_, .top = top_};
}

const ImDrawData* PlayerUi::preparation(std::int64_t now_us, const PrepareView& view) {
  home_ = false;
  begin_frame(now_us);
  clicks_.clear();
  draw_preparation(view, look_, now_us);
  ImGui::Render();
  draw_ = nullptr;  // la barre de lecture se recompose à son premier tour
  controls_.framed(now_us, ControlsView{}, false);
  return ImGui::GetDrawData();
}

const ImDrawData* PlayerUi::welcome(std::int64_t now_us, FirstRunView& view) {
  home_ = false;
  begin_frame(now_us);
  clicks_.clear();
  draw_first_run(view, look_.scale);
  ImGui::Render();
  draw_ = nullptr;
  controls_.framed(now_us, ControlsView{}, false);
  return ImGui::GetDrawData();
}

const ImDrawData* PlayerUi::home(std::int64_t now_us, HomeView& view, std::vector<Input>& actions) {
  if (cursor_hidden_) {  // retour d'une lecture où la souris s'était effacée
    SDL_ShowCursor();
    cursor_hidden_ = false;
  }
  home_ = true;
  input_pending_ = false;
  begin_frame(now_us);
  clicks_.clear();
  draw_home(view, look_.scale, panel_.is_open());
  if (panel_.is_open()) {
    const SettingsPanel::Result result = panel_.draw(running_, choices_, look_.scale, actions);
    if (result.live) actions.emplace_back(ApplySettings{.settings = *result.live});
    if (result.reopen) actions.emplace_back(ApplySettings{.settings = *result.reopen, .reopen = true});
  }
  ImGui::Render();
  follow_text_input();
  draw_ = nullptr;  // la barre de lecture se recompose à son premier tour
  controls_.framed(now_us, ControlsView{}, false);
  return ImGui::GetDrawData();
}

void PlayerUi::set_settings(const PlaySettings& running, SettingsChoices choices) {
  running_ = running;
  choices_ = std::move(choices);
  input_pending_ = true;
}

void PlayerUi::set_black_status(std::string status) {
  if (choices_.black_status == status) return;
  choices_.black_status = std::move(status);
  input_pending_ = true;
}

void PlayerUi::set_profiles(ProfilesView profiles) {
  choices_.profiles = std::move(profiles);
  input_pending_ = true;
}

void PlayerUi::set_running(const PlaySettings& running) {
  running_ = running;
  input_pending_ = true;
}

void PlayerUi::toggle_settings() {
  if (panel_.is_open()) {
    panel_.close();
  } else {
    panel_.open(running_);
  }
  look_.keep_visible = panel_.is_open();
  // Ouvert au clavier, barre effacée : sans cela, aucune image ne se dessine avant que la souris
  // bouge, et le menu n'apparaît pas.
  controls_.set_held(look_.keep_visible);
  input_pending_ = true;
}

void PlayerUi::open_settings() {
  if (!panel_.is_open()) toggle_settings();
}

void PlayerUi::set_title(std::string title) {
  look_.title = std::move(title);
  input_pending_ = true;
}

void PlayerUi::close_settings() {
  panel_.close();
  look_.keep_visible = false;
  controls_.set_held(false);
  input_pending_ = true;
}

void PlayerUi::set_tracks(TrackMenu tracks) {
  look_.tracks = std::move(tracks);
  input_pending_ = true;
}

void PlayerUi::release(PlaceboScreen& screen) {
  screen.release_ui(ImGui::GetPlatformIO());
  draw_ = nullptr;
}

}  // namespace ajp::app::play
