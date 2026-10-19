#include "app/play/ui.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_video.h>
#include <imgui.h>
#if AJP_WITH_FONTCONFIG
#include <fontconfig/fontconfig.h>
#endif

#include "app/play/placebo_imgui.h"
#include "app/play/placebo_screen.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

constexpr float kFontSize = 17.0F;  // pixels à l'échelle 1, multipliés par l'échelle de l'écran

struct FontFile {
  std::string path;
  int index = 0;
};

// Le fichier de police que fontconfig choisit pour pattern (« sans-serif », « sans-serif:lang=ja ») ;
// seulement TrueType ou OpenType, ce que lit ImGui.
std::optional<FontFile> system_font(const char* pattern) {
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
    std::string path(reinterpret_cast<const char*>(file));
    std::string ext = std::filesystem::path(path).extension().string();
    std::ranges::transform(ext, ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".ttf" || ext == ".otf" || ext == ".ttc" || ext == ".otc")
      found = FontFile{std::move(path), index};
  }
  if (match != nullptr) FcPatternDestroy(match);
  FcPatternDestroy(wanted);
  return found;
#else
  static_cast<void>(pattern);
  return std::nullopt;
#endif
}

bool add_font(ImFontAtlas& fonts, const FontFile& file, bool merge) {
  ImFontConfig config;
  config.MergeMode = merge;
  config.FontNo = static_cast<ImU32>(std::max(file.index, 0));
  return fonts.AddFontFromFileTTF(file.path.c_str(), kFontSize, &config) != nullptr;
}

// Police du bureau (sans-serif), complétée par une police japonaise pour les titres de pistes ;
// sans fontconfig, la police vectorielle d'ImGui. ImGui 1.92 rastérise à la taille demandée.
void load_fonts(ImFontAtlas& fonts) {
  const auto log = log_channel("app");
  const std::optional<FontFile> main = system_font("sans-serif");
  if (!main || !add_font(fonts, *main, false)) {
    fonts.AddFontDefaultVector();
    log->info("interface : police vectorielle d'ImGui");
    return;
  }
  log->info("interface : police {}", main->path);
  const std::optional<FontFile> japanese = system_font("sans-serif:lang=ja");
  if (japanese && japanese->path != main->path) add_font(fonts, *japanese, true);
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

std::expected<std::unique_ptr<PlayerUi>, Refusal> PlayerUi::create(SDL_Window* window, std::string title) {
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
  load_fonts(*io.Fonts);
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
  return controls_.drawn() && ImGui::GetIO().WantCaptureMouse;
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
      if (!text_input_) return false;
      const SDL_Keymod mod = event.key.mod;
      io.AddKeyEvent(ImGuiMod_Ctrl, (mod & SDL_KMOD_CTRL) != 0);
      io.AddKeyEvent(ImGuiMod_Shift, (mod & SDL_KMOD_SHIFT) != 0);
      io.AddKeyEvent(ImGuiMod_Alt, (mod & SDL_KMOD_ALT) != 0);
      io.AddKeyEvent(ImGuiMod_Super, (mod & SDL_KMOD_GUI) != 0);
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
  if (!controls_.needs_frame(now_us, view, input_pending_))
    return UiFrame{.draw = draw_, .changed = false, .bottom = bottom_, .top = top_};
  input_pending_ = false;

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
  ImGui::Render();
  controls_.framed(now_us, view, drawn);
  if (io.WantTextInput != text_input_) {
    text_input_ = io.WantTextInput;
    if (text_input_) {
      SDL_StartTextInput(window_);
    } else {
      SDL_StopTextInput(window_);
    }
  }
  draw_ = drawn ? ImGui::GetDrawData() : nullptr;
  const bool bar = controls_.shown(now_us, view.paused);
  bottom_ = bar ? static_cast<int>(std::ceil(controls_height(look_.scale))) : 0;
  top_ = bar ? static_cast<int>(std::ceil(title_height(look_.scale))) : 0;
  return UiFrame{.draw = draw_, .changed = true, .bottom = bottom_, .top = top_};
}

const ImDrawData* PlayerUi::preparation(std::int64_t now_us, const PrepareView& view) {
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
  clicks_.clear();
  draw_preparation(view, look_, now_us);
  ImGui::Render();
  draw_ = nullptr;  // la barre de lecture se recompose à son premier tour
  controls_.framed(now_us, ControlsView{}, false);
  return ImGui::GetDrawData();
}

void PlayerUi::set_settings(const PlaySettings& running, SettingsChoices choices) {
  running_ = running;
  choices_ = std::move(choices);
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
  input_pending_ = true;
}

void PlayerUi::close_settings() {
  panel_.close();
  look_.keep_visible = false;
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
