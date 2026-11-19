// L'aide des commandes du lecteur (options.h), à part de la lecture des options.
#include <format>
#include <optional>
#include <string>

#include "app/play/options.h"
#include "app/play/settings.h"
#include "app/play/updater.h"

namespace kirari::app::play {

// L'aide, en anglais comme l'interface par défaut : écrite avant la lecture des réglages et de la langue.
std::string compare_usage() {
  return "kirari compare <file> --a=\"OPTIONS\" --b=\"OPTIONS\" [options]   two chains side by side, sound "
         "off\n"
         "  --a, --b            \"kirari play\" options of each variant, e.g. --a=\"--rife=3\" "
         "--b=\"--rife=4\"\n"
         "  --a-settings=FILE, --b-settings=FILE   starting play.toml (DRBA poses, interpolation engine…)\n"
         "  --seconds=S         length of one run (5 to 120, default 12); A, B, B, A: the worse of two "
         "passes\n"
         "  --report=FILE       the report, besides standard output; --windowed: in a window\n";
}

std::string bench_usage() {
  return "kirari bench [file] [options]   rates the machine at each level, sound off, hidden window; without "
         "a "
         "file, the player's test sequence\n"
         "  --seconds=S      length of one run (5 to 120, default 12)\n"
         "  --report=FILE    the report, besides standard output\n"
         "  --windowed       plays in a window instead of full screen\n"
         "  --visible        shows the runs' window (hidden by default)\n"
         "The result is kept in bench.toml (card, driver, level, date) and replaces the card's estimate.\n";
}

namespace {

std::string trial_usage();

}  // namespace

std::string play_usage(bool debug) {
  const std::string usage = std::format(
      "kirari play [file] [options]   (without a file: home screen, Open a video); kirari --version\n"
      "  --model=FILE.onnx      SPAN ×2 model (default: {})\n"
      "  --no-upscale           source picture, reduced only to the working resolution\n"
      "  --no-restore-lines     a source above the working resolution keeps its detail, no SPAN\n"
      "  --window=WxH           window size (default 1280x720)\n"
      "  --fullscreen           full screen from the start\n"
      "  --start=S              start position, in seconds (otherwise, where playback stopped)\n"
      "  --no-resume            neither resume nor remember the position for this playback\n"
      "  --seconds=S            quits after S seconds of playback, from the first frame shown\n"
      "  --fullscreen-after=S   goes full screen after S seconds\n"
      "  --stats=FILE           timing of each presented frame\n"
      "  --snapshot=FILE        the window as PPM, after --snapshot-at=S seconds (3) of playback,\n"
      "                         or of an engine build if it lasts that long; on an HDR10\n"
      "                         output, only a .pq.ppm name keeps the graded picture\n"
      "  --snapshot-frames=N    N presentations in a row (FILE-001.ppm…), for flicker\n"
      "  --sdr                  SDR output even on an HDR screen (SDR grade tests)\n"
      "  --no-probe-flow        shot cuts without NVIDIA's optical flow, as on other cards\n"
      "  --bake-looks=FILE.cube writes play.toml's LUT stack (strengths included) as a single\n"
      "                         65-point LUT, then stops\n"
      "  --anime-contrast=N, --anime-saturation=N, --anime-highlights=N\n"
      "                         the anime grade's punch: contrast, saturation, highlights, 0 to 200% (100: "
      "as is)\n"
      "  --software-gpu         accepts a software Vulkan (lavapipe), for tests without a GPU\n"
      "  --mute                 sound off for this playback\n"
      "  --data-dir=FOLDER      settings and log in this folder (engines and models shared)\n"
      "  --bench-level=N        plays the chain of level 0 to 4 (or auto); --bench-out=FILE writes the\n"
      "                         measurements there\n"
      "  --first-run            welcome screen (engines), even if already seen\n"
      "  --keys=S:key,…         keys pressed automatically (tests), e.g. 3:Right,5:Space;\n"
      "                         export or export=FILE.mkv (or .mp4, .mov) starts exporting\n"
      "                         the current chain (export-lossless=FILE.mkv: lossless FFV1)\n"
      "  --work-height=N        shrinks the source to N lines before SPAN (default: the preset's; 0 never)\n"
      "  --rife=N               ×N interpolation (0 none, 1, 1.5 to 9.5 by halves, 10; default: preset's)\n"
      "  --rife-model=FILE      RIFE model (default: the preset's, {} without one)\n"
      "  --motion=rife|drba|gmfss  RIFE, DistilDRBA or GMFSS union alone (default: the preset's)\n"
      "  --gmfss-dir=FOLDER     GMFSS union's two graphs from this folder (default: beside RIFE, or the\n"
      "                         gmfss folder next to RIFE's)\n"
      "  --truehdr-peak=NITS    peak aimed at by the HDR grade, {} to {} (0 or default: the screen's,\n"
      "                         else 1000)\n"
      "kirari --register           registers the player with the system (\"Open with\", default\n"
      "                         player); --unregister removes this registration\n"
      "Keys: space pause, ← → seek 5 s, ↑ ↓ volume, m mute, f full screen (Esc leaves it), q quit,\n"
      "  a next audio track, j next subtitles, n / Shift+n next / previous episode,\n"
      "  Page↑ / Page↓ next / previous chapter, o skip the credits, . and , frame by frame, z / x\n"
      "  subtitle delay, Ctrl+- / Ctrl++ audio delay, p settings, Tab statistics, Ctrl+0 neither grade nor\n"
      "  interpolation (the upscaling stays), Ctrl+G grade, Ctrl+I interpolation.\n"
      "Other commands: kirari bench (rates the machine), kirari compare (two chains side by side),\n"
      "  kirari calibrate (the bench of the Results page); kirari bench --help, kirari compare --help\n",
      kDefaultModelFile, kDefaultRifeFile, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi);
  if (!debug) return usage + "Trial options: kirari --debug --help\n";
  return usage + trial_usage();
}

namespace {

// Les options d'essai (--debug), à part de play_usage : la liste s'allonge à chaque essai.
std::string trial_usage() {
  return "Trial options, with --debug:\n"
         "  --rife-image           image RIFE at every instant (without the motion variant)\n"
         "  --ring-rgb10a2         RGB10A2 ring under TensorRT (half the memory, banding not\n"
         "                         measured yet); default and --no-ring-rgb10a2: RGBA16\n"
         "  --hdr-metadata[=NITS]  HDR10 output with its metadata, KWin's compression done by the\n"
         "                         grade (NITS: KDE's SDR brightness, read from KDE without a value);\n"
         "                         default on Linux and Windows; --no-hdr-metadata: compression\n"
         "                         left to KWin (Windows: clipping at the screen's peak)\n"
         "  --probe-nvdec          shot pre-analysis decoded by NVDEC at 256x144 (default;\n"
         "                         --no-probe-nvdec: software decoding)\n"
         "  --no-shot-cuts         interpolation cuts measured by RIFE alone (default: the pre-analysis\n"
         "                         shots decide, RIFE when they do not cover the pair)\n"
         "  --shot-plan            trial: Auto per shot, DistilDRBA on limited animation, RIFE on\n"
         "                         sakuga, TemporalFix only on grainy still shots (pre-analysis rhythm)\n"
         "  --one-lane             a single compute queue (the default): the GPU profile gives each\n"
         "                         engine's own cost\n"
         "  --lanes                three compute queues instead of one\n"
         "  --profile-dispatches   Vulkan engine: GPU time and dispatch count per shader of each network,\n"
         "                         logged when the network is unloaded\n"
         "  --skip-held            playback: an instant identical to the shown frame is neither computed "
         "nor\n"
         "                         written (trial; off by default)\n"
         "  --anime-shots          SDR output only: the grade decided per shot (default); on HDR output the "
         "lift is\n"
         "                         fixed whatever this says; --no-anime-shots: fixed lift on SDR too\n"
         "  --fast-steady          trial: Instant mode at a constant step, N frames per held source in a "
         "shot\n"
         "  --smooth-drawings      a drawing held 2 to 6 frames blends into motion (motion RIFE; the\n"
         "                         Video menu option \"Smooth held drawings\" for this playback)\n"
         "  --no-tfix-cache        trial: temporal stabilization in one engine, each frame encoded at\n"
         "                         every pass, instead of the encoder cache (same output)\n"
         "  --no-tfix-fused        trial: temporal stabilization by the encoder-cache graph instead of\n"
         "                         the fused engines (TensorRT)\n"
         "  --pair-triage          trial, with --scene-map: held drawings copied, impossible pairs and\n"
         "                         doubt crossfaded (RIFE), pure pans warped by the camera (GMFSS)\n"
         "  --rife-calm-model=FILE trial: a lighter RIFE for the middle pass of calm pairs (motion RIFE)\n"
         "  --rife-encode-all      measurement: fused RIFE encodes both images at every pass (same output)\n"
         "  --no-span-reuse        measurement: SPAN on every frame, held drawings included\n"
         "  --rife-native, --gmfss-native  trial: fused RIFE, GMFSS small nets by our CUDA kernels\n"
         "  --span-tiles           trial: SPAN by tiles when a frame is not reused whole\n"
         "                         (--span-tiles-check: 1 in 24 also computed whole and compared)\n"
         "  --rife-at-slot         measurement: each RIFE pass of a strong pair waits for the ring slot\n"
         "                         of its image instead of being queued when the pair arrives (same output)\n"
         "  --scene-map            trial: NVIDIA optical flow (RTX 30 and later) maps each pair: camera,\n"
         "                         tiles that move, cuts; same output until a feature reads it\n"
         "  --no-effect-highlights with --scene-map: highlights on every light, not only on effects\n"
         "  --effect-mask-view     with --scene-map: the grade paints the effects mask, not the picture\n"
         "                         (lights red, sure effects green, held margin blue)\n"
         "  --pause-at=S           pauses on the first source image at S seconds or later, the same\n"
         "                         image from one run to the next (to compare two --snapshot runs)\n"
         "  --warp-l2              trial: TensorRT warp keeps the flow in persisting L2 (same output)\n"
         "  --no-rife-hwc8         trial: fused RIFE blocks built with planar inputs and outputs under\n"
         "                         TensorRT, with reformat copies (default: HWC8, same output)\n"
         "  --rife-warm            trial: calm pairs reuse the coarse RIFE blocks of the previous pair;\n"
         "                         trades quality for speed, neither measured (ghosting if a calm shot\n"
         "                         speeds up)\n"
         "  --ring-watcher         a thread waits for ring writes and wakes the display (default;\n"
         "                         --no-ring-watcher: the display polls)\n"
         "  --precise-wait         waits for the next frame to a few tens of µs on Linux, a high-resolution\n"
         "                         timer on Windows (default; --no-precise-wait: to the millisecond)\n"
         "  --audio-pll            audio clock filtered by a phase-locked loop (default;\n"
         "                         --no-audio-pll: the raw clock)\n"
         "  --vsync-index          frame chosen by its measured vblank number, with hysteresis (default;\n"
         "                         --no-vsync-index: the clock)\n"
         "  --present=LEVEL        presentation feedback wait (default), auto, timing, timing-absolute\n"
         "                         (timing with a target date), wayland or open; --no-present: open, the\n"
         "                         former presentation; trials: bare (extensions only), poll (present\n"
         "                         ids and waits only), timing-plain (timing, default swapchain)\n"
         "  --no-vrr               fixed refresh screen even if VRR is armed (Instant mode trial)\n"
         "  --swap-mode=MODE       swapchain present mode: fifo (default), mailbox, immediate, relaxed\n"
         "  --vrr-margin=N         Instant mode stays N % under a VRR screen's maximum (default 1)\n"
         "  --display-loss-pressure  frames the display lost while ready widen the Instant step too\n"
         "                         (former count)\n";
}

}  // namespace

std::optional<std::string> help_or_version(const PlayOptions& options) {
  if (options.help) return play_usage(options.debug);
  if (options.version) return std::format("kirari {}\n", installed_version());
  return std::nullopt;
}

}  // namespace kirari::app::play
