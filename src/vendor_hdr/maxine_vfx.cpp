#include "vendor_hdr/maxine_vfx.h"

#include <array>
#include <climits>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vendor_hdr/dynamic_library.h"
#include "vendor_hdr/log.h"
#include "vendor_hdr/maxine_abi.h"
#include "vendor_hdr/maxine_runtime.h"
#include "vendor_hdr/sdk_layout.h"

namespace ajp::vendor_hdr {
namespace {

namespace fs = std::filesystem;
using namespace maxine;

struct MaxineApi {
  CreateEffectFn* create = nullptr;
  DestroyEffectFn* destroy = nullptr;
  SetU32Fn* set_u32 = nullptr;
  SetStringFn* set_string = nullptr;
  SetImageFn* set_image = nullptr;
  SetCudaStreamFn* set_stream = nullptr;
  LoadFn* load = nullptr;
  RunFn* run = nullptr;
  ImageInitFn* image_init = nullptr;
  ErrorStringFn* error_string = nullptr;
  GetVersionFn* version = nullptr;
};

template <class Fn>
Done bind(const DynamicLibrary& library, const char* name, Fn*& slot) {
  auto address = library.symbol<Fn>(name);
  if (!address) return std::unexpected(address.error());
  slot = *address;
  return {};
}

std::expected<MaxineApi, Refusal> resolve_api(const std::vector<LoadedLibrary>& loaded) {
  const DynamicLibrary* effects = find_library(loaded, "lib/libVideoFX.so");
  const DynamicLibrary* images = find_library(loaded, "lib/libNVCVImage.so");
  if (effects == nullptr || images == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable, "libVideoFX.so ou libNVCVImage.so n'est pas ouverte");
  }
  MaxineApi api;
  for (const Done& bound :
       {bind(*effects, "NvVFX_CreateEffect", api.create), bind(*effects, "NvVFX_DestroyEffect", api.destroy),
        bind(*effects, "NvVFX_SetU32", api.set_u32), bind(*effects, "NvVFX_SetString", api.set_string),
        bind(*effects, "NvVFX_SetImage", api.set_image),
        bind(*effects, "NvVFX_SetCudaStream", api.set_stream), bind(*effects, "NvVFX_Load", api.load),
        bind(*effects, "NvVFX_Run", api.run), bind(*effects, "NvVFX_GetVersion", api.version),
        bind(*images, "NvCVImage_Init", api.image_init),
        bind(*images, "NvCV_GetErrorStringFromCode", api.error_string)}) {
    if (!bound) return std::unexpected(bound.error());
  }
  return api;
}

class MaxineEffect final : public VendorEffect {
 public:
  MaxineEffect(std::vector<LoadedLibrary> libraries, const MaxineApi& api, NvVFX_Handle effect)
      : libraries_(std::move(libraries)), api_(api), effect_(effect) {}

  MaxineEffect(const MaxineEffect&) = delete;
  MaxineEffect& operator=(const MaxineEffect&) = delete;

  ~MaxineEffect() override {
    api_.destroy(effect_);
    while (!libraries_.empty()) libraries_.pop_back();  // ordre inverse de l'ouverture
  }

  // Repris de pipeline_hdr.c:57-66 : dossier des modèles (TrueHDR 1.3.0.0 répond
  // NVCV_ERR_UNIMPLEMENTED sur la RTX, seul ce code est toléré), puis flux 0 hérité. Changer
  // de flux après Load exige un nouveau Load (NVCV_ERR_INITIALIZATION sinon, mesuré) : le flux
  // est fixé une fois, avant.
  Done configure(const fs::path& root) {
    const std::string models = (root / "lib" / "models").string();
    const NvCV_Status directory = api_.set_string(effect_, kModelDirectory, models.c_str());
    if (directory != kErrUnimplemented) {
      if (Done set = check(directory, "NvVFX_SetString(ModelDir)"); !set) return set;
    }
    return check(api_.set_stream(effect_, kCudaStream, nullptr), "NvVFX_SetCudaStream(0)");
  }

  TrueHdrPixel input_pixel() const override { return TrueHdrPixel::Rgba8; }

  // Chaque réglage est passé, le pic compris : le lecteur ne dépend jamais des défauts du SDK.
  Done apply(const TrueHdrParams& params) override {
    struct Setting {
      const char* selector;
      std::string_view call;
      int value;
    };
    const std::array<Setting, 4> values{{{kContrast, "NvVFX_SetU32(Contrast)", params.contrast},
                                         {kSaturation, "NvVFX_SetU32(Saturation)", params.saturation},
                                         {kMiddleGray, "NvVFX_SetU32(MiddleGray)", params.middle_gray},
                                         {kLuminance, "NvVFX_SetU32(Luminance)", params.peak_nits}}};
    for (const Setting& setting : values) {
      const NvCV_Status status =
          api_.set_u32(effect_, setting.selector, static_cast<unsigned>(setting.value));
      if (Done set = check(status, setting.call); !set) return set;
    }
    // DebandingOff ne vaut qu'au chargement du modèle (mesuré sur la RTX, truehdr.md) : un
    // changement demande un nouveau NvVFX_Load, fait une fois à l'image suivante.
    const unsigned off = params.debanding ? 0U : 1U;
    if (debanding_off_ == off) return {};
    if (Done set = check(api_.set_u32(effect_, kDebandingOff, off), "NvVFX_SetU32(DebandingOff)"); !set)
      return set;
    debanding_off_ = off;
    loaded_ = false;
    return {};
  }

  // Repris de pipeline_hdr.c:68-86 : descripteurs réinitialisés et liés à chaque image, modèle
  // chargé une fois, puis exécution synchrone sur le flux 0.
  Done run_on_legacy(const TrueHdrImage& in, const TrueHdrImage& out) override {
    if (Done bound = describe(source_, in, kRgba, kU8, "NvCVImage_Init de l'entrée"); !bound) return bound;
    if (Done bound = describe(target_, out, kRgb10A2, kP32, "NvCVImage_Init de la sortie"); !bound)
      return bound;
    if (Done bound = check(api_.set_image(effect_, kInputImage, &source_), "NvVFX_SetImage(SrcImage0)");
        !bound)
      return bound;
    if (Done bound = check(api_.set_image(effect_, kOutputImage, &target_), "NvVFX_SetImage(DstImage0)");
        !bound) {
      return bound;
    }
    if (!loaded_) {
      if (Done loaded = check(api_.load(effect_), "NvVFX_Load"); !loaded) return loaded;
      loaded_ = true;
    }
    return check(api_.run(effect_, 0), "NvVFX_Run");
  }

 private:
  Done check(NvCV_Status status, std::string_view operation) const {
    if (status == kSuccess) return {};
    const char* text = api_.error_string(status);
    return refuse(RefusalCode::VendorCallFailed,
                  std::format("TrueHDR Maxine, {} : {} ({})", operation,
                              text != nullptr ? text : "erreur du SDK", status));
  }

  Done describe(NvCVImage& image, const TrueHdrImage& buffer, int format, int type,
                std::string_view role) const {
    if (buffer.pitch > static_cast<std::size_t>(INT_MAX)) {
      return refuse(RefusalCode::InputMismatch, std::format("{} : pas de ligne trop grand", role));
    }
    const NvCV_Status status =
        api_.image_init(&image, static_cast<unsigned>(buffer.width), static_cast<unsigned>(buffer.height),
                        static_cast<int>(buffer.pitch),
                        reinterpret_cast<void*>(buffer.device),  // NOLINT(performance-no-int-to-ptr)
                        format, type, kInterleaved, kGpu);
    if (Done valid = check(status, role); !valid) return valid;
    // Une constante de maxine_abi.h fausse se voit ici plutôt que dans les pixels.
    if (image.pixel_bytes != 4 || image.planar != 0 || image.gpu_mem != kGpu) {
      return refuse(RefusalCode::VendorCallFailed,
                    std::format("TrueHDR Maxine, {} : descripteur inattendu ({} octets par pixel)", role,
                                image.pixel_bytes));
    }
    return {};
  }

  std::vector<LoadedLibrary> libraries_;
  MaxineApi api_;
  NvVFX_Handle effect_;
  NvCVImage source_{};  // persistants : le SDK garde leur adresse entre SetImage et Run
  NvCVImage target_{};
  bool loaded_ = false;
  std::optional<unsigned> debanding_off_;  // valeur passée au SDK, aucune avant le premier apply
};

// La bibliothèque chargée doit être de la version que le magasin a vérifiée dans les fichiers.
Done check_loaded_version(const MaxineApi& api) {
  unsigned version = 0;
  const NvCV_Status status = api.version(&version);
  if (status != kSuccess) {
    const char* text = api.error_string(status);
    return refuse(RefusalCode::VendorCallFailed,
                  std::format("TrueHDR Maxine, NvVFX_GetVersion : {} ({})",
                              text != nullptr ? text : "erreur du SDK", status));
  }
  const SdkLayout& sdk = layout(SdkKind::MaxineVfx);
  if (version_major(version) != sdk.major || version_minor(version) != sdk.minor) {
    return refuse(RefusalCode::SdkRejected,
                  std::format("libVideoFX.so chargée est en {}.{}, ce lecteur prend en charge {}.{}",
                              version_major(version), version_minor(version), sdk.major, sdk.minor));
  }
  return {};
}

}  // namespace

namespace detail {

std::expected<std::unique_ptr<TrueHdr>, Refusal> open_maxine_at(const fs::path& root,
                                                                const TrueHdrConfig& config) {
  if (Done valid = check(config); !valid) return std::unexpected(valid.error());
  auto libraries = load_maxine_runtime(root);
  if (!libraries) return std::unexpected(libraries.error());
  auto api = resolve_api(*libraries);
  if (!api) return std::unexpected(api.error());
  if (Done same = check_loaded_version(*api); !same) return std::unexpected(same.error());

  NvVFX_Handle handle = nullptr;
  const NvCV_Status created = api->create(kFxTrueHdr, &handle);
  if (created != kSuccess || handle == nullptr) {
    const char* text = api->error_string(created);
    return refuse(
        RefusalCode::VendorCallFailed,
        std::format("TrueHDR Maxine, création de l'effet : {} ({}) ; la fonction nvvfxtruehdr est-elle "
                    "installée ?",
                    text != nullptr ? text : "erreur du SDK", created));
  }
  auto effect = std::make_unique<MaxineEffect>(std::move(*libraries), *api, handle);
  if (Done configured = effect->configure(root); !configured) return std::unexpected(configured.error());
  auto opened = make_true_hdr(config, std::move(effect));
  if (opened) make_logger()->info("TrueHDR Maxine prêt en {}x{}", config.width, config.height);
  return opened;
}

}  // namespace detail

std::expected<std::unique_ptr<TrueHdr>, Refusal> open_maxine(const InstalledSdk& sdk,
                                                             const TrueHdrConfig& config) {
  if (sdk.kind() != SdkKind::MaxineVfx) {
    return refuse(RefusalCode::LibraryUnavailable, "le SDK fourni n'est pas Maxine VFX");
  }
  return detail::open_maxine_at(sdk.root(), config);
}

}  // namespace ajp::vendor_hdr
