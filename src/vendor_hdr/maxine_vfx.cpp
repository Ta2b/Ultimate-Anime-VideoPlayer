#include "vendor_hdr/maxine_vfx.h"

#include <array>
#include <climits>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "vendor_hdr/dynamic_library.h"
#include "vendor_hdr/log.h"
#include "vendor_hdr/maxine_abi.h"
#include "vendor_hdr/maxine_runtime.h"

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
        bind(*effects, "NvVFX_Run", api.run), bind(*images, "NvCVImage_Init", api.image_init),
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

  // Repris de pipeline_hdr.c:57-66 : dossier des modèles (TrueHDR 1.3 répond UNIMPLEMENTED,
  // seul ce code est toléré), puis flux 0 hérité.
  Done configure(const fs::path& root) {
    const std::string models = (root / "lib" / "models").string();
    const NvCV_Status directory = api_.set_string(effect_, kModelDirectory, models.c_str());
    if (directory != kErrUnimplemented) {
      if (Done set = check(directory, "dossier des modèles"); !set) return set;
    }
    return check(api_.set_stream(effect_, kCudaStream, nullptr), "flux CUDA");
  }

  TrueHdrPixel input_pixel() const override { return TrueHdrPixel::Rgba8; }

  Done apply(const TrueHdrParams& params) override {
    const std::array<std::pair<const char*, int>, 4> values{{{kContrast, params.contrast},
                                                             {kSaturation, params.saturation},
                                                             {kMiddleGray, params.middle_gray},
                                                             {kLuminance, params.peak_nits}}};
    for (const auto& [selector, value] : values) {
      if (Done set = check(api_.set_u32(effect_, selector, static_cast<unsigned>(value)), selector); !set)
        return set;
    }
    return {};
  }

  // Repris de pipeline_hdr.c:68-86 : descripteurs réinitialisés et liés à chaque image, modèle
  // chargé une fois, puis exécution synchrone sur le flux 0.
  Done run_on_legacy(const TrueHdrImage& in, const TrueHdrImage& out) override {
    if (Done bound = describe(source_, in, kRgba, kU8, "image source"); !bound) return bound;
    if (Done bound = describe(target_, out, kRgb10A2, kP32, "image de sortie"); !bound) return bound;
    if (Done bound = check(api_.set_image(effect_, kInputImage, &source_), "liaison de l'entrée"); !bound)
      return bound;
    if (Done bound = check(api_.set_image(effect_, kOutputImage, &target_), "liaison de la sortie"); !bound) {
      return bound;
    }
    if (!loaded_) {
      if (Done loaded = check(api_.load(effect_), "chargement du modèle"); !loaded) return loaded;
      loaded_ = true;
    }
    return check(api_.run(effect_, 0), "exécution");
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
};

}  // namespace

namespace detail {

std::expected<std::unique_ptr<TrueHdr>, Refusal> open_maxine_at(const fs::path& root,
                                                                const TrueHdrConfig& config) {
  if (Done valid = check(config); !valid) return std::unexpected(valid.error());
  auto libraries = load_maxine_runtime(root);
  if (!libraries) return std::unexpected(libraries.error());
  auto api = resolve_api(*libraries);
  if (!api) return std::unexpected(api.error());

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
  if (opened) log().info("TrueHDR Maxine prêt en {}x{}", config.width, config.height);
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
