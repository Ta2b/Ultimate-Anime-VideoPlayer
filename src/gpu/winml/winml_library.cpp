#include "gpu/winml/winml_library.h"

#include <format>

#ifdef _WIN32
#include <WinMLAsync.h>
#include <WinMLEpCatalog.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <string_view>
#include <vector>
#include <windows.h>
#endif

namespace ajp::gpu::winml_rt {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

}  // namespace

#ifndef _WIN32

std::expected<void, Refusal> load_libraries() {
  return refuse(RefusalCode::LibraryUnavailable, "Windows ML : Windows seulement");
}

std::string ort_version() {
  return {};
}

std::expected<Provider, Refusal> find_provider(Ep /*ep*/) {
  return refuse(RefusalCode::LibraryUnavailable, "Windows ML : Windows seulement");
}

std::expected<Provider, Refusal> ensure_provider(Ep /*ep*/, const std::function<void(double)>& /*progress*/,
                                                 const std::stop_token& /*stop*/) {
  return refuse(RefusalCode::LibraryUnavailable, "Windows ML : Windows seulement");
}

#else

namespace {

constexpr const wchar_t* kOrtLibrary = L"onnxruntime.dll";
constexpr const wchar_t* kWinmlLibrary = L"Microsoft.Windows.AI.MachineLearning.dll";

// Ce qui distingue un fournisseur dans le catalogue : son nom pour le lecteur, le mot de son nom
// de catalogue (« MIGraphXExecutionProvider »), et les conditions de Windows ML, dites quand le
// catalogue ne le propose pas.
struct Kind {
  std::string_view label;
  std::string_view key;
  std::string_view conditions;
};

Kind kind_of(Ep ep) {
  if (ep == Ep::VitisAi)
    return {.label = "VitisAI", .key = "vitisai", .conditions = "NPU de Ryzen AI (XDNA 2) et son pilote AMD"};
  return {.label = "MIGraphX",
          .key = "migraphx",
          .conditions =
              "GPU AMD RDNA 3 ou plus récent, pilote AMD récent (25.10.13.09 ou après) et Windows 11 24H2"};
}

std::string hresult(HRESULT hr) {
  return std::format("0x{:08X}", static_cast<std::uint32_t>(hr));
}

// Les fonctions du catalogue, résolues dans Microsoft.Windows.AI.MachineLearning.dll.
struct Catalog {
  decltype(&WinMLEpCatalogCreate) create = nullptr;
  decltype(&WinMLEpCatalogRelease) release = nullptr;
  decltype(&WinMLEpCatalogEnumProviders) enumerate = nullptr;
  decltype(&WinMLEpCatalogFindProvider) find = nullptr;
  decltype(&WinMLEpGetVersionSize) version_size = nullptr;
  decltype(&WinMLEpGetVersion) version = nullptr;
  decltype(&WinMLEpGetLibraryPathSize) path_size = nullptr;
  decltype(&WinMLEpGetLibraryPath) path = nullptr;
  decltype(&WinMLEpGetReadyState) ready_state = nullptr;
  decltype(&WinMLEpEnsureReadyAsync) ensure_async = nullptr;
  decltype(&WinMLAsyncGetStatus) async_status = nullptr;
  decltype(&WinMLAsyncCancel) async_cancel = nullptr;
  decltype(&WinMLAsyncClose) async_close = nullptr;
};

struct Loaded {
  const OrtApiBase* ort = nullptr;
  Catalog catalog;
};

// Le dossier de ce module (ajp_backend_migraphx.dll, ou l'exécutable qui lie le runtime).
std::expected<std::filesystem::path, Refusal> own_directory() {
  HMODULE self = nullptr;
  if (GetModuleHandleExW(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(&own_directory), &self) == 0) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("Windows ML : module introuvable (erreur Windows {})", GetLastError()));
  }
  std::vector<wchar_t> name(MAX_PATH);
  for (;;) {
    const DWORD length = GetModuleFileNameW(self, name.data(), static_cast<DWORD>(name.size()));
    if (length == 0) {
      return refuse(RefusalCode::LibraryUnavailable,
                    std::format("Windows ML : chemin du module (erreur Windows {})", GetLastError()));
    }
    if (length < name.size()) return std::filesystem::path(std::wstring(name.data(), length)).parent_path();
    name.resize(name.size() * 2);
  }
}

std::expected<HMODULE, Refusal> open_beside(const std::filesystem::path& dir, const wchar_t* file) {
  const std::filesystem::path path = dir / file;
  // Les dépendances se cherchent à côté de la bibliothèque, puis dans les dossiers du système.
  HMODULE module = LoadLibraryExW(path.c_str(), nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (module == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("Windows ML : {} ne se charge pas (erreur Windows {})",
                              std::filesystem::path(file).string(), GetLastError()));
  }
  return module;
}

template <class Fn>
bool resolve(HMODULE module, const char* name, Fn& out) {
  // GetProcAddress rend un FARPROC, à convertir vers le type de la fonction déclarée.
  out = reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(module, name)));
  return out != nullptr;
}

std::expected<Catalog, Refusal> resolve_catalog(HMODULE module) {
  Catalog c;
  const bool all = resolve(module, "WinMLEpCatalogCreate", c.create) &&
                   resolve(module, "WinMLEpCatalogRelease", c.release) &&
                   resolve(module, "WinMLEpCatalogEnumProviders", c.enumerate) &&
                   resolve(module, "WinMLEpCatalogFindProvider", c.find) &&
                   resolve(module, "WinMLEpGetVersionSize", c.version_size) &&
                   resolve(module, "WinMLEpGetVersion", c.version) &&
                   resolve(module, "WinMLEpGetLibraryPathSize", c.path_size) &&
                   resolve(module, "WinMLEpGetLibraryPath", c.path) &&
                   resolve(module, "WinMLEpGetReadyState", c.ready_state) &&
                   resolve(module, "WinMLEpEnsureReadyAsync", c.ensure_async) &&
                   resolve(module, "WinMLAsyncGetStatus", c.async_status) &&
                   resolve(module, "WinMLAsyncCancel", c.async_cancel) &&
                   resolve(module, "WinMLAsyncClose", c.async_close);
  if (!all)
    return refuse(RefusalCode::LibraryUnavailable, "Windows ML : catalogue sans les fonctions attendues");
  return c;
}

std::expected<const Loaded*, Refusal> loaded() {
  static std::mutex mutex;
  static std::optional<Loaded> once;  // sous mutex ; une fois chargées, les bibliothèques restent
  const std::scoped_lock lock(mutex);
  if (once) return &*once;
  auto dir = own_directory();
  if (!dir) return std::unexpected(dir.error());
  auto ort = open_beside(*dir, kOrtLibrary);
  if (!ort) return std::unexpected(ort.error());
  using GetApiBase = const OrtApiBase*(ORT_API_CALL*)();
  GetApiBase get_base = nullptr;
  if (!resolve(*ort, "OrtGetApiBase", get_base))
    return refuse(RefusalCode::LibraryUnavailable, "Windows ML : onnxruntime.dll sans OrtGetApiBase");
  const OrtApiBase* base = get_base();
  const OrtApi* api = base != nullptr ? base->GetApi(ORT_API_VERSION) : nullptr;
  if (api == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("Windows ML : onnxruntime.dll {} sans l'API {}",
                              base != nullptr ? base->GetVersionString() : "?", ORT_API_VERSION));
  }
  auto winml = open_beside(*dir, kWinmlLibrary);
  if (!winml) return std::unexpected(winml.error());
  auto catalog = resolve_catalog(*winml);
  if (!catalog) return std::unexpected(catalog.error());
  Ort::InitApi(api);
  once = Loaded{.ort = base, .catalog = *catalog};
  return &*once;
}

// Une chaîne du catalogue : sa taille, puis son contenu (terminé par un nul, compté ou non).
template <class SizeFn, class TextFn>
std::expected<std::string, HRESULT> text_of(WinMLEpHandle ep, SizeFn size_fn, TextFn text_fn) {
  std::size_t size = 0;
  if (const HRESULT hr = size_fn(ep, &size); FAILED(hr)) return std::unexpected(hr);
  std::string text(size + 1, '\0');
  std::size_t used = 0;
  if (const HRESULT hr = text_fn(ep, text.size(), text.data(), &used); FAILED(hr)) return std::unexpected(hr);
  text.resize(std::min(used, text.size()));
  text.resize(std::strlen(text.c_str()));
  return text;
}

// Un chemin UTF-8 du catalogue.
std::filesystem::path utf8_path(const std::string& text) {
  return std::filesystem::path(std::u8string(text.begin(), text.end()));
}

std::string lower(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

// Les fournisseurs du catalogue, et le premier dont le nom contient key
// (« migraphx » : « MIGraphXExecutionProvider »).
struct Found {
  std::string_view key;
  std::string name;
  std::string family;  // son paquet, pour la recherche par nom
  std::vector<std::string> all;
};

BOOL CALLBACK visit(WinMLEpHandle /*ep*/, const WinMLEpInfo* info, void* context) {
  auto* found = static_cast<Found*>(context);
  if (info == nullptr || info->name == nullptr) return TRUE;
  found->all.emplace_back(info->name);
  if (found->name.empty() && lower(info->name).contains(found->key)) {
    found->name = info->name;
    found->family = info->packageFamilyName != nullptr ? info->packageFamilyName : "";
  }
  return TRUE;
}

// Le catalogue ouvert, rendu à la destruction.
class OpenCatalog {
 public:
  explicit OpenCatalog(const Catalog& fns) : fns_(fns) {}
  OpenCatalog(const OpenCatalog&) = delete;
  OpenCatalog& operator=(const OpenCatalog&) = delete;
  OpenCatalog(OpenCatalog&&) = delete;
  OpenCatalog& operator=(OpenCatalog&&) = delete;
  ~OpenCatalog() {
    if (handle_ != nullptr) fns_.release(handle_);
  }

  std::expected<void, Refusal> open() {
    if (const HRESULT hr = fns_.create(&handle_); FAILED(hr)) {
      handle_ = nullptr;
      return refuse(
          RefusalCode::LibraryUnavailable,
          std::format("Windows ML : le catalogue des fournisseurs ne s'ouvre pas ({})", hresult(hr)));
    }
    return {};
  }

  // Le fournisseur : son nom exact parmi ceux du catalogue, puis sa poignée.
  std::expected<WinMLEpHandle, Refusal> provider(const Kind& kind) {
    Found found{.key = kind.key};
    if (const HRESULT hr = fns_.enumerate(handle_, &visit, &found); FAILED(hr)) {
      return refuse(RefusalCode::LibraryUnavailable,
                    std::format("Windows ML : catalogue illisible ({})", hresult(hr)));
    }
    if (found.name.empty()) {
      std::string offered;
      for (const std::string& name : found.all) offered += (offered.empty() ? "" : ", ") + name;
      return refuse(RefusalCode::LibraryUnavailable,
                    std::format("Windows ML ne propose pas {} sur cette machine ({} ; proposés : {})",
                                kind.label, kind.conditions, offered.empty() ? "aucun" : offered));
    }
    WinMLEpHandle ep = nullptr;
    const HRESULT hr =
        fns_.find(handle_, found.name.c_str(), found.family.empty() ? nullptr : found.family.c_str(), &ep);
    if (FAILED(hr) || ep == nullptr) {
      return refuse(
          RefusalCode::LibraryUnavailable,
          std::format("Windows ML : {} introuvable dans le catalogue ({})", found.name, hresult(hr)));
    }
    return ep;
  }

  Provider describe(WinMLEpHandle ep) const {
    Provider p;
    p.version = text_of(ep, fns_.version_size, fns_.version).value_or("?");
    WinMLEpReadyState state = WinMLEpReadyState_NotPresent;
    p.ready = SUCCEEDED(fns_.ready_state(ep, &state)) && state == WinMLEpReadyState_Ready;
    if (p.ready) {
      auto path = text_of(ep, fns_.path_size, fns_.path);
      if (path && !path->empty()) {
        p.library = utf8_path(*path);
      } else {
        p.ready = false;
      }
    }
    return p;
  }

 private:
  const Catalog& fns_;
  WinMLEpCatalogHandle handle_ = nullptr;
};

// L'attente d'une installation : rappels du catalogue (fil quelconque) vers le fil de l'appelant.
struct Install {
  std::mutex mutex;
  std::condition_variable changed;
  bool done = false;
  std::atomic<double> progress{0.0};
};

void CALLBACK install_done(WinMLAsyncBlock* async) {
  auto* install = static_cast<Install*>(async->context);
  // Signalé sous le verrou : l'attente ne reprend, et ne détruit l'état, qu'une fois le verrou rendu.
  const std::scoped_lock lock(install->mutex);
  install->done = true;
  install->changed.notify_all();
}

void CALLBACK install_progress(WinMLAsyncBlock* async, double percent) {
  static_cast<Install*>(async->context)->progress.store(percent / 100.0);
}

}  // namespace

std::expected<void, Refusal> load_libraries() {
  auto libraries = loaded();
  if (!libraries) return std::unexpected(libraries.error());
  return {};
}

std::string ort_version() {
  auto libraries = loaded();
  return libraries ? std::string((*libraries)->ort->GetVersionString()) : std::string();
}

std::expected<Provider, Refusal> find_provider(Ep which) {
  auto libraries = loaded();
  if (!libraries) return std::unexpected(libraries.error());
  OpenCatalog catalog((*libraries)->catalog);
  if (auto ok = catalog.open(); !ok) return std::unexpected(ok.error());
  auto ep = catalog.provider(kind_of(which));
  if (!ep) return std::unexpected(ep.error());
  return catalog.describe(*ep);
}

std::expected<Provider, Refusal> ensure_provider(Ep which, const std::function<void(double)>& progress,
                                                 const std::stop_token& stop) {
  auto libraries = loaded();
  if (!libraries) return std::unexpected(libraries.error());
  const Kind kind = kind_of(which);
  const Catalog& fns = (*libraries)->catalog;
  OpenCatalog catalog(fns);
  if (auto ok = catalog.open(); !ok) return std::unexpected(ok.error());
  auto ep = catalog.provider(kind);
  if (!ep) return std::unexpected(ep.error());
  if (Provider p = catalog.describe(*ep); p.ready) return p;

  Install install;
  WinMLAsyncBlock async{};
  async.context = &install;
  async.callback = &install_done;
  async.progress = &install_progress;
  if (const HRESULT hr = fns.ensure_async(*ep, &async); FAILED(hr)) {
    return refuse(
        RefusalCode::DeviceUnavailable,
        std::format("Windows ML : l'installation de {} ne démarre pas ({})", kind.label, hresult(hr)));
  }
  bool cancelled = false;
  {
    std::unique_lock lock(install.mutex);
    while (!install.changed.wait_for(lock, std::chrono::milliseconds(200), [&] { return install.done; })) {
      lock.unlock();
      if (progress) progress(install.progress.load());
      if (stop.stop_requested() && !cancelled) {
        cancelled = true;
        (void)fns.async_cancel(&async);
      }
      lock.lock();
    }
  }
  const HRESULT status = fns.async_status(&async, FALSE);
  fns.async_close(&async);
  if (cancelled && FAILED(status))
    return refuse(RefusalCode::Cancelled, std::format("installation de {} annulée", kind.label));
  if (FAILED(status)) {
    return refuse(RefusalCode::DeviceUnavailable,
                  std::format("Windows ML : l'installation de {} échoue ({})", kind.label, hresult(status)));
  }
  Provider p = catalog.describe(*ep);
  if (!p.ready) {
    return refuse(
        RefusalCode::DeviceUnavailable,
        std::format("Windows ML : {} installé mais pas prêt (redémarrer Windows peut être nécessaire)",
                    kind.label));
  }
  return p;
}

#endif

}  // namespace ajp::gpu::winml_rt
