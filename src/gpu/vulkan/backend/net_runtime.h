// Réseaux confiés à un autre runtime (MIGraphX sur AMD, OpenVINO sur Intel) derrière le backend
// Vulkan : la chaîne (ingestion, rife_input, warp, ring…) garde les noyaux Vulkan sur l'appareil
// de rendu, seul le réseau passe par le runtime du fabricant.
//
// Les tenseurs des réseaux vivent en mémoire hôte visible du GPU (MemoryUse::Readback, en cache
// côté hôte) : le runtime les lit et les écrit par leur projection, après l'attente sur l'hôte
// de la file Vulkan (BackendCaps::needs_host_sync), et rend la main une fois ses sorties écrites.
// Sur un GPU intégré, cette mémoire hôte peut être celle du runtime (Runtime::allocate_host),
// importée par Vulkan : le réseau n'y copie plus rien.
// Avec l'interop (MIGraphX sous Linux), ils vivent en mémoire d'appareil Vulkan exportée et
// importée par le runtime : même attente, sans aller-retour par le bus PCIe. Si le runtime importe
// aussi le sémaphore timeline de la file Vulkan, plus d'attente hôte du tout : le réseau attend
// la chaîne et la chaîne attend le réseau côté GPU (RuntimeNet::run_ordered).
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/vulkan/compute/caps.h"

namespace ajp::gpu::vk::net {

// Un tenseur d'entrée ou de sortie : NCHW dense, dans le type du réseau. Vu par l'hôte (data),
// ou, en interop, par le runtime sur l'appareil (device : la mémoire Vulkan importée).
struct HostBinding {
  std::byte* data = nullptr;
  std::uint64_t bytes = 0;
  TensorType type = TensorType::F16;
  Shape shape;
  void* device = nullptr;
};

// Mémoire d'appareil Vulkan importée par le runtime ; rendue à sa destruction.
class ImportedMemory {
 public:
  virtual ~ImportedMemory() = default;
  virtual void* device() const = 0;  // adresse dans l'espace du runtime (pointeur HIP)
};

// Sémaphore timeline Vulkan importé par le runtime ; rendu à sa destruction.
class ImportedSemaphore {
 public:
  virtual ~ImportedSemaphore() = default;
};

// Mémoire hôte allouée par le runtime, que la chaîne Vulkan importe (VK_EXT_external_memory_host) :
// sur un GPU intégré (OpenVINO, mémoire USM), le réseau lit et écrit les tenseurs de la chaîne
// sans aucune copie. Rendue à sa destruction, après le tampon Vulkan qui la voit.
class HostMemory {
 public:
  virtual ~HostMemory() = default;
  virtual std::byte* data() const = 0;
};

// La forme fixée d'une entrée, par son nom.
struct InputShape {
  std::string name;
  std::vector<std::int64_t> dims;
};

// Un réseau chargé par le runtime.
class RuntimeNet {
 public:
  virtual ~RuntimeNet() = default;
  virtual std::span<const std::string> input_names() const = 0;
  virtual std::span<const std::string> output_names() const = 0;
  // Calcule les sorties, écrites avant le retour. Refus : celui du runtime, nommé.
  virtual std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                           std::span<const HostBinding> outputs) = 0;
  // Comme run, ordonné côté GPU et sans attente hôte : le calcul attend que semaphore atteigne
  // wait et le signale à signal une fois ses sorties écrites ; le retour ne dit que la mise en
  // file. Refus FeatureUnavailable (par défaut, ou liaisons qui passeraient par l'hôte) avant de
  // rien mettre en file : l'appelant fait alors l'étape par run.
  virtual std::expected<void, Refusal> run_ordered(std::span<const HostBinding> /*inputs*/,
                                                   std::span<const HostBinding> /*outputs*/,
                                                   ImportedSemaphore& /*semaphore*/, std::uint64_t /*wait*/,
                                                   std::uint64_t /*signal*/) {
    return std::unexpected(
        Refusal{RefusalCode::FeatureUnavailable, "exécution ordonnée non prise en charge"});
  }
  // Comme run, sans bloquer l'appelant : sur son propre fil, le runtime attend ready() (les
  // entrées écrites : une attente hôte de la file Vulkan, hors du fil qui nourrit le GPU),
  // calcule, puis appelle done(résultat) une fois ses sorties écrites (le signal hôte de l'étape
  // sur la timeline, que la soumission suivante attend côté GPU). L'appelant ne touche ni aux
  // entrées ni aux sorties avant done ; ready et done viennent d'un autre fil, et les appels se
  // suivent dans l'ordre. Refus FeatureUnavailable (par défaut) avant de rien lancer :
  // l'appelant fait alors l'étape par run.
  virtual std::expected<void, Refusal> run_deferred(
      std::span<const HostBinding> /*inputs*/, std::span<const HostBinding> /*outputs*/,
      std::function<std::expected<void, Refusal>()> /*ready*/,
      std::function<void(std::expected<void, Refusal>)> /*done*/) {
    return std::unexpected(
        Refusal{RefusalCode::FeatureUnavailable, "exécution différée non prise en charge"});
  }
};

class Runtime {
 public:
  virtual ~Runtime() = default;
  virtual std::string_view name() const = 0;  // BackendCaps::name : « migraphx », « openvino »
  virtual std::string device_name() const = 0;
  virtual std::string version() const = 0;  // entre dans la clé des artefacts
  // Ce que build fait du modèle, quand cela change sans que version change : entre aussi dans la
  // clé, et une autre recette reconstruit les artefacts. Vide : aucune (la clé d'avant).
  virtual std::string_view recipe() const { return {}; }
  virtual std::string_view extension() const = 0;
  // Compile le modèle, ses entrées fixées à ces formes (vide : celles du fichier, toutes
  // statiques), et l'écrit à path. Refus : ceux du runtime, Cancelled.
  virtual std::expected<void, Refusal> build(const ModelSpec& model, std::span<const InputShape> inputs,
                                             const std::filesystem::path& path, Progress& progress) = 0;
  virtual std::expected<std::unique_ptr<RuntimeNet>, Refusal> load(const std::filesystem::path& path) = 0;

  // Interop (Linux) : le runtime sait-il importer une mémoire Vulkan exportée en descripteur
  // opaque ? Le backend appelle alors use_device_memory(true) avant toute construction si
  // l'appareil Vulkan exporte aussi (VK_KHR_external_memory_fd) : les réseaux lisent et écrivent
  // la mémoire de la chaîne, sans copie par l'hôte.
  virtual bool imports_memory() const { return false; }
  virtual void use_device_memory(bool /*on*/) {}
  // Importe bytes octets ; le runtime prend le descripteur en cas de succès. Refus
  // FeatureUnavailable par défaut.
  virtual std::expected<std::unique_ptr<ImportedMemory>, Refusal> import_memory(int /*fd*/,
                                                                                std::uint64_t /*bytes*/) {
    return std::unexpected(Refusal{RefusalCode::FeatureUnavailable, "import de mémoire non pris en charge"});
  }
  // Mémoire hôte du runtime pour un tenseur (GPU intégré) : bytes octets, adresse multiple de
  // alignment, que le backend importe dans Vulkan. Refus FeatureUnavailable par défaut : le
  // backend prend alors sa propre mémoire hôte, que le runtime copie.
  virtual std::expected<std::unique_ptr<HostMemory>, Refusal> allocate_host(std::uint64_t /*bytes*/,
                                                                            std::uint64_t /*alignment*/) {
    return std::unexpected(
        Refusal{RefusalCode::FeatureUnavailable, "mémoire hôte du runtime non prise en charge"});
  }
  // Importe le sémaphore timeline de la file Vulkan (descripteur opaque) ; le runtime prend le
  // descripteur en cas de succès. Seulement avec use_device_memory(true). Refus
  // FeatureUnavailable par défaut.
  virtual std::expected<std::unique_ptr<ImportedSemaphore>, Refusal> import_semaphore(int /*fd*/) {
    return std::unexpected(
        Refusal{RefusalCode::FeatureUnavailable, "import de sémaphore non pris en charge"});
  }
};

// Crée le runtime sur l'appareil de cette description Vulkan (même UUID, sinon le premier du
// fabricant). Refus DeviceMismatch : appareil d'un autre fabricant, que le backend Vulkan seul
// sert ; LibraryUnavailable, DeviceUnavailable : runtime absent ou sans appareil.
using RuntimeFactory = std::function<std::expected<std::shared_ptr<Runtime>, Refusal>(const DeviceCaps&)>;

// Noms des entrées et des sorties d'un réseau, rangés à côté de son artefact (un par ligne,
// « in » ou « out » puis le nom), pour les runtimes dont l'artefact ne les garde pas.
std::expected<void, Refusal> write_io_names(const std::filesystem::path& artifact,
                                            std::span<const std::string> inputs,
                                            std::span<const std::string> outputs);
struct IoNames {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};
std::expected<IoNames, Refusal> read_io_names(const std::filesystem::path& artifact);

// Publie sous final_path l'artefact construit à staged, avec ses fichiers voisins (même nom
// suivi d'une extension : noms d'E/S, appareil choisi) : chacun par renommage, les noms d'E/S
// en dernier, qui font foi pour read_io_names. Un arrêt ou un second processus ne laisse donc
// jamais un artefact tronqué sous le nom final. Refus CacheIo.
std::expected<void, Refusal> publish_artifact(const std::filesystem::path& staged,
                                              const std::filesystem::path& final_path);

// Efface l'artefact et ses fichiers voisins, pour que la construction suivante le refasse.
// Ce qui ne s'efface pas reste ; l'appelant a déjà son refus à rendre.
void discard_artifact(const std::filesystem::path& artifact);

// Copie count éléments de from vers to en convertissant entre F32 et F16 (même type : une copie).
// Refus FormatUnsupported : un type entier différent de l'autre.
std::expected<void, Refusal> copy_converted(const std::byte* from, TensorType from_type, std::byte* to,
                                            TensorType to_type, std::size_t count);

// Les noms de l'interface d'un fichier ONNX, lus par gpu_ir.
std::expected<IoNames, Refusal> onnx_io_names(const std::filesystem::path& onnx);

// Les dimensions symboliques des entrées d'un fichier ONNX fixées par ces formes : (symbole,
// valeur), une fois chacun. Refus InputMismatch : un symbole qui recevrait deux valeurs.
using DimOverride = std::pair<std::string, std::int64_t>;
std::expected<std::vector<DimOverride>, Refusal> symbolic_overrides(const std::filesystem::path& onnx,
                                                                    std::span<const InputShape> inputs);

// Le fichier ONNX contient-il un nœud de ce type (domaine par défaut) ? Refus : lecture, décodage.
std::expected<bool, Refusal> onnx_uses_op(const std::filesystem::path& onnx, std::string_view op_type);

}  // namespace ajp::gpu::vk::net
