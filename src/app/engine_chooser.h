// Écran « Moteurs » : au premier lancement, puis depuis Paramètres > Moteurs.
//
// État et actions seulement, sans dessin ni E/S : le dessin ImGui (quand render saura
// dessiner) et « ajp engines install » lisent rows() et les totaux, appellent
// toggle() ou set_advanced(), puis confirm() d'un clic. La détection du système, la
// lecture de l'index et de la place libre se font avant ; le téléchargement, après
// (install.h), et record_installed() note dans les réglages ce qui a réussi.
//
// La présélection vient de core::recommend_engines : un utilisateur qui valide sans
// rien toucher obtient les moteurs les plus rapides pour son GPU.
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/engines.h"
#include "core/model_index.h"
#include "core/settings.h"
#include "core/types.h"

namespace ajp::app {

// Moteur que ce système ne fera pas tourner tel quel, faute d'un composant que le lecteur
// n'installe pas (pilote trop ancien, runtime absent) : jamais présélectionné, sa ligne dit
// pourquoi, et il reste installable en mode avancé. Voir engine_blockers() (system_report.h).
struct EngineBlocker {
  core::EngineId engine = core::EngineId::Vulkan;
  std::string reason;  // « pilote CUDA 12.8 trop ancien : TensorRT 11 exige CUDA 13 (…) »
};

struct ChooserInput {
  core::Os os = core::Os::Linux;
  core::SystemInfo system;                       // le premier GPU est celui du rendu
  core::Settings settings;                       // réglages actuels ; first_run_done dit d'où l'on vient
  std::vector<core::EnginePackage> packages;     // engines.lock, toutes plateformes ; vide sans index
  std::vector<core::ModelEntry> missing_models;  // modèles des profils encore à télécharger
  std::uint64_t free_bytes = 0;                  // place libre dans le dossier de données
  std::vector<EngineBlocker> blockers;           // ce que les runtimes du système empêchent
};

struct EngineRow {
  core::EngineOffer offer;
  bool installed = false;  // dans settings.installed_engines
  bool checked = false;
  std::optional<core::EnginePackage> package;  // paquet de ce système dans l'index
  std::string blocker;                         // pourquoi ce système ne le fera pas tourner ; vide sinon
};

// Pourquoi une ligne ne peut pas être cochée, ou décochée pour Bundled.
enum class RowLock : std::uint8_t {
  Free,       // se coche et se décoche
  Bundled,    // livré avec le lecteur : toujours coché
  Advanced,   // grisé (aucun GPU compatible) : se coche seulement en mode avancé
  Blocked,    // GPU compatible, mais pilote ou runtime manquant : se coche seulement en mode avancé
  NoPackage,  // aucun paquet pour ce système dans l'index : rien à télécharger
};

// Ce que le clic « Valider » décide ; install.h l'exécute.
struct ChooserPlan {
  core::Settings settings;                     // first_run_done, moteurs choisis, retraits appliqués
  std::vector<core::EnginePackage> downloads;  // paquets cochés, pas encore installés
  std::vector<core::ModelEntry> models;        // modèles des profils à télécharger
  std::vector<core::EngineId> imports;         // SDK cochés à importer depuis leur dossier (ajp import-sdk)
  std::vector<core::EngineId> removals;        // moteurs installés décochés, à retirer du disque
  std::uint64_t download_bytes = 0;
};

class EngineChooser {
 public:
  explicit EngineChooser(ChooserInput input);

  std::span<const EngineRow> rows() const { return rows_; }
  const EngineRow* row(core::EngineId engine) const;
  // Premier lancement : titre d'accueil et bouton « Commencer » ; sinon, Paramètres > Moteurs.
  bool first_run() const { return !input_.settings.first_run_done; }

  bool advanced() const { return advanced_; }
  // Quitter le mode avancé décoche les moteurs grisés qui ne sont pas installés.
  void set_advanced(bool on);
  // Revient à la présélection (bouton « Recommandé ») : un moteur installé non recommandé se décoche.
  void reset_to_recommended();

  RowLock lock_of(core::EngineId engine) const;
  // Refus FieldOutOfRange dont le détail, en français, dit pourquoi la ligne ne bouge pas.
  std::expected<void, Refusal> set_checked(core::EngineId engine, bool checked);
  std::expected<void, Refusal> toggle(core::EngineId engine);

  std::uint64_t download_bytes() const;  // paquets cochés non installés, plus les modèles
  std::uint64_t free_bytes() const { return input_.free_bytes; }
  bool fits() const { return download_bytes() <= free_bytes(); }

  // Le clic « Valider ». Refus DiskFull si la place manque, avec les deux tailles.
  std::expected<ChooserPlan, Refusal> confirm() const;

 private:
  EngineRow* find(core::EngineId engine);
  bool can_check(const EngineRow& row) const;

  ChooserInput input_;
  std::vector<EngineRow> rows_;
  bool advanced_ = false;
};

// Note un moteur installé (téléchargé ou importé) dans les réglages, et le préfère quand il
// est le plus rapide pour ce GPU (ou le SDK RTX HDR), que l'usage n'a encore que Vulkan et
// qu'aucun blocage ne l'empêche de tourner.
core::Settings record_installed(core::Settings settings, const core::SystemInfo& system,
                                core::EngineId engine, std::span<const EngineBlocker> blockers = {});

}  // namespace ajp::app
