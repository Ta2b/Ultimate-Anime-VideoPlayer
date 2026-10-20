// La couche de validation Vulkan dans les tests. Les bancs qui créent un appareil la demandent
// selon AJP_TEST_VULKAN_VALIDATION : « off » jamais ; « on » (le défaut) quand elle est installée,
// et sans elle sinon, avec un avertissement ; « require » : son absence fait échouer le test, ce
// que la CI et le préflight exigent. Chaque erreur de la couche est notée sous le nom du test
// courant, et le main de Catch2 (vulkan_test_main.cpp) fait échouer l'exécutable s'il en reste
// une : une erreur de validation est un échec, même quand le test passait.
#pragma once

#include <expected>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"

namespace ajp::test {

enum class ValidationMode : std::uint8_t { Off, IfPresent, Require };
ValidationMode validation_mode();

using ValidationSink = std::function<void(bool error, std::string_view message)>;
// Relais pour CreateOptions::validation_sink ou BackendOptions::validation_sink : les erreurs
// sont notées sous le nom du test courant (lu à la création), les avertissements écrits sur
// stderr. Appelable depuis n'importe quel thread.
ValidationSink validation_sink();
// Note une erreur sous le nom du test courant (thread du test), ou sous un nom donné.
void note_validation_error(std::string_view message);
void note_validation_error(std::string_view test, std::string_view message);
std::vector<std::string> validation_errors();

// Le refus qui dit que la couche manque (FeatureUnavailable nommant VK_LAYER_KHRONOS_validation).
bool layer_missing(const Refusal& refusal);
// Dit une fois, sur stderr, que les tests tournent sans la couche.
void warn_without_layer(std::string_view why);

// Crée avec la couche selon le mode : create(validation, sink) rend un std::expected<T, Refusal>.
// Quand la couche manque sans être exigée, recrée sans elle.
template <class Create>
auto with_validation(Create&& create) {
  const ValidationMode mode = validation_mode();
  if (mode == ValidationMode::Off) return create(false, ValidationSink{});
  auto made = create(true, validation_sink());
  if (made || mode == ValidationMode::Require || !layer_missing(made.error())) return made;
  warn_without_layer(made.error().detail);
  return create(false, ValidationSink{});
}

}  // namespace ajp::test
