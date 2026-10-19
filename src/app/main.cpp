// Point d'entrée d'ajp : analyse la ligne de commande, trouve les dossiers une seule
// fois (SDL3 ou --data-dir), puis exécute la commande. Rien d'autre ne vit ici.
#include <iostream>
#include <memory>
#include <variant>

#include "app/cli.h"
#include "app/commands.h"
#include "app/data_paths.h"
#include "app/library_loader.h"

int main(int argc, char** argv) {
  using namespace ajp::app;
  const auto invocation = parse_invocation(argc, argv);
  if (!invocation) {
    std::cerr << refusal_text(invocation.error()) << "Aide : ajp --help\n";
    return kExitUsage;
  }

  const ajp::core::Os os = host_os();
  auto paths = invocation->data_dir
                   ? data_paths_in(*invocation->data_dir, resources_from_base(os, executable_dir()))
                   : system_data_paths(os);
  const bool needs_paths = !std::holds_alternative<HelpArgs>(invocation->command) &&
                           !std::holds_alternative<VersionArgs>(invocation->command);
  if (!paths && needs_paths) {
    std::cerr << refusal_text(paths.error());
    return kExitRefused;
  }

  const std::unique_ptr<LibraryLoader> loader = system_library_loader();
  const AppContext context{.os = os,
                           .paths = paths.value_or(DataPaths{}),
                           .loader = loader.get(),
                           .out = &std::cout,
                           .err = &std::cerr};
  return run(*invocation, context);
}
