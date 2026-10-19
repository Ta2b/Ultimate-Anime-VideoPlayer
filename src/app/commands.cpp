#include "app/commands.h"

#include <format>
#include <system_error>
#include <type_traits>
#include <variant>

#include "app/system_report.h"
#include "core/model_index.h"
#include "vendor_hdr/sdk_kind.h"
#include "vendor_hdr/sdk_store.h"
#include "vendor_hdr/sdk_version.h"

namespace ajp::app {
namespace {

namespace fs = std::filesystem;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::expected<void, Refusal> require_media(const fs::path& file) {
  std::error_code error;
  if (!fs::is_regular_file(file, error)) {
    return refuse(RefusalCode::MediaUnreadable, std::format("{} : fichier absent ou illisible", utf8(file)));
  }
  return {};
}

std::unexpected<Refusal> not_yet(CommandKind kind) {
  return refuse(
      RefusalCode::CommandUnavailable,
      std::format("« ajp {} » arrive avec les modules de rendu et d'affichage ; en attendant, « ajp "
                  "engines » prépare les moteurs",
                  command_name(kind)));
}

std::expected<void, Refusal> run_export(const ExportArgs& args) {
  if (auto r = require_media(args.input); !r) return r;
  std::error_code error;
  if (fs::exists(args.output, error) && !args.overwrite) {
    return refuse(RefusalCode::FileUnwritable,
                  std::format("{} existe déjà ; --overwrite pour le remplacer", utf8(args.output)));
  }
  return not_yet(CommandKind::Export);
}

vendor_hdr::HostOs host_of(core::Os os) {
  switch (os) {
    case core::Os::Windows:
      return vendor_hdr::HostOs::Windows;
    case core::Os::MacOs:
      return vendor_hdr::HostOs::MacOs;
    case core::Os::Linux:
      return vendor_hdr::HostOs::Linux;
  }
  return vendor_hdr::HostOs::Linux;
}

core::EngineId engine_of(vendor_hdr::SdkKind kind) {
  return kind == vendor_hdr::SdkKind::RtxVideo ? core::EngineId::RtxVideoSdk : core::EngineId::MaxineVfx;
}

// Copie et vérifie le SDK (vendor_hdr::SdkStore), puis le note installé et préféré pour RTX HDR.
std::expected<void, Refusal> import_sdk(const ImportSdkArgs& args, const AppContext& context) {
  const std::optional<vendor_hdr::SdkKind> kind = vendor_hdr::sdk_for(host_of(context.os));
  if (!kind) return refuse(RefusalCode::HdrUnavailable, "aucun SDK TrueHDR n'existe pour macOS");
  *context.out << std::format("Import de {} depuis {}…\n", vendor_hdr::display_name(*kind),
                              utf8(args.folder));
  vendor_hdr::SdkStore store(context.paths.user_data, *kind);
  auto status = store.import(args.folder, context.stop);
  if (!status) return std::unexpected(status.error());
  const std::string version = status->installed ? status->installed->version().text() : std::string("?");
  *context.out << std::format("{} {} importé et choisi pour RTX HDR.\n", vendor_hdr::display_name(*kind),
                              version);

  auto settings = core::load_settings(context.paths.settings());
  if (!settings) return std::unexpected(settings.error());
  const core::SystemInfo system = to_system_info(detect_system(*context.loader, context.os));
  return core::save_settings(context.paths.settings(), record_installed(*settings, system, engine_of(*kind)));
}

std::string version_text() {
  return std::format("ajp {}\n", AJP_VERSION);
}

}  // namespace

std::string refusal_text(const Refusal& refusal) {
  return std::format("ajp : {} : {}\n", name(refusal.code), refusal.detail);
}

std::expected<void, Refusal> run_command(const Invocation& invocation, const AppContext& context) {
  return std::visit(
      [&](const auto& args) -> std::expected<void, Refusal> {
        using Args = std::decay_t<decltype(args)>;
        if constexpr (std::is_same_v<Args, HelpArgs>) {
          *context.out << (args.topic ? help_text(*args.topic) : help_text(utf8(context.paths.user_data)));
          return {};
        } else if constexpr (std::is_same_v<Args, VersionArgs>) {
          *context.out << version_text();
          return {};
        } else if constexpr (std::is_same_v<Args, PlayArgs>) {
          if (auto r = require_media(args.file); !r) return r;
          return not_yet(CommandKind::Play);
        } else if constexpr (std::is_same_v<Args, BenchArgs>) {
          if (auto r = require_media(args.file); !r) return r;
          return not_yet(CommandKind::Bench);
        } else if constexpr (std::is_same_v<Args, ExportArgs>) {
          return run_export(args);
        } else if constexpr (std::is_same_v<Args, ImportSdkArgs>) {
          return import_sdk(args, context);
        } else {
          static_assert(std::is_same_v<Args, EnginesArgs>);
          return args.install ? install_engines(args, context) : list_engines(context);
        }
      },
      invocation.command);
}

int run(const Invocation& invocation, const AppContext& context) {
  if (auto r = run_command(invocation, context); !r) {
    *context.err << refusal_text(r.error());
    return kExitRefused;
  }
  return kExitOk;
}

}  // namespace ajp::app
