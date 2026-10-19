#include "present/display_info.h"

#include <format>

#include "core/debug_flags.h"
#include "present/refresh.h"

namespace ajp::present {

namespace {

using core::DisplayInfo;
using core::OutputSpace;
using core::Provenance;

std::unexpected<Refusal> refuse(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, std::move(detail)});
}

OutputSpace space_for(bool hdr) {
  return hdr ? OutputSpace::Pq : OutputSpace::Srgb;
}

// Rafraîchissement retenu : 0 (inconnu) ou sa valeur nominale.
std::expected<void, Refusal> set_refresh(DisplayDetection& out, double hz, Provenance from) {
  if (hz == 0.0) {
    out.refresh.reset();
    out.info.refresh_hz = 0.0;
  } else {
    auto nominal = nominal_refresh(hz);
    if (!nominal) {
      return refuse(std::format("{} (provenance : {})", nominal.error().detail, provenance_name(from)));
    }
    out.refresh = *nominal;
    out.info.refresh_hz = nominal->value();
  }
  out.info.provenance.refresh = from;
  return {};
}

void apply_system(const SystemDisplay& system, DisplayDetection& out) {
  DisplayInfo& d = out.info;
  if (system.width > 0 && system.height > 0) {
    d.width = system.width;
    d.height = system.height;
    d.provenance.size = Provenance::System;
  }
  if (system.hdr) {
    d.hdr = *system.hdr;
    d.space = space_for(d.hdr);
    d.provenance.hdr = d.provenance.space = Provenance::System;
  }
  if (d.hdr && system.peak_nits) {
    d.peak_nits = *system.peak_nits;
    d.provenance.peak = Provenance::System;
  }
  if (system.min_nits) {
    d.min_nits = *system.min_nits;
    d.provenance.min = Provenance::System;
  }
  out.proposed_sdr_white_nits = system.sdr_white_nits;
  out.unknown = system.unknown;
}

std::expected<void, Refusal> apply_profile(const DisplayOverrides& profile, DisplayDetection& out) {
  DisplayInfo& d = out.info;
  constexpr Provenance from = Provenance::Profile;
  if (profile.width || profile.height) {
    if (!profile.width || !profile.height)
      return refuse("écran : largeur et hauteur se forcent ensemble (profil)");
    d.width = *profile.width;
    d.height = *profile.height;
    d.provenance.size = from;
  }
  if (profile.refresh_hz) {
    if (auto set = set_refresh(out, *profile.refresh_hz, from); !set) return set;
  }
  if (profile.hdr) {
    d.hdr = *profile.hdr;
    d.space = space_for(d.hdr);
    d.provenance.hdr = d.provenance.space = from;
  }
  if (profile.vrr) {
    d.vrr = *profile.vrr;
    d.provenance.vrr = from;
  }
  if (profile.peak_nits) {
    d.peak_nits = *profile.peak_nits;
    d.provenance.peak = from;
  }
  if (profile.min_nits) {
    d.min_nits = *profile.min_nits;
    d.provenance.min = from;
  }
  if (profile.sdr_white_nits) {
    d.sdr_white_nits = *profile.sdr_white_nits;
    d.provenance.sdr_white = from;
  }
  if (profile.space) {
    d.space = *profile.space;
    d.provenance.space = from;
  }
  return {};
}

// --display : seuls les champs que la ligne de commande a posés.
std::expected<void, Refusal> apply_command_line(const DisplayInfo& line, DisplayDetection& out) {
  DisplayInfo& d = out.info;
  constexpr Provenance from = Provenance::CommandLine;
  const core::DisplayProvenance& p = line.provenance;
  if (p.size == from) {
    d.width = line.width;
    d.height = line.height;
    d.provenance.size = from;
  }
  if (p.refresh == from) {
    if (auto set = set_refresh(out, line.refresh_hz, from); !set) return set;
  }
  if (p.hdr == from) {
    d.hdr = line.hdr;
    d.provenance.hdr = from;
  }
  if (p.vrr == from) {
    d.vrr = line.vrr;
    d.provenance.vrr = from;
  }
  if (p.peak == from) {
    d.peak_nits = line.peak_nits;
    d.provenance.peak = from;
  }
  if (p.min == from) {
    d.min_nits = line.min_nits;
    d.provenance.min = from;
  }
  if (p.sdr_white == from) {
    d.sdr_white_nits = line.sdr_white_nits;
    d.provenance.sdr_white = from;
  }
  if (p.space == from) {
    d.space = line.space;
    d.provenance.space = from;
  }
  return {};
}

std::expected<void, Refusal> check(const DisplayInfo& d) {
  const core::DisplayProvenance& p = d.provenance;
  if (p.size != Provenance::Default &&
      (!core::limits::kDisplaySide.contains(d.width) || !core::limits::kDisplaySide.contains(d.height))) {
    return refuse(std::format("taille d'écran {}x{} hors de [1, 16384] (provenance : {})", d.width, d.height,
                              provenance_name(p.size)));
  }
  if (auto valid = core::check_display(d); !valid) {
    // check_display nomme le champ ; on y ajoute d'où viennent les luminances.
    return refuse(std::format("{} (provenance : crête {}, noir {}, blanc SDR {})", valid.error().detail,
                              provenance_name(p.peak), provenance_name(p.min), provenance_name(p.sdr_white)));
  }
  return {};
}

}  // namespace

std::string_view provenance_name(Provenance provenance) {
  switch (provenance) {
    case Provenance::Default:
      return "défaut";
    case Provenance::Profile:
      return "profil";
    case Provenance::System:
      return "système";
    case Provenance::CommandLine:
      return "--display";
  }
  return "inconnue";
}

std::expected<DisplayDetection, Refusal> detect_display(const SystemDisplay& system,
                                                        const DisplayOverrides& profile,
                                                        const std::optional<DisplayInfo>& command_line) {
  DisplayDetection out;
  apply_system(system, out);
  if (system.refresh) {
    auto nominal = nominal_refresh(*system.refresh);
    if (!nominal) return refuse(std::format("{} (provenance : système)", nominal.error().detail));
    out.refresh = *nominal;
    out.info.refresh_hz = nominal->value();
    out.info.provenance.refresh = Provenance::System;
  }
  if (auto applied = apply_profile(profile, out); !applied) return std::unexpected(applied.error());
  if (command_line) {
    if (auto applied = apply_command_line(*command_line, out); !applied)
      return std::unexpected(applied.error());
  }
  if (auto valid = check(out.info); !valid) return std::unexpected(valid.error());
  return out;
}

}  // namespace ajp::present
