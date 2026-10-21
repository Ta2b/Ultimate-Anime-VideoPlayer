// Bibliothèque libass : messages vers le journal, polices des pièces jointes.
// Elle doit survivre aux pistes et aux moteurs de rendu créés avec elle.
#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <string_view>

#include "core/types.h"

struct ass_library;

namespace spdlog {
class logger;
}

namespace ajp::subs {

// Taille maximale d'une police jointe.
inline constexpr std::size_t kMaxFontBytes = std::size_t{64} << 20;

class AssLibrary {
 public:
  static std::expected<AssLibrary, Refusal> create();

  // Police d'une pièce jointe Matroska. libass la copie ; les moteurs de rendu
  // la trouvent par son nom de famille dès leur image suivante.
  // Refus : nom vide, données vides ou au-delà de kMaxFontBytes.
  std::expected<void, Refusal> add_font(std::string_view name, std::span<const std::byte> data);

  ass_library* get() const { return library_.get(); }

 private:
  struct Free {
    void operator()(ass_library* library) const;
  };
  AssLibrary(std::shared_ptr<spdlog::logger> log, std::unique_ptr<ass_library, Free> library);

  std::shared_ptr<spdlog::logger> log_;  // reçoit les messages de libass : survit à library_
  std::unique_ptr<ass_library, Free> library_;
};

}  // namespace ajp::subs
