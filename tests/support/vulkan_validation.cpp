#include "vulkan_validation.h"

#include <cstdio>
#include <cstdlib>
#include <format>
#include <mutex>

#include <catch2/interfaces/catch_interfaces_capture.hpp>

namespace ajp::test {
namespace {

std::mutex& errors_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<std::string>& errors_store() {
  static std::vector<std::string> errors;
  return errors;
}

std::string current_test() {
  return Catch::getResultCapture().getCurrentTestName();
}

// Une variable d'environnement, vide si absente. Sous Windows, _dupenv_s : la CRT déprécie getenv
// (C4996), une erreur avec /WX.
std::string env_value(const char* name) {
#ifdef _WIN32
  char* value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) return {};
  std::string text(value);
  std::free(value);
  return text;
#else
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string{value};
#endif
}

}  // namespace

ValidationMode validation_mode() {
  // Les tests lisent l'environnement ; src/ jamais (tools/ci/check-no-getenv.sh).
  const std::string text = env_value("AJP_TEST_VULKAN_VALIDATION");
  if (text == "off" || text == "0") return ValidationMode::Off;
  if (text == "require") return ValidationMode::Require;
  return ValidationMode::IfPresent;
}

void note_validation_error(std::string_view test, std::string_view message) {
  const std::lock_guard lock(errors_mutex());
  errors_store().push_back(std::format("{} : {}", test, message));
}

void note_validation_error(std::string_view message) {
  note_validation_error(current_test(), message);
}

ValidationSink validation_sink() {
  return [test = current_test()](bool error, std::string_view message) {
    if (error) {
      note_validation_error(test, message);
    } else {
      std::fprintf(stderr, "%s : %.*s\n", test.c_str(), static_cast<int>(message.size()), message.data());
    }
  };
}

std::vector<std::string> validation_errors() {
  const std::lock_guard lock(errors_mutex());
  return errors_store();
}

bool layer_missing(const Refusal& refusal) {
  return refusal.code == RefusalCode::FeatureUnavailable &&
         refusal.detail.find("VK_LAYER_KHRONOS_validation") != std::string::npos;
}

void warn_without_layer(std::string_view why) {
  static std::once_flag once;
  std::call_once(once, [why] {
    std::fprintf(stderr,
                 "tests sans la couche de validation Vulkan (%.*s) ; AJP_TEST_VULKAN_VALIDATION=require "
                 "pour l'exiger\n",
                 static_cast<int>(why.size()), why.data());
  });
}

}  // namespace ajp::test
