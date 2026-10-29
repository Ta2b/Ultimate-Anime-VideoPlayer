#include "core/auto_memory.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <iterator>

#include "core/user_overlay.h"

namespace ajp::core {

std::string auto_memory_key(std::string_view file, std::uintmax_t size, std::string_view engine,
                            int work_height, std::string_view gpu) {
  std::string key = std::format("{}|{}|{}|{}|{}", file, size, engine, work_height, gpu);
  for (char& c : key) {
    if (c == '"' || c == '\\' || static_cast<unsigned char>(c) < 0x20 || c == 0x7f) c = '_';
  }
  return key;
}

AutoMemory AutoMemory::parse(std::string_view text) {
  AutoMemory memory;
  std::string key;
  bool have_key = false;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    std::string_view line = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.remove_suffix(1);
    if (line.size() >= 4 && line.starts_with("[\"") && line.ends_with("\"]")) {
      key = std::string(line.substr(2, line.size() - 4));
      have_key = !key.empty();
    } else if (have_key && line.starts_with("step = ")) {
      int centi = 0;
      const std::string_view number = line.substr(7);
      const auto [stop, error] = std::from_chars(number.data(), number.data() + number.size(), centi);
      if (error == std::errc{} && stop == number.data() + number.size() && centi >= 100 && centi <= 600)
        memory.put(key, centi);
      have_key = false;
    }
  }
  return memory;
}

std::string AutoMemory::dump() const {
  std::string text =
      "# Mode Auto : dernier palier tenu par video (centiemes de multiplicateur : 250 = x2,5). Tenu par le "
      "lecteur.\n";
  for (const auto& [key, centi] : entries_) text += std::format("[\"{}\"]\nstep = {}\n", key, centi);
  return text;
}

std::optional<int> AutoMemory::find(std::string_view key) const {
  const auto found = std::ranges::find(entries_, key, &std::pair<std::string, int>::first);
  if (found == entries_.end()) return std::nullopt;
  return found->second;
}

void AutoMemory::put(std::string_view key, int centi) {
  std::erase_if(entries_, [&](const auto& entry) { return entry.first == key; });
  entries_.emplace_back(std::string(key), centi);
  if (entries_.size() > kAutoMemoryEntries)
    entries_.erase(entries_.begin(), entries_.end() - static_cast<std::ptrdiff_t>(kAutoMemoryEntries));
}

namespace {

AutoMemory load(const std::filesystem::path& store) {
  std::ifstream in(store, std::ios::binary);
  if (!in) return {};
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return AutoMemory::parse(text);
}

}  // namespace

std::optional<int> load_auto_step(const std::filesystem::path& store, std::string_view key) {
  return load(store).find(key);
}

std::expected<void, Refusal> store_auto_step(const std::filesystem::path& store, std::string_view key,
                                             int centi) {
  AutoMemory memory = load(store);
  if (memory.find(key) == centi) return {};
  memory.put(key, centi);
  return write_text_atomically(store, memory.dump());
}

}  // namespace ajp::core
