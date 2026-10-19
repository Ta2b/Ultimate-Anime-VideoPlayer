// Emplacements de capacité fixe, comptés par référence : ce que le thread gpu garde d'une
// image à l'autre (tenseurs, contenus d'image) sans allocation hôte une fois en régime. Les
// emplacements se créent au premier besoin, jusqu'à la capacité, puis se recyclent.
// Un seul thread : les compteurs ne sont pas atomiques.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace ajp::gpu {

// Sans kResetOnFree, la valeur d'un emplacement libre reste en place pour être reprise
// (un tenseur alloué) ; avec, elle revient à T{} dès sa dernière référence rendue (un
// contenu, qui rend alors ses propres références).
template <class T, bool kResetOnFree>
class RefSlots {
 public:
  class Ref {
   public:
    Ref() = default;
    Ref(const Ref& other) : owner_(other.owner_), index_(other.index_) {
      if (owner_ != nullptr) ++owner_->entries_[index_].refs;
    }
    Ref(Ref&& other) noexcept : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_) {}
    Ref& operator=(Ref other) noexcept {
      std::swap(owner_, other.owner_);
      std::swap(index_, other.index_);
      return *this;
    }
    ~Ref() { reset(); }

    void reset() {
      if (owner_ == nullptr) return;
      std::exchange(owner_, nullptr)->release(index_);
    }

    explicit operator bool() const { return owner_ != nullptr; }
    T& operator*() const { return owner_->entries_[index_].value; }
    T* operator->() const { return &owner_->entries_[index_].value; }
    friend bool operator==(const Ref& a, const Ref& b) {
      return a.owner_ == b.owner_ && (a.owner_ == nullptr || a.index_ == b.index_);
    }

   private:
    friend class RefSlots;
    Ref(RefSlots* owner, std::size_t index) : owner_(owner), index_(index) {
      ++owner_->entries_[index_].refs;
    }

    RefSlots* owner_ = nullptr;
    std::size_t index_ = 0;
  };

  explicit RefSlots(std::size_t capacity) : capacity_(capacity) { entries_.reserve(capacity); }
  // Les références pointent sur l'objet : il ne bouge pas et meurt après elles.
  RefSlots(const RefSlots&) = delete;
  RefSlots& operator=(const RefSlots&) = delete;
  ~RefSlots() = default;

  // Un emplacement libre dont la valeur satisfait match, repris tel quel.
  template <class Match>
  std::optional<Ref> take_free(Match match) {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].refs == 0 && match(std::as_const(entries_[i].value))) return Ref(this, i);
    }
    return std::nullopt;
  }

  // Y a-t-il de quoi ranger une valeur neuve : un emplacement à créer, ou un libre à remplacer ?
  bool can_put() const { return entries_.size() < capacity_ || first_free().has_value(); }

  // Range value dans un emplacement neuf tant que la capacité le permet, sinon à la place de
  // la valeur d'un emplacement libre ; rien si tous sont pris.
  std::optional<Ref> put(T value) {
    if (entries_.size() < capacity_) {
      entries_.push_back({.value = std::move(value), .refs = 0});
      return Ref(this, entries_.size() - 1);
    }
    const std::optional<std::size_t> free = first_free();
    if (!free) return std::nullopt;
    entries_[*free].value = std::move(value);
    return Ref(this, *free);
  }

  std::size_t created() const { return entries_.size(); }
  std::size_t in_use() const {
    std::size_t count = 0;
    for (const Entry& entry : entries_) count += entry.refs > 0 ? 1U : 0U;
    return count;
  }

 private:
  struct Entry {
    T value;
    std::uint32_t refs = 0;
  };

  void release(std::size_t index) {
    Entry& entry = entries_[index];
    --entry.refs;
    if constexpr (kResetOnFree) {
      if (entry.refs == 0) entry.value = T{};
    }
  }

  std::optional<std::size_t> first_free() const {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].refs == 0) return i;
    }
    return std::nullopt;
  }

  std::vector<Entry> entries_;  // réservé à la capacité : les adresses ne bougent pas
  std::size_t capacity_;
};

// File de capacité fixe, sans allocation : un tableau tournant.
template <class T, std::size_t N>
class FixedQueue {
 public:
  bool empty() const { return count_ == 0; }
  bool full() const { return count_ == N; }
  std::size_t size() const { return count_; }

  // false si la file est pleine : value n'est pas prise.
  bool push(T value) {
    if (full()) return false;
    items_[(head_ + count_) % N] = std::move(value);
    ++count_;
    return true;
  }
  T& front() { return items_[head_]; }
  T& operator[](std::size_t i) { return items_[(head_ + i) % N]; }
  // Retire la tête ; sa place revient à T{}, ce qui rend ce qu'elle tenait.
  T pop() {
    T value = std::exchange(items_[head_], T{});
    head_ = (head_ + 1) % N;
    --count_;
    return value;
  }
  void clear() {
    while (!empty()) pop();
  }

 private:
  std::array<T, N> items_{};
  std::size_t head_ = 0;
  std::size_t count_ = 0;
};

}  // namespace ajp::gpu
