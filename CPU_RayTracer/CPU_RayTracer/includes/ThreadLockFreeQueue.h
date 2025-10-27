#pragma once
#include <atomic>
#include <cassert>
#include <optional>
#include <vector>

template <typename T>
class LockFreeQueue {
 private:
  struct Cell {
    std::atomic<uint64_t> sequence;
    T data;
  };

  static constexpr size_t CACHE_LINE = 64;

  alignas(CACHE_LINE) std::atomic<uint64_t> enqueue_pos;
  alignas(CACHE_LINE) std::atomic<uint64_t> dequeue_pos;
  alignas(CACHE_LINE) const uint64_t capacity_mask;

  std::vector<Cell> buffer;

 public:
  explicit LockFreeQueue(size_t capacity)
      : enqueue_pos(0),
        dequeue_pos(0),
        capacity_mask(capacity - 1),
        buffer(capacity) {
    assert((capacity & capacity_mask) == 0);
    for (size_t i = 0; i < capacity; ++i) {
      buffer[i].sequence.store(i, std::memory_order_relaxed);
    }
  }

  bool TryEnqueue(const T& item) {
    Cell* cell;
    uint64_t pos = enqueue_pos.load(std::memory_order_relaxed);

    while (true) {
      cell = &buffer[pos & capacity_mask];

      // Acq to ensure we don't reuse a cell
      uint64_t seq = cell->sequence.load(std::memory_order_acquire);
      intptr_t diff = (intptr_t)seq - (intptr_t)pos;
      // Empty Cell
      if (diff == 0) {
        // Ready for write

        // If CAS(Comp&Swap) fails, another thread took the prio, retry
        if (enqueue_pos.compare_exchange_weak(pos, pos + 1,
                                              std::memory_order_relaxed)) {
          break;
        }
      } else if (diff < 0) {
        return false;  // Queue full
      } else {
        // Another thread has advanced further than our position, reload and try
        // again from updated pos
        pos = enqueue_pos.load(std::memory_order_relaxed);
      }
    }

    // Write data (non-atomic, cell is owned)
    cell->data = item;

    // All writes to cell->data must be visible before we publish
    cell->sequence.store(pos + 1, std::memory_order_release);
    return true;
  }

  std::optional<T> TryDequeue() {
    Cell* cell;
    uint64_t pos = dequeue_pos.load(std::memory_order_relaxed);

    while (true) {
      cell = &buffer[pos & capacity_mask];

      uint64_t seq = cell->sequence.load(std::memory_order_acquire);
      intptr_t diff = (intptr_t)seq - (intptr_t)(pos + 1);

      if (diff == 0) {
        // Data ready

        if (dequeue_pos.compare_exchange_weak(pos, pos + 1,
                                              std::memory_order_relaxed)) {
          break;
        }
      } else if (diff < 0) {
        return std::nullopt;  // Queue empty
      } else {
        pos = dequeue_pos.load(std::memory_order_relaxed);
      }
    }
    // Read data (non-atomic, cell is owned)}
    T data = cell->data;
    cell->sequence.store(pos + capacity_mask + 1, std::memory_order_release);
    return data;
  }

  size_t TryEnqueueBatch(const T* items, size_t count) {
    size_t enqueued = 0;
    for (size_t i = 0; i < count; ++i) {
      if (!TryEnqueue(items[i])) break;
      enqueued++;
    }
    return enqueued;
  }

  size_t TryDequeueBatch(T* items, size_t max_count) {
    size_t dequeued = 0;
    for (size_t i = 0; i < max_count; ++i) {
      auto item = TryDequeue();
      if (!item) break;
      items[i] = *item;
      dequeued++;
    }
    return dequeued;
  }

  size_t ApproximateSize() const {
    uint64_t enq = enqueue_pos.load(std::memory_order_relaxed);
    uint64_t deq = dequeue_pos.load(std::memory_order_relaxed);
    return enq - deq;
  }
};