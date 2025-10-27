#pragma once
#include <atomic>
#include <optional>
#include <thread>
#include <vector>

#include "ThreadLockFreeQueue.h"

// Per Thread Context, used for stats
struct WorkerContext {
  uint32_t worker_id;

  std::atomic<uint64_t> batches_stolen{0};

  WorkerContext(uint32_t id) : worker_id(id) {}
};

template <typename Processor, typename WorkItem>
concept WorkProcessor =
    requires(Processor processor, const WorkItem& item, WorkerContext& ctx) {
      { processor.Process(item, ctx) } -> std::same_as<void>;
    };

template <typename WorkItem, WorkProcessor<WorkItem> Processor>
class ThreadPool {
 private:
  struct WorkerQueue {
    LockFreeQueue<WorkItem> queue;
    std::atomic<bool> has_work{false};
    WorkerQueue() : queue(2048) {}
  };

  std::vector<std::thread> threads;
  std::vector<std::unique_ptr<WorkerQueue>> worker_queues;
  std::vector<std::unique_ptr<WorkerContext>> worker_contexts;
  std::atomic<bool> running{false};
  std::atomic<uint32_t> active_workers{0};
  const uint32_t num_workers;

  Processor processor;

 public:
  explicit ThreadPool(Processor proc, uint32_t thread_count = 0)
      : processor(std::move(proc)),
        num_workers(thread_count > 0 ? thread_count
                                     : std::thread::hardware_concurrency()) {
    worker_queues.reserve(num_workers);
    worker_contexts.reserve(num_workers);

    for (uint32_t i = 0; i < num_workers; ++i) {
      worker_queues.push_back(std::make_unique<WorkerQueue>());
      worker_contexts.push_back(std::make_unique<WorkerContext>(i));
    }
  }

  ~ThreadPool() { Shutdown(); }

  void Start() {
    running.store(true, std::memory_order_release);
    for (uint32_t i = 0; i < num_workers; ++i) {
      threads.emplace_back(&ThreadPool::WorkerMain, this, i);
    }
  }

  void Shutdown() {
    running.store(false, std::memory_order_release);
    for (auto& thread : threads) {
      if (thread.joinable()) thread.join();
    }
    threads.clear();
  }

  bool SubmitBatch(uint32_t worker_id, const WorkItem& item) {
    if (worker_id >= num_workers) return false;

    bool success = worker_queues[worker_id]->queue.TryEnqueue(item);
    if (success) {
      worker_queues[worker_id]->has_work.store(true, std::memory_order_release);
    }
    return success;
  }

  // Submit to any available worker (load balance)
  bool SubmitBatch(const WorkItem& item) {
    // Try to find worker with least work
    uint32_t best_worker = 0;
    size_t min_size = worker_queues[0]->queue.ApproximateSize();

    for (uint32_t i = 1; i < num_workers; ++i) {
      size_t size = worker_queues[i]->queue.ApproximateSize();
      if (size < min_size) {
        min_size = size;
        best_worker = i;
      }
    }

    return SubmitBatch(best_worker, item);
  }

  void WaitForCompletion() {
    while (HasWork() || active_workers.load(std::memory_order_acquire) > 0) {
      std::this_thread::yield();
    }
  }

  struct Stats {
    uint64_t total_batches_stolen = 0;
  };

  Stats GetStats() const {
    Stats stats;
    for (const auto& ctx : worker_contexts) {
      stats.total_batches_stolen +=
          ctx->batches_stolen.load(std::memory_order_relaxed);
    }
    return stats;
  }

  uint32_t NumWorkers() const { return num_workers; }

 private:
  bool HasWork() const {
    for (const auto& wq : worker_queues) {
      if (wq->queue.ApproximateSize() > 0) return true;
    }
    return false;
  }

  std::optional<WorkItem> GetWork(uint32_t worker_id) {
    // Try own queue first (best cache locality)
    auto item = worker_queues[worker_id]->queue.TryDequeue();
    if (item) return item;

    worker_queues[worker_id]->has_work.store(false, std::memory_order_release);

    // Try stealing from other workers
    for (uint32_t i = 1; i < num_workers; ++i) {
      uint32_t victim = (worker_id + i) % num_workers;
      if (!worker_queues[victim]->has_work.load(std::memory_order_acquire)) {
        continue;
      }

      item = worker_queues[victim]->queue.TryDequeue();
      if (item) {
        worker_contexts[worker_id]->batches_stolen.fetch_add(
            1, std::memory_order_relaxed);
        return item;
      }
    }

    return std::nullopt;
  }

  void WorkerMain(uint32_t worker_id) {
    WorkerContext& ctx = *worker_contexts[worker_id];

    while (running.load(std::memory_order_acquire)) {
      auto work_opt = GetWork(worker_id);

      if (!work_opt) {
        std::this_thread::yield();
        continue;
      }
      active_workers.fetch_add(1, std::memory_order_relaxed);
      processor.Process(*work_opt, ctx);
      active_workers.fetch_sub(1, std::memory_order_relaxed);
    }
  }
};