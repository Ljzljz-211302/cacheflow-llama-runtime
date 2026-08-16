#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-benefit-checkpoint.h"

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct blocking_state {
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::string> saves;
    bool entered = false;
    bool released = false;
};

class blocking_store final : public server_benefit_checkpoint_store {
public:
    explicit blocking_store(std::shared_ptr<blocking_state> state) : state_(std::move(state)) {}

    server_benefit_checkpoint_load_result load() override { return {}; }

    bool save(const std::string & payload, std::string &) override {
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->entered = true;
        state_->condition.notify_all();
        state_->condition.wait(lock, [&]() { return state_->released; });
        state_->saves.push_back(payload);
        return true;
    }

private:
    std::shared_ptr<blocking_state> state_;
};

class failing_store final : public server_benefit_checkpoint_store {
public:
    server_benefit_checkpoint_load_result load() override { return {}; }
    bool save(const std::string &, std::string & error) override {
        error = "injected disk failure";
        return false;
    }
};

class throwing_store final : public server_benefit_checkpoint_store {
public:
    server_benefit_checkpoint_load_result load() override {
        throw std::runtime_error("injected load exception");
    }
    bool save(const std::string &, std::string &) override {
        throw std::runtime_error("injected save exception");
    }
};

static fs::path temporary_checkpoint() {
    return fs::temp_directory_path() /
            ("cacheflow-benefit-checkpoint-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
}

int main() {
    const fs::path path = temporary_checkpoint();
    fs::remove(path);
    fs::remove(path.string() + ".tmp");

    // Force the first write to remain in flight while later snapshots arrive:
    // only the newest pending value may survive the bounded queue.
    {
        auto state = std::make_shared<blocking_state>();
        server_benefit_checkpoint checkpoint(std::make_unique<blocking_store>(state));
        checkpoint.enqueue("first");
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->condition.wait(lock, [&]() { return state->entered; });
        }
        checkpoint.enqueue("obsolete");
        checkpoint.enqueue("latest");
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->released = true;
        }
        state->condition.notify_all();
        assert(checkpoint.flush());
        assert((state->saves == std::vector<std::string>{"first", "latest"}));
        assert(checkpoint.snapshot().coalesced == 1);
    }

    // Persistence failure is observable and cannot stall flush/shutdown.
    {
        server_benefit_checkpoint checkpoint(std::make_unique<failing_store>());
        checkpoint.enqueue("state");
        assert(!checkpoint.flush());
        assert(checkpoint.snapshot().save_failures == 1);
    }

    // Store implementations are an exception boundary: a filesystem/library
    // exception must degrade to status, wake flush(), and never kill the server.
    {
        server_benefit_checkpoint checkpoint(std::make_unique<throwing_store>());
        assert(checkpoint.load().status ==
                server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR);
        checkpoint.enqueue("state");
        assert(!checkpoint.flush());
        assert(checkpoint.snapshot().save_failures == 1);
    }

    {
        server_benefit_checkpoint checkpoint(server_benefit_create_file_checkpoint_store(path));
        const auto missing = checkpoint.load();
        assert(missing.status == server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_MISSING);

        // The producer never waits for disk I/O. A one-element latest-value
        // queue coalesces obsolete snapshots instead of growing without bound.
        checkpoint.enqueue("generation-1");
        checkpoint.enqueue("generation-2");
        checkpoint.enqueue("generation-3");
        assert(checkpoint.flush());
        const auto status = checkpoint.snapshot();
        assert(status.saves_completed >= 1);
        assert(status.pending == 0);
        assert(status.enqueued == 3);
        assert(status.coalesced <= 2);
    }

    {
        server_benefit_checkpoint checkpoint(server_benefit_create_file_checkpoint_store(path));
        const auto restored = checkpoint.load();
        assert(restored.status == server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_RESTORED);
        assert(restored.payload == "generation-3");
    }

    // A stale temporary file is never mistaken for committed state.
    {
        std::ofstream stale(path.string() + ".tmp", std::ios::binary | std::ios::trunc);
        stale << "uncommitted";
    }
    {
        server_benefit_checkpoint checkpoint(server_benefit_create_file_checkpoint_store(path));
        const auto restored = checkpoint.load();
        assert(restored.status == server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_RESTORED);
        assert(restored.payload == "generation-3");
    }

    fs::remove(path);
    fs::remove(path.string() + ".tmp");
    return 0;
}
