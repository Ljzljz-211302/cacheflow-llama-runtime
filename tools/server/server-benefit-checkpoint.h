#pragma once

#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

enum class server_benefit_checkpoint_load_status {
    SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_MISSING,
    SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_RESTORED,
    SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR,
};

struct server_benefit_checkpoint_load_result {
    server_benefit_checkpoint_load_status status =
            server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_MISSING;
    std::string payload;
    std::string error;
};

struct server_benefit_checkpoint_io_snapshot {
    uint64_t enqueued = 0;
    uint64_t coalesced = 0;
    uint64_t saves_completed = 0;
    uint64_t save_failures = 0;
    uint64_t pending = 0;
};

// Storage is deliberately narrower than the policy state format. The policy
// owns schema/compatibility validation; stores only provide durable bytes.
class server_benefit_checkpoint_store {
public:
    virtual ~server_benefit_checkpoint_store() = default;
    virtual server_benefit_checkpoint_load_result load() = 0;
    virtual bool save(const std::string & payload, std::string & error) = 0;
};

std::unique_ptr<server_benefit_checkpoint_store> server_benefit_create_file_checkpoint_store(
        const std::filesystem::path & path);

// A single-worker, latest-value queue. enqueue() never performs file I/O and
// never grows an unbounded backlog on the inference thread.
class server_benefit_checkpoint {
public:
    explicit server_benefit_checkpoint(std::unique_ptr<server_benefit_checkpoint_store> store);
    ~server_benefit_checkpoint();

    server_benefit_checkpoint(const server_benefit_checkpoint &) = delete;
    server_benefit_checkpoint & operator=(const server_benefit_checkpoint &) = delete;

    server_benefit_checkpoint_load_result load();
    void enqueue(std::string payload);
    bool flush();
    server_benefit_checkpoint_io_snapshot snapshot() const;

private:
    void run();

    std::unique_ptr<server_benefit_checkpoint_store> store_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable flushed_;
    std::thread worker_;
    std::string pending_payload_;
    uint64_t enqueued_ = 0;
    uint64_t pending_sequence_ = 0;
    uint64_t completed_sequence_ = 0;
    uint64_t coalesced_ = 0;
    uint64_t saves_completed_ = 0;
    uint64_t save_failures_ = 0;
    bool pending_ = false;
    bool writing_ = false;
    bool stopping_ = false;
};
