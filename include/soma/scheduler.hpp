#pragma once

// Soma — the step-major scheduler. Concurrency is a first-class primitive here,
// not a FIFO in front of a single-sequence engine.
//
//   for each step {
//       collect ready sequences   -> ragged batch of (seq*, token)
//       one batched forward       -> dense GEMMs + union MoE
//       scatter outputs           -> per-sequence sampling, stop checks
//   }
//
// Decode rows and prefill rows are just rows.

#include "soma/f32_model.hpp"
#include "soma/model.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace soma {

class MemoryHierarchy;
class KvCheckpointStore;

struct SchedulerConfig {
    std::uint32_t max_batch = 0; ///< 0 = derive from the cache-aware gate
    std::uint32_t kv_slots = 4;
    std::uint32_t ctx_size = 4096;

    /// Chunked prefill interleaves with decode in the SAME union forward.
    std::uint32_t prefill_chunk_tokens = 512;

    /// Fairness cap: max prefill rows one sequence may contribute per step, so a
    /// long prompt cannot starve interactive turns.
    std::uint32_t max_prefill_rows_per_seq = 256;

    /// Speculation is disabled when batch > 1 in v1: batching already amortizes
    /// the reads speculation was buying, and the interaction between draft
    /// acceptance and batch composition is a second-order problem. Grammar-forced
    /// drafts compose more easily and stay on.
    bool enable_speculation = true;
};

/// Why admit() refused. Distinguished because the remedies differ: Thrash means
/// wait, NoKvSlot means preempt, ContextTooLong means fail the request.
enum class AdmitRejection : std::uint8_t {
    None = 0,
    NoKvSlot,
    Thrash,
    ContextTooLong,
    ShuttingDown,
};

struct SeqRequest {
    std::vector<TokenId> prompt;
    SamplerState sampler{};
    Determinism determinism = Determinism::Batched;
    std::uint32_t max_tokens = 0;
    std::vector<std::string> stop_strings;

    /// Optional: replay this persisted checkpoint as the prompt's prefix.
    ///
    /// The checkpoint carries the token ids it covers, and admit() VERIFIES they
    /// are a prefix of `prompt` before attaching the cache. A mismatch is not an
    /// error — it is a cold start with a log line — because the alternative,
    /// attaching a cache that belongs to different tokens, produces fluent
    /// output about a context nobody supplied and nothing detects it.
    std::string resume_key;
};

/// Fired on the scheduler thread as tokens are produced. Must not block.
///
/// ALL THREE fire from inside step(), which holds the scheduler's own lock. A
/// callback that calls back into the Scheduler therefore deadlocks. They are the
/// hot path's only outward edge, so they must be short: append and signal, never
/// decode-and-write-a-socket.
using TokenCallback = std::function<void(SeqId, TokenId, bool is_last)>;
using ErrorCallback = std::function<void(SeqId, StatusCode, const char* what)>;

/// A sequence reached its end. The unambiguous completion signal.
///
/// `is_last` on TokenCallback is not enough: a sequence can end without
/// producing a token — a prompt that fills the context during prefill does —
/// and a caller keying completion off `is_last` would wait forever.
using FinishCallback = std::function<void(SeqId)>;

struct SchedulerStats {
    std::uint32_t active_sequences = 0;
    std::uint32_t queued_sequences = 0;
    std::uint32_t current_batch = 0;

    /// High-water mark of `current_batch` since the engine started.
    ///
    /// `current_batch` is an INSTANT. Asking whether the engine ever batched by
    /// sampling it is a race the observer usually wins and sometimes loses: the
    /// sampler is an HTTP round-trip, so its real interval is milliseconds, and a
    /// batch that forms and drains between two samples leaves no trace. That is
    /// what D43 was — `soma_engine_g5` failing roughly 1 run in 25 on "the engine
    /// really did batch them", with the engine having done nothing wrong.
    ///
    /// Monotonic, so one read after the fact answers the question the sampling
    /// was approximating.
    std::uint32_t max_batch_seen = 0;
    std::uint32_t effective_max_batch = 0; ///< after the cache-aware gate
    std::uint32_t prefill_rows_last_step = 0;
    std::uint32_t decode_rows_last_step = 0;

    /// The payoff, made observable: unique experts read per step vs. the naive
    /// count (rows × top_k). A ratio near 1.0 means the union is buying nothing
    /// and something upstream is wrong.
    std::uint32_t unique_experts_last_step = 0;
    std::uint32_t naive_expert_reads_last_step = 0;

    std::uint64_t steps = 0;
    std::uint64_t tokens_out = 0;
    std::uint64_t preemptions = 0;
};

class Scheduler {
public:
    Scheduler();
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    ~Scheduler();

    /// Drive the scheduler against the fp32 reference model.
    ///
    /// The path that exists at G3. Every conformance gate is expressed against
    /// F32Model, so scheduling it directly is what lets "a batch of N sequences
    /// produces the same tokens as N separate runs" be checked against the same
    /// numbers the ladder already trusts. `memory` may be null for a resident
    /// model. This is the only entry point; the `ModelState` overload that once
    /// sat beside it belonged to an execution path no family implemented.
    /// `checkpoints` may be null; preempt/resume then report Unsupported rather
    /// than silently doing nothing.
    Status open_f32(const F32Model& model,
                    MemoryHierarchy* memory,
                    const SchedulerConfig& config,
                    KvCheckpointStore* checkpoints = nullptr);

    void close();

    /// No sequences left — the step loop's termination condition.
    bool idle() const;

    void set_token_callback(TokenCallback cb);
    void set_error_callback(ErrorCallback cb);
    void set_finish_callback(FinishCallback cb);

    /// Cache-aware admission control — the constraint most likely to be got
    /// wrong, and getting it wrong inverts the entire benefit of batching.
    ///
    ///     max_batch <= cap_per_layer / expected_unique_experts_per_step
    ///
    /// N sequences x top_k against a small per-layer LRU thrashes: every step
    /// evicts what the next needs, the union degenerates into per-row reads plus
    /// eviction overhead, and throughput falls BELOW single-sequence. The gate
    /// drops only when the expert set is fully resident, at which point there is
    /// no read to amortize and concurrency is bounded by compute instead.
    ///
    /// A fixed max_batch constant would be a bug wearing a config key's clothing.
    Status admit(SeqRequest request, SeqId& out_id, AdmitRejection& out_reason);

    /// One ragged-batch forward. Called in a loop by the serve thread.
    Status step();

    /// Continue a sequence that has finished its turn, with more prompt.
    ///
    /// This is what makes a sequence SESSION-scoped rather than request-scoped. A
    /// finished sequence keeps its KV, its emitted history (the input to the
    /// repetition penalties) and its sampler RNG; extend() appends the new turn
    /// and prefills only the suffix.
    ///
    /// `prompt` is the FULL conversation, not the delta. The scheduler holds the
    /// ids its cache covers and checks they are a prefix — so a client that
    /// edits earlier turns gets a correct cold start rather than a warm cache
    /// describing a conversation that no longer exists. Returns ArchMismatch on
    /// that check, and the caller's remedy is to cancel() and admit() fresh.
    Status extend(SeqId id, std::vector<TokenId> prompt, std::uint32_t max_tokens);

    /// Persist a sequence WITHOUT releasing it — the node's suspend path.
    ///
    /// preempt() releases the KV buffer because reclaiming memory is its whole
    /// point. Suspend is different: the checkpoint must exist before the process
    /// is stopped, and if the stop is abandoned the engine has to still be
    /// serving. One format, two lifetimes.
    Status checkpoint(SeqId id, const std::string& key);

    /// The token ids currently covered by a sequence's KV, in order.
    ///
    /// The authority for what a warm cache actually contains. A caller that keeps
    /// its own copy and trusts it will eventually be wrong about one of them.
    Status sequence_tokens(SeqId id, std::vector<TokenId>& out) const;

    /// Ids of every live sequence, finished ones included — a finished session is
    /// still holding a KV slot and is still resumable.
    std::vector<SeqId> sequence_ids() const;

    /// Evict a sequence by persisting its KV checkpoint; re-admit later.
    ///
    /// Nearly free once KV persistence exists, and it is THE SAME MECHANISM as
    /// warm conversation reopen and as cluster-level slot suspend/restore. One
    /// format, three callers. See kv_checkpoint.hpp.
    Status preempt(SeqId id);
    Status resume(SeqId id);

    Status cancel(SeqId id);

    SchedulerStats stats() const noexcept;

    /// The same values, but NEVER waiting for the work being measured.
    ///
    /// An observer that blocks on the path it observes stops being an observer.
    /// `stats()` takes the lock the step loop holds across an entire forward, so
    /// a telemetry sampler calling it goes quiet exactly while the engine is
    /// busy — measured on a live 7B MoE at **17.3 frames/s idle and 1.3/s during
    /// generation**, a 13x collapse at the one moment an operator is watching.
    /// Defect D11.
    ///
    /// Returns false when the lock was held, leaving `out` untouched so the
    /// caller can reuse its previous value and SAY that it did. A stale number
    /// labelled stale is worth more than a frame that never arrives.
    bool try_stats(SchedulerStats& out) const noexcept;

    /// The gate's current value, recomputed as residency changes. Reported on
    /// /v1/engines/{id}/slots so operators can see WHY concurrency is limited
    /// rather than inferring it from throughput.
    std::uint32_t effective_max_batch() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char* to_string(AdmitRejection reason) noexcept;

} // namespace soma
