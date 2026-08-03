// Soma — the step-major scheduler.
//
//   for each step {
//       collect ready sequences   -> ragged batch of (seq*, token)
//       one batched forward       -> dense GEMMs + union MoE
//       scatter outputs           -> per-sequence sampling, stop checks
//   }
//
// The inversion this represents: the engine is no longer a single-sequence loop
// with a queue bolted to the front. A step is a batch of ROWS, and which
// sequence a row belongs to is just a field on it. Prefill rows and decode rows
// go through the same forward, in the same call, with no branch between them —
// they differ only in whether the token came from a prompt or from the previous
// step's sample.
//
// This is also where the batch union starts earning its keep for real. Until
// now the "batch" was one prompt's tokens, whose expert sets are correlated;
// rows from independent sequences are independent draws, which is the case the
// union was designed for.

#include "soma/scheduler.hpp"

#include "soma/f32_model.hpp"
#include "soma/kv_cache.hpp"
#include "soma/kv_checkpoint.hpp"
#include "soma/memory_hierarchy.hpp"
#include "soma/plan.hpp"
#include "soma/sampler.hpp"

#include <algorithm>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace soma {

const char* to_string(AdmitRejection reason) noexcept {
    switch (reason) {
    case AdmitRejection::None:
        return "none";
    case AdmitRejection::NoKvSlot:
        return "no_kv_slot";
    case AdmitRejection::Thrash:
        return "thrash";
    case AdmitRejection::ContextTooLong:
        return "context_too_long";
    case AdmitRejection::ShuttingDown:
        return "shutting_down";
    }
    return "unknown";
}

namespace {

/// Tier 2 of the three state tiers: everything owned by ONE sequence.
///
/// Deliberately a plain struct with no synchronisation. Only the scheduler
/// thread touches it, which is what lets the step loop be lock-free over
/// sequence state — the model tier is immutable and the exec tier is per-step
/// scratch, so `seq` is the only mutable per-caller state and it is never shared.
struct Seq {
    SeqId id = 0;
    KvCache kv;

    std::vector<TokenId> prompt;
    std::uint32_t prompt_pos = 0; ///< next prompt token to feed
    TokenId next_token = 0;       ///< the row this sequence contributes when decoding
    bool have_next = false;

    std::uint32_t generated = 0;
    std::uint32_t max_tokens = 0;
    SamplerState sampler{};
    Determinism determinism = Determinism::Batched;

    /// Tokens this sequence has emitted — the input to the repetition and
    /// presence penalties, and per-sequence for the same reason the RNG is.
    std::vector<TokenId> emitted;
    SamplerScratch scratch;

    /// The token ids occupying the cached positions, in order.
    ///
    /// Appended exactly where kv.commit() advances the cache, so the two cannot
    /// drift. This is what a checkpoint persists and what extend() checks a new
    /// prompt against — without it, "resume from this cache" is an act of faith
    /// whose failure mode is fluent output about a context nobody supplied.
    std::vector<TokenId> history;

    /// Finished its turn. The sequence STAYS in the map: its KV, its emitted
    /// history and its sampler RNG are the session, and a session outlives the
    /// request that created it. cancel() is what actually retires one.
    bool finished = false;
    /// KV is on disk and the buffer has been released; the sequence stays in the
    /// map so its prompt, position and sampler survive the round trip.
    bool preempted = false;

    /// Prefill while the prompt has tokens left; decode after. The scheduler
    /// never branches on this — it only decides which token to contribute.
    bool prefilling() const noexcept { return prompt_pos < prompt.size(); }
};

/// Checkpoint key for a sequence.
///
/// Deliberately a plain function rather than an inline string: the same key must
/// be produced by the scheduler, by warm reopen and by cluster restore, and three
/// call sites formatting it independently is how they diverge.
std::string checkpoint_key(SeqId id) {
    return "seq-" + std::to_string(id);
}

} // namespace

struct Scheduler::Impl {
    const F32Model* model = nullptr;
    MemoryHierarchy* memory = nullptr;
    KvCheckpointStore* checkpoints = nullptr;
    SchedulerConfig cfg{};
    bool open = false;

    std::unordered_map<SeqId, std::unique_ptr<Seq>> seqs;
    std::deque<SeqId> ready;   ///< round-robin order
    std::deque<SeqId> waiting; ///< admitted but no KV slot / gated out
    SeqId next_id = 1;

    TokenCallback on_token;
    ErrorCallback on_error;
    FinishCallback on_finish;

    F32Workspace ws;
    std::vector<float> logits;
    SchedulerStats stats{};
    std::uint32_t gate = 1;

    mutable std::mutex mu; ///< guards admit/cancel against stats reads

    /// The cache-aware gate.
    ///
    ///     max_batch <= cap_per_layer / expected_unique_experts_per_step
    ///
    /// N sequences x top_k against a small per-layer LRU thrashes: every step
    /// evicts what the next needs, the union degenerates into per-row reads plus
    /// eviction overhead, and throughput falls BELOW single-sequence. Recomputed
    /// rather than cached as a constant, because residency changes underneath it.
    std::uint32_t compute_gate() const {
        if (cfg.max_batch > 0) return cfg.max_batch;

        // No hierarchy at all means a RESIDENT model: every expert is already in
        // memory, there is no read to amortise, and concurrency is bounded by
        // compute rather than by the cache.
        if (memory == nullptr) return cfg.kv_slots;

        const auto cap = memory->cap_per_layer();

        // cap == 0 is MAXIMUM pressure, not the absence of it.
        //
        // cap_per_layer is `budget / (expert_bytes * n_moe_layers)`, so zero means
        // the cache cannot hold even one expert per layer — the most constrained
        // state there is. An earlier version read it as "unbounded" and returned
        // the full kv_slots, so the tightest possible cache produced the widest
        // possible batch: the gate reversed exactly where it was needed. It is a
        // one-line confusion between "no limit recorded" and "limit is zero", and
        // it inverts the safety property the gate exists for.
        if (cap == 0) return 1;

        const auto E = std::max<std::uint32_t>(1, model->arch.router.n_experts);
        const auto k = std::max<std::uint32_t>(1, model->arch.router.top_k);
        const double p = static_cast<double>(k) / static_cast<double>(E);

        // Grow while the expected unique set still fits the per-layer cap. The
        // expectation is the coupon-collector one and is CONSERVATIVE by 1.25-1.56x
        // against measurement (real routers concentrate, uniform draws do not), so
        // this throttles earlier than strictly necessary. That is the safe
        // direction: over-admitting thrashes, under-admitting merely queues.
        std::uint32_t best = 1;
        for (std::uint32_t rows = 1; rows <= cfg.kv_slots; ++rows) {
            const double uniq = static_cast<double>(E) * (1.0 - std::pow(1.0 - p, rows));
            if (uniq > static_cast<double>(cap)) break;
            best = rows;
        }
        return best;
    }
};

Scheduler::Scheduler() : impl_(std::make_unique<Impl>()) {}

Scheduler::~Scheduler() {
    close();
}

Status Scheduler::open(const ModelState& model,
                       MemoryHierarchy& memory,
                       KvCheckpointStore& checkpoints,
                       const SchedulerConfig& config) {
    (void)model;
    (void)checkpoints;
    (void)memory;
    (void)config;
    // The ModelState-shaped entry point belongs to the G5 engine wiring; the G3
    // scheduler is driven through open_f32() below, against the fp32 reference
    // model that every conformance gate is expressed in. Kept declared so the
    // header stays the contract.
    return {StatusCode::Unsupported,
            "Scheduler::open(ModelState) lands with the engine at G5; use open_f32()"};
}

Status Scheduler::open_f32(const F32Model& model,
                           MemoryHierarchy* memory,
                           const SchedulerConfig& config,
                           KvCheckpointStore* checkpoints) {
    close();
    auto& im = *impl_;
    im.model = &model;
    im.memory = memory;
    im.checkpoints = checkpoints;
    im.cfg = config;
    im.open = true;
    im.gate = im.compute_gate();
    im.stats = {};
    im.stats.effective_max_batch = im.gate;
    return {};
}

void Scheduler::close() {
    auto& im = *impl_;
    im.seqs.clear();
    im.ready.clear();
    im.waiting.clear();
    im.open = false;
}

void Scheduler::set_token_callback(TokenCallback cb) {
    impl_->on_token = std::move(cb);
}

void Scheduler::set_error_callback(ErrorCallback cb) {
    impl_->on_error = std::move(cb);
}

void Scheduler::set_finish_callback(FinishCallback cb) {
    impl_->on_finish = std::move(cb);
}

Status Scheduler::admit(SeqRequest request, SeqId& out_id, AdmitRejection& out_reason) {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    out_reason = AdmitRejection::None;

    if (!im.open) {
        out_reason = AdmitRejection::ShuttingDown;
        return {StatusCode::InvalidArgument, "scheduler is not open"};
    }
    if (request.prompt.empty()) {
        return {StatusCode::InvalidArgument, "empty prompt"};
    }
    const auto ctx_needed = request.prompt.size() + request.max_tokens;
    if (ctx_needed > im.cfg.ctx_size) {
        out_reason = AdmitRejection::ContextTooLong;
        return {StatusCode::InvalidArgument,
                "prompt + max_tokens (" + std::to_string(ctx_needed) + ") exceeds ctx_size " +
                    std::to_string(im.cfg.ctx_size)};
    }
    if (im.seqs.size() >= im.cfg.kv_slots) {
        // Refused, not queued-without-limit: a KV slot is real memory, and the
        // remedy (preempt something) differs from the remedy for the gate
        // (wait), which is why AdmitRejection distinguishes them.
        out_reason = AdmitRejection::NoKvSlot;
        return {StatusCode::CapacityPressure, "no free KV slot"};
    }

    auto s = std::make_unique<Seq>();
    s->id = im.next_id++;
    if (auto st = s->kv.open(im.model->arch, im.cfg.ctx_size); !st.ok()) return st;
    s->prompt = std::move(request.prompt);
    s->sampler = request.sampler;
    s->determinism = request.determinism;
    s->max_tokens = request.max_tokens;

    // ── warm reopen ──────────────────────────────────────────────────────────
    //
    // Attach a persisted cache as the prompt's prefix, but only after proving it
    // IS the prefix. Every failure here degrades to a cold start rather than an
    // error: a missing checkpoint, a stale architecture, or an edited earlier
    // turn all mean "prefill from scratch", and none of them should fail a
    // request the engine can serve correctly.
    if (!request.resume_key.empty() && im.checkpoints != nullptr) {
        std::vector<TokenId> cached;
        const auto st = im.checkpoints->load(request.resume_key, s->kv, cached);
        const bool is_prefix = st.ok() && cached.size() <= s->prompt.size() &&
                               std::equal(cached.begin(), cached.end(), s->prompt.begin());
        if (is_prefix && !cached.empty()) {
            s->history = std::move(cached);
            s->prompt_pos = static_cast<std::uint32_t>(s->history.size());
        } else {
            // Reopen the cache: load() may have written into it before failing,
            // and a partially populated cache with length 0 is worse than none.
            if (auto reopen = s->kv.open(im.model->arch, im.cfg.ctx_size); !reopen.ok()) {
                return reopen;
            }
        }
    }

    out_id = s->id;
    im.ready.push_back(s->id);
    im.seqs.emplace(s->id, std::move(s));
    im.stats.active_sequences = static_cast<std::uint32_t>(im.seqs.size());
    return {};
}

Status Scheduler::step() {
    auto& im = *impl_;

    // The step now runs UNDER the same lock admit/cancel/extend take.
    //
    // It read and wrote `ready`, `seqs` and `stats` unlocked, which was correct
    // exactly as long as one thread did everything — the serve path held a mutex
    // across the whole turn. A shared step loop with concurrent admits makes that
    // a data race, and the batch union's payoff is unreachable without one.
    //
    // A forward is milliseconds and an admit is a memcpy, so the contention this
    // introduces is not the interesting cost. The rule it creates IS interesting:
    // the token and finish callbacks fire from inside this lock, so a callback
    // that calls back into the scheduler deadlocks. Callers get told once, here.
    std::lock_guard<std::mutex> lk(im.mu);
    if (!im.open) return {StatusCode::InvalidArgument, "scheduler is not open"};

    im.gate = im.compute_gate();
    im.stats.effective_max_batch = im.gate;

    // ── collect ──────────────────────────────────────────────────────────────
    //
    // Round-robin over ready sequences, capped by the gate. Rotating rather than
    // always taking the head is what keeps a long generation from monopolising
    // the batch when more sequences are ready than the gate admits.
    std::vector<Seq*> batch;
    std::vector<TokenId> tokens;
    std::vector<KvRow> rows;

    const auto want = std::min<std::size_t>(im.gate, im.ready.size());
    batch.reserve(want);
    for (std::size_t i = 0; i < want; ++i) {
        const auto id = im.ready.front();
        im.ready.pop_front();
        auto it = im.seqs.find(id);
        if (it == im.seqs.end()) continue;
        Seq* s = it->second.get();
        if (s->finished || s->preempted) continue;
        batch.push_back(s);
    }
    if (batch.empty()) {
        im.stats.current_batch = 0;
        return {};
    }

    std::uint32_t prefill_rows = 0, decode_rows = 0;
    const auto hkv = im.model->arch.attention.n_kv_heads * im.model->arch.attention.head_dim;
    const auto stride = im.cfg.ctx_size * hkv;

    // ── chunked prefill, with two independent fairness limits ────────────────
    //
    // A prefilling sequence contributes MANY rows per step, not one. Consecutive
    // prompt tokens at consecutive positions batch into the same forward, which
    // is what turns a 512-token prompt from 512 steps into one.
    //
    // Two caps, because they prevent different failures:
    //
    //   max_prefill_rows_per_seq   one sequence cannot take the whole step, so a
    //                              32k prompt never starves an interactive turn
    //                              sharing the batch with it.
    //   prefill_chunk_tokens       a shared per-step budget, so N sequences each
    //                              under the per-seq cap still cannot multiply
    //                              into an unbounded forward. Row count drives
    //                              workspace size and attention cost; without
    //                              this, 8 sequences x 256 rows is one 2048-row
    //                              step and every decode row waits behind it.
    //
    // Decode rows are exempt from both. They are one row each and they are the
    // latency that chunking exists to protect.
    std::uint32_t prefill_budget = std::max(1u, im.cfg.prefill_chunk_tokens);
    const std::uint32_t per_seq_cap = std::max(1u, im.cfg.max_prefill_rows_per_seq);

    // Row ranges per sequence: prefill contributes [first, first+count).
    std::vector<std::pair<std::uint32_t, std::uint32_t>> span;
    span.reserve(batch.size());

    for (Seq* s : batch) {
        const auto first = static_cast<std::uint32_t>(tokens.size());
        std::uint32_t count = 0;

        if (s->prefilling()) {
            const auto left = static_cast<std::uint32_t>(s->prompt.size() - s->prompt_pos);
            count = std::min({left, per_seq_cap, prefill_budget});
            // Never zero: a sequence that reaches here with an exhausted budget
            // would be dropped from the batch entirely and re-queued forever.
            count = std::max(1u, count);
            prefill_budget -= std::min(prefill_budget, count);
            prefill_rows += count;
        } else {
            count = 1;
            ++decode_rows;
        }

        for (std::uint32_t j = 0; j < count; ++j) {
            tokens.push_back(s->prefilling() ? s->prompt[s->prompt_pos + j] : s->next_token);

            KvRow r{};
            r.k_base = s->kv.k_at(0, 0);
            r.v_base = s->kv.v_at(0, 0);
            r.stride = stride;
            r.hkv = hkv;
            r.pos = s->kv.length() + j;
            // Attends over everything before it in this sequence INCLUDING the
            // earlier rows of this same chunk. That is sound because
            // attention_kv writes every row's K/V before any row reads: rows
            // p..p+k of one sequence see each other exactly as they would have
            // across k separate steps.
            r.len = r.pos + 1;
            rows.push_back(r);
        }
        span.emplace_back(first, count);
    }

    // ── one batched forward ──────────────────────────────────────────────────
    if (auto st = forward_step_f32(*im.model, tokens, rows, im.ws, im.logits); !st.ok()) {
        if (im.on_error) {
            for (Seq* s : batch)
                im.on_error(s->id, st.code(), st.message().c_str());
        }
        // Rows go back on the queue: the failure is a property of the step
        // (capacity pressure, usually), not of the sequences in it.
        for (Seq* s : batch)
            im.ready.push_back(s->id);
        return st;
    }

    const auto vocab = im.model->vocab();

    // ── scatter ──────────────────────────────────────────────────────────────
    for (std::size_t i = 0; i < batch.size(); ++i) {
        Seq* s = batch[i];
        const auto [first, count] = span[i];
        s->kv.commit(count);
        // Alongside the commit, never before it: a forward that fails re-queues
        // its rows without committing, and a history recorded at row-build time
        // would then claim positions the cache does not hold.
        s->history.insert(s->history.end(), tokens.begin() + first, tokens.begin() + first + count);

        // The LAST row of this sequence's span carries its prediction. Interior
        // prefill rows exist to populate the cache; sampling from one would emit
        // a token for a prompt position the user already supplied.
        const float* lg = im.logits.data() + static_cast<std::size_t>(first + count - 1) * vocab;

        // Sampled with the SEQUENCE's own sampler and its own RNG state, so the
        // draw does not depend on how many other sequences shared this step.
        const auto emit = [&]() {
            s->next_token =
                sample_token(std::span<const float>(lg, vocab), s->sampler, s->emitted, s->scratch);
            s->emitted.push_back(s->next_token);
            ++s->generated;
            ++im.stats.tokens_out;
        };

        bool produced = false;
        if (s->prefilling()) {
            s->prompt_pos += count;
            if (!s->prefilling()) {
                emit();
                s->have_next = true;
                produced = true;
            }
        } else {
            emit();
            produced = true;
        }

        // Finished sequences are RETAINED, not erased. The KV, the emitted
        // history and the sampler RNG are the session; erasing here would make
        // every follow-up turn re-prefill the whole conversation, which is the
        // cost warm reopen exists to remove. The slot is real memory, so the
        // caller is responsible for cancel()ing sessions it no longer wants —
        // admit() reports NoKvSlot rather than silently evicting someone.
        const bool done = (!s->prefilling() && s->generated >= s->max_tokens) ||
                          s->kv.length() >= s->kv.capacity();

        // `last` is computed BEFORE the token goes out, so a listener that keys
        // completion off it is not told about a token and then, separately, that
        // the turn is over.
        if (produced && im.on_token) im.on_token(s->id, s->next_token, done);

        if (done) {
            s->finished = true;
            // The unambiguous signal. A sequence can end without producing a
            // token — a prompt that fills the context during prefill does — so a
            // caller waiting on `last` from on_token would wait forever.
            if (im.on_finish) im.on_finish(s->id);
            continue;
        }
        im.ready.push_back(s->id);
    }

    ++im.stats.steps;
    im.stats.current_batch = static_cast<std::uint32_t>(batch.size());
    im.stats.prefill_rows_last_step = prefill_rows;
    im.stats.decode_rows_last_step = decode_rows;
    im.stats.active_sequences = static_cast<std::uint32_t>(im.seqs.size());
    im.stats.queued_sequences = static_cast<std::uint32_t>(im.waiting.size());
    im.stats.unique_experts_last_step = static_cast<std::uint32_t>(im.ws.unique_expert_reads);
    im.stats.naive_expert_reads_last_step = static_cast<std::uint32_t>(im.ws.naive_expert_reads);
    return {};
}

Status Scheduler::extend(SeqId id, std::vector<TokenId> prompt, std::uint32_t max_tokens) {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    if (!im.open) return {StatusCode::InvalidArgument, "scheduler is not open"};

    auto it = im.seqs.find(id);
    if (it == im.seqs.end()) return {StatusCode::NotFound, "no such sequence"};
    Seq* s = it->second.get();
    if (s->preempted) return {StatusCode::InvalidArgument, "sequence is preempted; resume first"};
    if (!s->finished) return {StatusCode::InvalidArgument, "sequence is still generating"};

    // The check that makes a warm cache safe. A client that edits an earlier turn
    // sends a prompt this cache does not describe, and attaching it anyway
    // produces fluent output about a conversation that no longer exists.
    if (prompt.size() < s->history.size() ||
        !std::equal(s->history.begin(), s->history.end(), prompt.begin())) {
        return {StatusCode::ArchMismatch,
                "the cached " + std::to_string(s->history.size()) +
                    " tokens are not a prefix of this prompt; cancel and admit fresh"};
    }
    if (prompt.size() == s->history.size()) {
        // Nothing new to prefill. The last emitted token was sampled but never
        // fed back, so an empty extension has no row to contribute and would
        // spin the step loop forever.
        return {StatusCode::InvalidArgument, "prompt adds no tokens beyond the cached prefix"};
    }
    const auto ctx_needed = prompt.size() + max_tokens;
    if (ctx_needed > im.cfg.ctx_size) {
        return {StatusCode::InvalidArgument,
                "prompt + max_tokens (" + std::to_string(ctx_needed) + ") exceeds ctx_size " +
                    std::to_string(im.cfg.ctx_size)};
    }

    s->prompt = std::move(prompt);
    s->prompt_pos = static_cast<std::uint32_t>(s->history.size());
    s->max_tokens = max_tokens;
    s->generated = 0;
    s->have_next = false;
    s->finished = false;
    // `emitted` and `sampler` deliberately survive: the repetition penalties and
    // the RNG stream are properties of the conversation, not of one turn.
    im.ready.push_back(id);
    im.stats.active_sequences = static_cast<std::uint32_t>(im.seqs.size());
    return {};
}

Status Scheduler::checkpoint(SeqId id, const std::string& key) {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    if (im.checkpoints == nullptr) {
        return {StatusCode::Unsupported, "no checkpoint store; scheduler opened without one"};
    }
    auto it = im.seqs.find(id);
    if (it == im.seqs.end()) return {StatusCode::NotFound, "no such sequence"};
    Seq* s = it->second.get();
    if (s->preempted) return {StatusCode::InvalidArgument, "sequence is preempted; nothing live"};
    return im.checkpoints->save(key, s->kv, s->history);
}

Status Scheduler::sequence_tokens(SeqId id, std::vector<TokenId>& out) const {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    auto it = im.seqs.find(id);
    if (it == im.seqs.end()) return {StatusCode::NotFound, "no such sequence"};
    out = it->second->history;
    return {};
}

std::vector<SeqId> Scheduler::sequence_ids() const {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    std::vector<SeqId> out;
    out.reserve(im.seqs.size());
    for (const auto& [id, _] : im.seqs)
        out.push_back(id);
    return out;
}

Status Scheduler::preempt(SeqId id) {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    if (im.checkpoints == nullptr) {
        return {StatusCode::Unsupported, "no checkpoint store; scheduler opened without one"};
    }
    auto it = im.seqs.find(id);
    if (it == im.seqs.end()) return {StatusCode::NotFound, "no such sequence"};
    Seq* s = it->second.get();

    if (auto st = im.checkpoints->save(checkpoint_key(id), s->kv, s->history); !st.ok()) return st;

    // The KV buffer is released here — that is the entire point. Preemption that
    // keeps the memory it was called to reclaim is a no-op with extra steps.
    s->kv = KvCache{};
    s->preempted = true;
    im.ready.erase(std::remove(im.ready.begin(), im.ready.end(), id), im.ready.end());
    im.waiting.push_back(id);
    ++im.stats.preemptions;
    im.stats.queued_sequences = static_cast<std::uint32_t>(im.waiting.size());
    return {};
}

Status Scheduler::resume(SeqId id) {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    if (im.checkpoints == nullptr) {
        return {StatusCode::Unsupported, "no checkpoint store; scheduler opened without one"};
    }
    auto it = im.seqs.find(id);
    if (it == im.seqs.end()) return {StatusCode::NotFound, "no such sequence"};
    Seq* s = it->second.get();
    if (!s->preempted) return {StatusCode::InvalidArgument, "sequence is not preempted"};

    if (auto st = s->kv.open(im.model->arch, im.cfg.ctx_size); !st.ok()) return st;
    if (auto st = im.checkpoints->load(checkpoint_key(id), s->kv, s->history); !st.ok()) return st;

    s->preempted = false;
    im.waiting.erase(std::remove(im.waiting.begin(), im.waiting.end(), id), im.waiting.end());
    im.ready.push_back(id);
    im.stats.queued_sequences = static_cast<std::uint32_t>(im.waiting.size());
    // The checkpoint has served its purpose. Leaving it behind would make a
    // later sweep the only thing standing between preemption churn and an
    // unbounded directory.
    (void)im.checkpoints->remove(checkpoint_key(id));
    return {};
}

Status Scheduler::cancel(SeqId id) {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    if (im.seqs.erase(id) == 0) return {StatusCode::NotFound, "no such sequence"};
    im.ready.erase(std::remove(im.ready.begin(), im.ready.end(), id), im.ready.end());
    im.stats.active_sequences = static_cast<std::uint32_t>(im.seqs.size());
    return {};
}

SchedulerStats Scheduler::stats() const noexcept {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->stats;
}

std::uint32_t Scheduler::effective_max_batch() const noexcept {
    return impl_->gate;
}

bool Scheduler::idle() const {
    // Locked, and no longer noexcept. `ready` is mutated by step() on the driver
    // thread and by admit() on request threads; reading it unlocked was safe only
    // while one thread did everything.
    //
    // Preempted sequences are still live work — they are waiting for a resume,
    // not finished. Reporting idle here would let a driver loop exit while a
    // sequence sits half-generated on disk.
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->ready.empty();
}

} // namespace soma
