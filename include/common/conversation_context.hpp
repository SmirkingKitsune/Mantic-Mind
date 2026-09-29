#pragma once

// Mantic-Mind — what one turn sends, and keeping it inside the model's window.
//
// ── The context ───────────────────────────────────────────────────────────────
//
// A turn sends ONE system message, then every message past the conversation's
// summary watermark. The system message is built in a fixed order:
//
//   1. the user's custom instructions
//   2. the conversation's own system prompt
//   3. the user's global memories            (when memories are on)
//   4. this conversation's notes             (when memories are on)
//   5. the rolling summary of everything before the watermark
//
// ── Rolling compaction ────────────────────────────────────────────────────────
//
// History grows without bound, and the model answering a conversation can
// change at any turn to one with a smaller window. So the older part of the
// history is folded, IN PLACE, into the conversation's rolling summary:
//
//   when   the prompt reaches `trigger` of the answering model's budget, or the
//          images past the watermark near the per-context limit
//   keeps  the longest recent tail that starts at a user turn and fits in
//          `keep_tail` of the budget — never less than the last user turn
//   folds  everything before that, chunk by chunk, each summarizer call sized
//          to fit the model doing the folding; the watermark is saved after
//          every chunk, so a failure part-way resumes where it stopped
//   fails  with `overflow` when even the folded prompt does not fit
//
// The conversation keeps its id, its local memories and its stored history;
// only what is SENT shrinks, and the summary can always be rebuilt from what is
// stored.
//
// This replaces compaction by forking, which sent every old message to the
// summarizer in one prompt — overflowing as soon as a conversation moved to a
// model with a smaller window — and then seeded a NEW conversation with the
// result.

#include "common/engine_capabilities.hpp"
#include "common/history_sanitizer.hpp"
#include "common/models.hpp"
#include "common/user.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace mm {

/// Images a single request may carry, and their decoded bytes. The limits
/// hydration enforces; compaction keeps the tail under them.
inline constexpr int kMaxContextImages = 8;
inline constexpr std::int64_t kMaxContextImageBytes = 400LL * 1024 * 1024;

/// What one image costs a vision model's context, as an estimate.
inline constexpr int kImageTokenEstimate = 2048;

/// Tokens a prompt may use: the window less the completion's reservation,
/// ctx_size − min(max_tokens, ctx_size − 1). At least 1.
int prompt_budget(const RuntimeSettings& settings);

/// A quarter of the bytes, rounded up.
int estimate_tokens(const std::string& text);

/// The larger of the message's recorded token count and estimate_tokens of its
/// text, plus kImageTokenEstimate per image when the model can see images —
/// without vision an image is sent as a placeholder, which is text.
int estimate_tokens(const Message& message, bool vision);

/// Image parts in `message`.
int image_count(const Message& message);

/// The system message's text (see the header comment for the order). The
/// memory sections appear only when the user has memories on; `tools_offered`
/// decides whether they mention the memory tools.
std::string build_system_text(const UserProfile& user,
                              const Conversation& conversation,
                              const std::vector<Memory>& global_memories,
                              const std::vector<LocalMemory>& local_memories,
                              bool tools_offered);

/// The messages a turn sends: `system_text` (when not empty), then `tail`
/// rewritten for `caps` (sanitize_history).
std::vector<Message> build_turn_context(const std::string& system_text,
                                        std::vector<Message> tail,
                                        const ModelCapabilities& caps,
                                        const AttachmentNamer& name_of = {});

// ── Rolling compaction ────────────────────────────────────────────────────────

struct CompactionPolicy {
    double trigger = 0.8;   ///< of the budget
    double keep_tail = 0.4; ///< of the budget
    int max_images = kMaxContextImages;
    std::int64_t max_image_bytes = kMaxContextImageBytes;
    /// Slack left in every summarizer call for its chat template and the error
    /// in a quarter-of-the-bytes estimate.
    int margin = 256;
};

/// One summarizer call. `prompt` is a system and a user message; the result is
/// the whole updated summary, or nullopt when the call failed. The caller runs
/// it on the model answering the turn, at low temperature, without tools.
using Summarize =
    std::function<std::optional<std::string>(const std::vector<Message>& prompt, int max_tokens)>;

/// Persist progress: `summary` covers everything up to and including
/// `through_seq`. False stops compaction (the store refused).
using SaveSummary = std::function<bool(const std::string& summary, int through_seq)>;

/// An attachment's size in bytes, for the image limits; 0 when unknown.
using AttachmentBytes = std::function<std::int64_t(const std::string& attachment_id)>;

struct CompactionRequest {
    std::string summary;
    int summary_through_seq = -1;
    /// Every stored message past the watermark, in order, sequence numbers set.
    std::vector<Message> messages;
    /// prompt_budget() of the model answering the turn.
    int budget = 0;
    /// estimate_tokens() of the system message WITHOUT its summary section.
    int base_system_tokens = 0;
    /// The answering model's; decides whether images cost tokens.
    bool vision = false;
    /// Fold everything but the latest user turn, regardless of the trigger
    /// (manual compaction).
    bool force = false;
};

struct CompactionResult {
    enum class Status {
        NotNeeded, ///< under the trigger; nothing changed
        Compacted, ///< folded, and the prompt now fits
        Failed,    ///< a summarizer call or a save failed; progress so far is saved
        Overflow,  ///< folded as far as possible and the prompt still does not fit
    };
    Status status = Status::NotNeeded;
    std::string summary;
    int summary_through_seq = -1;
    /// Estimated prompt after compaction: system, summary and tail.
    int prompt_tokens = 0;
    int summarizer_calls = 0;
    std::string error;
};

const char* to_string(CompactionResult::Status status);

/// The words a summary is introduced with in the system message. Exposed so
/// estimates and tests use the same text the context does.
std::string summary_section(const std::string& summary);

class RollingCompactor {
public:
    RollingCompactor(Summarize summarize,
                     SaveSummary save,
                     AttachmentBytes bytes_of = {},
                     CompactionPolicy policy = {});

    CompactionResult run(const CompactionRequest& request) const;

    /// The summary's size cap for a model with `budget`: a quarter of it,
    /// clamped to [256, 2048] tokens.
    static int summary_cap(int budget);

private:
    Summarize summarize_;
    SaveSummary save_;
    AttachmentBytes bytes_of_;
    CompactionPolicy policy_;
};

} // namespace mm
