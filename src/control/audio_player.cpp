#include "control/audio_player.hpp"

#include <algorithm>

#if __has_include(<miniaudio.h>)
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>
#define MM_HAS_MINIAUDIO 1
#else
#define MM_HAS_MINIAUDIO 0
#endif

namespace mm {

struct AudioPlayer::Impl {
    mutable std::mutex mutex;
    AudioPlayerStatus status;
#if MM_HAS_MINIAUDIO
    ma_engine engine{};
    ma_sound sound{};
    bool engine_initialized = false;
    bool sound_initialized = false;
#endif
};

AudioPlayer::AudioPlayer() : impl_(std::make_unique<Impl>()) {
#if MM_HAS_MINIAUDIO
    if (ma_engine_init(nullptr, &impl_->engine) == MA_SUCCESS) {
        impl_->engine_initialized = true;
        impl_->status.available = true;
    } else {
        impl_->status.error = "audio device initialization failed";
    }
#else
    impl_->status.error = "built without miniaudio";
#endif
}

AudioPlayer::~AudioPlayer() {
#if MM_HAS_MINIAUDIO
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->sound_initialized) ma_sound_uninit(&impl_->sound);
    if (impl_->engine_initialized) ma_engine_uninit(&impl_->engine);
#endif
}

bool AudioPlayer::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if MM_HAS_MINIAUDIO
    if (!impl_->engine_initialized) return false;
    if (impl_->sound_initialized) {
        ma_sound_uninit(&impl_->sound);
        impl_->sound_initialized = false;
    }
    const ma_result result = ma_sound_init_from_file(
        &impl_->engine, path.c_str(), MA_SOUND_FLAG_STREAM, nullptr, nullptr, &impl_->sound);
    if (result != MA_SUCCESS) {
        impl_->status.loaded = false;
        impl_->status.error = "failed to load audio: " + path;
        return false;
    }
    impl_->sound_initialized = true;
    impl_->status.loaded = true;
    impl_->status.playing = false;
    impl_->status.path = path;
    impl_->status.error.clear();
    impl_->status.position_seconds = 0.0f;
    float duration = 0.0f;
    if (ma_sound_get_length_in_seconds(&impl_->sound, &duration) == MA_SUCCESS)
        impl_->status.duration_seconds = duration;
    ma_sound_set_volume(&impl_->sound, impl_->status.volume);
    return true;
#else
    (void)path;
    return false;
#endif
}

bool AudioPlayer::play() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if MM_HAS_MINIAUDIO
    if (!impl_->sound_initialized) return false;
    if (ma_sound_start(&impl_->sound) != MA_SUCCESS) {
        impl_->status.error = "audio playback failed";
        return false;
    }
    impl_->status.playing = true;
    return true;
#else
    return false;
#endif
}

void AudioPlayer::pause() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if MM_HAS_MINIAUDIO
    if (impl_->sound_initialized) ma_sound_stop(&impl_->sound);
#endif
    impl_->status.playing = false;
}

void AudioPlayer::stop() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if MM_HAS_MINIAUDIO
    if (impl_->sound_initialized) {
        ma_sound_stop(&impl_->sound);
        ma_sound_seek_to_pcm_frame(&impl_->sound, 0);
    }
#endif
    impl_->status.playing = false;
    impl_->status.position_seconds = 0.0f;
}

void AudioPlayer::seek_relative(float seconds) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if MM_HAS_MINIAUDIO
    if (!impl_->sound_initialized) return;
    float cursor = 0.0f;
    if (ma_sound_get_cursor_in_seconds(&impl_->sound, &cursor) != MA_SUCCESS) return;
    const float target = std::clamp(cursor + seconds, 0.0f, impl_->status.duration_seconds);
    ma_uint64 frame = 0;
    const ma_uint32 sample_rate = ma_engine_get_sample_rate(&impl_->engine);
    frame = static_cast<ma_uint64>(target * static_cast<float>(sample_rate));
    ma_sound_seek_to_pcm_frame(&impl_->sound, frame);
#else
    (void)seconds;
#endif
}

void AudioPlayer::set_volume(float volume) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->status.volume = std::clamp(volume, 0.0f, 1.0f);
#if MM_HAS_MINIAUDIO
    if (impl_->sound_initialized) ma_sound_set_volume(&impl_->sound, impl_->status.volume);
#endif
}

AudioPlayerStatus AudioPlayer::status() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    AudioPlayerStatus out = impl_->status;
#if MM_HAS_MINIAUDIO
    if (impl_->sound_initialized) {
        float cursor = 0.0f;
        ma_sound_get_cursor_in_seconds(&impl_->sound, &cursor);
        out.position_seconds = cursor;
        out.playing = ma_sound_is_playing(&impl_->sound) != MA_FALSE;
    }
#endif
    return out;
}

}  // namespace mm
