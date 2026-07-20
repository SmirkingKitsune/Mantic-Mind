#pragma once

#include <memory>
#include <mutex>
#include <string>

namespace mm {

struct AudioPlayerStatus {
    bool available = false;
    bool loaded = false;
    bool playing = false;
    float position_seconds = 0.0f;
    float duration_seconds = 0.0f;
    float volume = 1.0f;
    std::string path;
    std::string error;
};

class AudioPlayer {
public:
    AudioPlayer();
    ~AudioPlayer();
    AudioPlayer(const AudioPlayer&) = delete;
    AudioPlayer& operator=(const AudioPlayer&) = delete;

    bool load(const std::string& path);
    bool play();
    void pause();
    void stop();
    void seek_relative(float seconds);
    void set_volume(float volume);
    AudioPlayerStatus status() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mm
