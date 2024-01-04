#pragma once
#include <chrono>

class Timer
{
public:
    void start() { mStart = mStop = std::chrono::high_resolution_clock::now(); }
    void stop() { mStop = std::chrono::high_resolution_clock::now(); }

    uint64_t microseconds() const
    {
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(mStop - mStart);
        return static_cast<uint64_t>(duration.count());
    }

private:
    std::chrono::high_resolution_clock::time_point mStart, mStop;
};