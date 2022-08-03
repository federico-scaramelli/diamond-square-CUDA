#pragma once
#include <chrono>
#include <iostream>

struct MeasureTime {
    char* msg;
    MeasureTime(char* msg) : _start(std::chrono::high_resolution_clock::now()) {
	    this->msg = msg;
    }

    ~MeasureTime() {
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = stop - _start;
        std::cout << msg << duration.count() * 1000 << "ms\n";
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock>  _start;
};

template <class T, class F, class... Args>
auto MeasureTimeFn(char* msg, T *t, F &&fn, Args&&... args) {
    MeasureTime timer(msg);
	return (t->*fn)(std::forward<Args>(args)...);
}

template <class F, class... Args>
auto MeasureTimeFn(char* msg, F &&fn, Args&&... args) {
	MeasureTime timer(msg);
	return (*fn)(std::forward<Args>(args)...);
}