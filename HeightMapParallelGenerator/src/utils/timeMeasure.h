#pragma once
#include <chrono>
#include <iostream>

struct MeasureTime {
    const char* msg;
    double* outputTime;

    MeasureTime(const char* msg, double* outputTime) : _start(std::chrono::high_resolution_clock::now()) {
	    this->msg = msg;
        this->outputTime = outputTime;
    }

    ~MeasureTime()
    {
	    auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = stop - _start;
        std::cout << msg << duration.count() * 1000 << "ms\n";
        if(outputTime != nullptr) {
	        *outputTime = duration.count() * 1000;
        }
    }
    
    std::chrono::time_point<std::chrono::high_resolution_clock>  _start;
};

template <class T, class F, class... Args>
auto MeasureTimeFn(double* outputTime, const char* msg, T *t, F &&fn, Args&&... args)
{
    MeasureTime timer(msg, outputTime);

	return (t->*fn)(std::forward<Args>(args)...);
}

template <class F, class... Args>
auto MeasureTimeFn(double* outputTime, const char* msg, F &&fn, Args&&... args)
{
	MeasureTime timer(msg, outputTime);

    return (*fn)(std::forward<Args>(args)...);
}

static void CompareTime(const char* message, const double* time1, const double* time2)
{
	std::cout << message << *time1 / *time2 << std::endl;
}

static void CompareTime(const char* message, const float* time1, const float* time2)
{
	std::cout << message << *time1 / *time2 << std::endl;
}