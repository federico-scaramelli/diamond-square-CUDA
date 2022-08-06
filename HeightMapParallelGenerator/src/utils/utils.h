#pragma once
#include <random>

static int MapValue(float fromMin, float fromMax, int toMin, int toMax, float value)
{
	return static_cast<int>((value - fromMin) / (fromMax - fromMin) * (toMax - toMin) + toMin);
}

static float random_float_uniform()
{
	std::random_device rd;
	std::mt19937 cpuGenerator(rd());
	std::uniform_real_distribution<float> unif{-1.0, 1.0};
	return unif(cpuGenerator);
}

static int random_int_uniform()
{
	std::random_device rd;
	std::mt19937 cpuGenerator(rd());
	std::uniform_int_distribution<int> unif{-21474836, 21474836};
	return unif(cpuGenerator);
}