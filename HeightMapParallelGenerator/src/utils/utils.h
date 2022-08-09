#pragma once
#include <random>

static int MapValue(float fromMin, float fromMax, int toMin, int toMax, float value)
{
	return static_cast<int>((value - fromMin) / (fromMax - fromMin) * (toMax - toMin) + toMin);
}

static std::random_device rd;
static std::mt19937 cpuGenerator(rd());

static float RandomFloatUniform()
{
	static std::uniform_real_distribution<float> unif{-1.0, 1.0};
	return unif(cpuGenerator);
}

static int RandomIntUniform()
{
	static std::uniform_int_distribution<int> unif{-21474836, 21474836};
	return unif(cpuGenerator);
}