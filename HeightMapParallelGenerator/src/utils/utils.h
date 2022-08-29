#pragma once
#include <random>

// Map a value from a range to another range
static int MapValue (const float fromMin, const float fromMax, 
					 const int toMin, const int toMax, 
					 const float value)
{
	return static_cast<int>((value - fromMin) / (fromMax - fromMin) * (toMax - toMin) + toMin);
}

// Static instances for random number generation CPU side
static std::random_device rd;
static std::mt19937 cpuGenerator(rd());

// Returns a random float using a uniform distribution
static float RandomFloatUniform()
{
	static std::uniform_real_distribution<float> unif{-1.0, 1.0};
	return unif(cpuGenerator);
}

// Returns a random int using a uniform distribution
static int RandomIntUniform()
{
	static std::uniform_int_distribution<int> unif{-21474836, 21474836};
	return unif(cpuGenerator);
}