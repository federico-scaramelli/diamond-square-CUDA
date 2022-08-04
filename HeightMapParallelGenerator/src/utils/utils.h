#pragma once


static int MapValue(float fromMin, float fromMax, int toMin, int toMax, float value)
{
	return static_cast<int>((value - fromMin) / (fromMax - fromMin) * (toMax - toMin) + toMin);
}