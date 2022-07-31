#pragma once


static int mapValue(double fromMin, double fromMax, int toMin, int toMax, double value)
{
	return static_cast<int>((value - fromMin) / (fromMax - fromMin) * (toMax - toMin) + toMin);
}