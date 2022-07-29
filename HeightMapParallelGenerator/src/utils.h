#pragma once


static int mapValue(double min1, double max1, int min2, int max2, double value)
{
	return static_cast<int>((value - min1) / (max1 - min1) * (max2 - min2) + min2);
}