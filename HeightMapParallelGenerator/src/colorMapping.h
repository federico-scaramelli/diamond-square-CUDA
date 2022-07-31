#pragma once
#include <vector>
#include "colorPixel.h"
#include "utils.h"

class ColorRangeMap {
public:
	const char* colorMin;
	const char* colorMax;
	uint8_t max;

	const ColorPixel* colorMinPixel;
	const ColorPixel* colorMaxPixel;
};

class ColorMapping {
private:
	static ColorRangeMap mappings[8];
	static uint8_t mappingsCount;
	static std::vector<ColorPixel> colors;

public:
	static void getColor(int value, ColorPixel* const outColor);

	static void generateColors();
};