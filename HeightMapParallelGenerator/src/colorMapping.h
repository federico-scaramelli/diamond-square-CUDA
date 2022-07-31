#pragma once
#include "colorPixel.h"
#include "utils.h"

class ColorRangeMap {
public:
	const char* colorMin;
	const char* colorMax;
	const int min;
	const int max;
};

class ColorMapping {
private:
	static const ColorRangeMap mappings[8];

public:
	static void getColor(int value, ColorPixel* const outColor) {
		if (outColor == nullptr) {
			throw std::runtime_error("Output color pointer is nullptr.");
		}
		if (value < 0 || value > 255) {
			throw std::runtime_error("Value out of range.");
		}

		for (auto& mapping : mappings) {
			if (value >= mapping.min && value <= mapping.max) {
				ColorPixel colorTest("#ffffff");
				ColorPixel colorMin(mapping.colorMin);
				ColorPixel colorMax(mapping.colorMax);
				outColor->B = static_cast<uint32_t>(mapValue(mapping.min, mapping.max, colorMin.B, colorMax.B, value));
				outColor->G = static_cast<uint32_t>(mapValue(mapping.min, mapping.max, colorMin.G, colorMax.G, value));
				outColor->R = static_cast<uint32_t>(mapValue(mapping.min, mapping.max, colorMin.R, colorMax.R, value));
				return;
			}
		}
	}
};

const ColorRangeMap ColorMapping::mappings[] {
	{"#064273", "#1da2d8", 0, 50},		//Sea
	{"#eccca2", "#679267", 50, 55},		//Sand
	{"#679267", "#489030", 55, 145},		//Grass
	{"#489030", "#485123", 145, 165},	//Dirt
	{"#485123", "#5a5b62", 165, 220},	//Mountain
	{"#5a5b62", "#c1c5d6", 220, 225},	//Snow merge
	{"#c1c5d6", "#ebecf0", 225, 255}		//Snow
};