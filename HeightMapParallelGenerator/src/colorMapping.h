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
	static const ColorRangeMap mappings[10];

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
	{"#064273", "#50C1E5", 0, 70},		//Sea
	{"#BAB280", "#FAE7AC", 70, 78},		//Sand
	{"#32A74F", "#7BA732", 78, 85},		//Grass 
	{"#7BA732", "#BCBF56", 85, 115},		//Grass 
	{"#BCBF56", "#5C8553", 115, 150},	//Hill
	{"#5C8553", "#80857F", 150, 155},	//Mountain start
	{"#80857F", "#AEAEAE", 155, 210},	//Mountain
	{"#c3dcdc", "#ffffff", 210, 255},	//Snow


	/*{"#55871E", "#697764", 155, 160},	//Mountain start
	{"#697764", "#8A8B87", 160, 175},	//Mountain low
	{"#8A8B87", "#6D582F", 175, 195},	//Mountain medium
	{"#6D582F", "#7B7C77", 195, 210},	//Mountain high*/

	/*{"#519843", "#948B7A", 150, 155},	//Mountain start
	{"#948B7A", "#6C6049", 155, 165},	//Mountain low
	{"#6C6049", "#8A8B87", 165, 200},	//Mountain medium
	{"#8A8B87", "#A6A6A3", 200, 210},	//Mountain high*/
};