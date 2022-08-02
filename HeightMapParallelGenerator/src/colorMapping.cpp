#include "colorMapping.h"


void ColorMapping::CacheColorsFromMapping() {
	std::cout << "Generating color mappings cache..." << std::endl;

	mappingsCount = sizeof(mappings) / sizeof(ColorRangeMap);
	colors.resize(sizeof(ColorPixel) * mappingsCount * 2);

	uint8_t count = 0;

	colors[count] = ColorPixel(mappings[0].colorMin);
	mappings[0].colorMinPixel = &colors[count++];
	colors[count] = ColorPixel(mappings[0].colorMax);
	mappings[0].colorMaxPixel = &colors[count++];

	for (uint8_t i = 1; i < mappingsCount; i++) {
		if (std::strcmp(mappings[i - 1].colorMax, mappings[i].colorMin) == 0) {
			mappings[i].colorMinPixel = mappings[i - 1].colorMaxPixel;
		}
		else {
			colors[count] = ColorPixel(mappings[i].colorMin);
			mappings[i].colorMinPixel = &colors[count++];
		}
		colors[count] = ColorPixel(mappings[i].colorMax);
		mappings[i].colorMaxPixel = &colors[count++];
	}

	colors.resize(sizeof(ColorPixel) * count);
}

void ColorMapping::GetColorLerp(const int value, ColorPixel* const outColor) {
	if (outColor == nullptr) {
		throw std::runtime_error("Output color pointer is nullptr.");
	}
	if (value < 0 || value > 255) {
		throw std::runtime_error("Value out of range.");
	}

	if (colors.empty()) {
		CacheColorsFromMapping();
	}

	mappingsCount = sizeof(mappings) / sizeof(ColorRangeMap);

	uint8_t minVal = 0;

	for (uint8_t i = 0; i < mappingsCount; i++) {
		if (value < mappings[i].max) {
			minVal = i > 0 ? mappings[i - 1].max : 0;
			auto B = static_cast<uint8_t>(MapValue(minVal, mappings[i].max,
			                                       mappings[i].colorMinPixel->GetB(), mappings[i].colorMaxPixel->GetB(),
			                                       value));
			auto G = static_cast<uint8_t>(MapValue(minVal, mappings[i].max,
			                                       mappings[i].colorMinPixel->GetG(), mappings[i].colorMaxPixel->GetG(),
			                                       value));
			auto R = static_cast<uint8_t>(MapValue(minVal, mappings[i].max,
			                                       mappings[i].colorMinPixel->GetR(), mappings[i].colorMaxPixel->GetR(),
			                                       value));

			outColor->SetColor(B, G, R);
			return;
		}
	}
}


ColorRangeMap ColorMapping::mappings[8]{
	{"#064273", "#50C1E5", 70}, //Sea
	{"#BAB280", "#FAE7AC", 78}, //Sand
	{"#32A74F", "#7BA732", 85}, //Green Grass 
	{"#7BA732", "#BCBF56", 115}, //Yellow Grass 
	{"#BCBF56", "#5C8553", 150}, //Hill
	{"#5C8553", "#80857F", 155}, //Mountain start
	{"#80857F", "#AEAEAE", 210}, //Mountain
	{"#c3dcdc", "#ffffff", 255} //Snow


	/*{"#55871E", "#697764", 155, 160},	//Mountain start
	{"#697764", "#8A8B87", 160, 175},	//Mountain low
	{"#8A8B87", "#6D582F", 175, 195},	//Mountain medium
	{"#6D582F", "#7B7C77", 195, 210},	//Mountain high

	{"#519843", "#948B7A", 150, 155},	//Mountain start
	{"#948B7A", "#6C6049", 155, 165},	//Mountain low
	{"#6C6049", "#8A8B87", 165, 200},	//Mountain medium
	{"#8A8B87", "#A6A6A3", 200, 210},	//Mountain high*/
};

//Static members definition
uint8_t ColorMapping::mappingsCount = 0;
std::vector<ColorPixel> ColorMapping::colors;