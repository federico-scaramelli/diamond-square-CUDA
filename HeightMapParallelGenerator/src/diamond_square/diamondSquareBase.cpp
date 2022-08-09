#include "diamondSquareBase.h"


#pragma region Constructors

DiamondSquareBase::DiamondSquareBase(const uint32_t size) {
	this->size = size;
	CheckSizeAdequate();
	this->totalSize = size * size;
	this->step = size - 1;
	this->map = new float[totalSize];
	memset(map, 0.0, sizeof(float) * totalSize);
	half = 0;
}

DiamondSquareBase::~DiamondSquareBase() {
	delete[] map;
	delete[] grayScaleMap;
}

void DiamondSquareBase::DeleteDoubleMap() {
	delete[] map;
	map = nullptr;
}

#pragma endregion

#pragma region Support Functions

void DiamondSquareBase::CheckSizeAdequate() {
	if ((size - 1 & size - 2) != 0) {
		throw std::exception("Size not adequate. The map size must be a power of two plus one.");
	}
}
uint32_t DiamondSquareBase::GetIndex(uint32_t x, uint32_t y) const {
	if (x >= size) x = size - 1;
	if (y >= size) y = size - 1;

	return x * size + y;
}

void DiamondSquareBase::PrintMap() const {
	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			std::cout << map[i * size + j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl << std::endl;
}
void DiamondSquareBase::PrintGrayScaleMap() {
	if (grayScaleMap == nullptr) GenerateGrayScaleMap();

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			std::cout << (int)grayScaleMap[i * size + j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl << std::endl;
}

#pragma endregion

#pragma region Setter Functions

void DiamondSquareBase::SetRandomScale(float randomScale) {
	this->randomScale = randomScale;
}

void DiamondSquareBase::SetInitialStepSize(uint32_t initValuesDistance) {
	if (initValuesDistance == 0)
		throw std::runtime_error("Init values distance is 0!");
	if (initValuesDistance > size - 1)
		throw std::runtime_error("Init values distance is too big!");
	if ((size - 1) % initValuesDistance != 0)
		throw std::runtime_error("Init values distance is not a multiple of the size!");

	step = initValuesDistance;
}

#pragma endregion

#pragma region Image Functions

void DiamondSquareBase::GenerateGrayScaleMap() {
  	std::cout << "\n----------MAP GENERATION----------" << std::endl;
	std::cout << "Creating grayscale matrix..." << std::endl;

	grayScaleMap = new uint8_t[totalSize]{0};
	float min = *std::min_element(map, map + totalSize);
	float max = *std::max_element(map, map + totalSize);

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			grayScaleMap[i * size + j] = static_cast<uint8_t>(MapValue(min, max, 0, 255, map[i * size + j]));
		}
	}

	DeleteDoubleMap();
}

void DiamondSquareBase::SaveGrayScaleImage(const char* fname, int tileSize) {
	if (grayScaleMap == nullptr) {
		MeasureTimeFn ("Grayscale map generated in ", this, &DiamondSquareBase::GenerateGrayScaleMap);
	}

	std::cout << "Creating grayscale image..." << std::endl;

	BMP image(size * tileSize, size * tileSize, true);

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			image.FillRegion(j * tileSize, i * tileSize, tileSize, grayScaleMap[i * size + j]);
		}
	}
	image.Write(fname);
}

void DiamondSquareBase::SaveColorImage(const char* fname, int tileSize) {
	if (grayScaleMap == nullptr) {
		MeasureTimeFn ("Grayscale map generated in ", this, &DiamondSquareBase::GenerateGrayScaleMap);
	}

	ColorMapping::CacheColorsFromMapping();

	std::cout << "Creating color image..." << std::endl;

	BMP image(size * tileSize, size * tileSize, true);
	ColorPixel color;

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			ColorMapping::GetColorLerp(grayScaleMap[i * size + j], &color);
			image.FillRegion(j * tileSize, i * tileSize, tileSize, color);
		}
	}
	image.Write(fname);
}

#pragma endregion

void DiamondSquareBase::ExecuteDiamondSquare() {

	MeasureTimeFn("\nMap initialized in ", this, &DiamondSquareBase::InitializeDiamondSquare);

  	std::cout << "\n----------EXECUTION----------" << std::endl;
	std::cout << "Executing diamond square..." << std::endl;

	MeasureTimeFn("Algorithm terminated in ", this, &DiamondSquareBase::DiamondSquare);
}