#include "diamondSquareBase.h"


#pragma region Constructors

DiamondSquareBase::DiamondSquareBase(const uint32_t size) {
	this->size = size;
	CheckSizeAdequate();

	this->step = size - 1;
	this->map = new float[size * size];
	memset(map, 0.0, sizeof(float) * size * size);
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
			std::cout << grayScaleMap[i * size + j] << ' ';
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

	step = initValuesDistance;
}

#pragma endregion

#pragma region Image Functions

void DiamondSquareBase::GenerateGrayScaleMap() {
	grayScaleMap = new uint8_t[size * size]{};
	float min = *std::min_element(map, map + size * size);
	float max = *std::max_element(map, map + size * size);

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			grayScaleMap[i * size + j] = static_cast<uint8_t>(MapValue(min, max, 0, 255, map[i * size + j]));
		}
	}

	DeleteDoubleMap();
}

void DiamondSquareBase::SaveGrayScaleImage(const char* fname, int tileSize) {
	if (grayScaleMap == nullptr) {
		std::cout << "Creating grayscale matrix..." << std::endl;
		GenerateGrayScaleMap();
		//PrintGrayScaleMap();
	}

	std::cout << "Creating grayscale image..." << std::endl;

	BMP image(size * tileSize, size * tileSize, true);
	//ColorPixel color{};

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {

			//color.SetColor(grayScaleMap[i * size + j]);

			image.FillRegion(j * tileSize, i * tileSize, tileSize, grayScaleMap[i * size + j]);
		}
	}
	image.Write(fname);
}

void DiamondSquareBase::SaveColorImage(const char* fname, int tileSize) {
	if (grayScaleMap == nullptr) {
		std::cout << "Creating grayscale matrix..." << std::endl;
		GenerateGrayScaleMap();
	}

	ColorMapping::CacheColorsFromMapping();

	BMP image(size * tileSize, size * tileSize, true);
	ColorPixel color;
	std::cout << "Creating color image..." << std::endl;


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
	InitializeDiamondSquare();

	std::cout << "Executing diamond square..." << std::endl;

	DiamondSquare();
}