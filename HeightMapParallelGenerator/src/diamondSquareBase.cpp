#include "diamondSquareBase.h"


DiamondSquareBase::DiamondSquareBase(const uint32_t size) {
	this->size = size;
	if (!CheckSizeAdequate())
		throw std::runtime_error("Size is not adequate!");

	this->map = new double[size * size];
	memset(map, 0.0, sizeof(double) * size * size);
	half = 0;
}

DiamondSquareBase::~DiamondSquareBase() {
	delete[] map;
}

void DiamondSquareBase::ExecuteDiamondSquare() {
	InitializeDiamondSquare(size - 1);
	DiamondSquare();
}

void DiamondSquareBase::ExecuteDiamondSquare(uint32_t initValuesDistance) {
	if (initValuesDistance == 0)
		throw std::runtime_error("Init values distance is 0!");
	if (initValuesDistance > size - 1)
		throw std::runtime_error("Init values distance is too big!");

	InitializeDiamondSquare(initValuesDistance);
	DiamondSquare();
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
	if (grayScaleMap == nullptr) CreateGrayScaleMap();

	for (uint32_t i = 0; i < size; ++i) {
        for (uint32_t j = 0; j < size; ++j) {
            std::cout << grayScaleMap[i * size + j] << ' ';
        }
        std::cout << std::endl;
    }
	std::cout << std::endl << std::endl;
}

void DiamondSquareBase::CreateGrayScaleMap() {
	grayScaleMap = new uint8_t[size * size] {};
	double min = *std::min_element(map, map + size * size);
	double max = *std::max_element(map, map + size * size);

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			grayScaleMap[i * size + j] = static_cast<uint8_t>(mapValue(min, max, 0, 255, map[i * size + j]));
		}
	}
}

void DiamondSquareBase::SaveGrayScaleImage(const char* fname, int tileSize) {
	if (grayScaleMap == nullptr) {
		std::cout << "Creating grayscale matrix..." << std::endl;
		CreateGrayScaleMap();
		//PrintGrayScaleMap();
	}

	std::cout << "Creating grayscale image..." << std::endl;

	BMP image(size * tileSize, size * tileSize, true);
	ColorPixel color {};

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			
			color.B = color.G = color.R	= grayScaleMap[i * size + j];

			image.FillRegion(j * tileSize, i * tileSize, tileSize, color);
		}
	}
	image.Write(fname);
}

void DiamondSquareBase::SaveColorImage(const char* fname, int tileSize) {
	if (grayScaleMap == nullptr) {
		std::cout << "Creating grayscale matrix..." << std::endl;
		CreateGrayScaleMap();
	}

	ColorMapping::generateColors();

	BMP image(size * tileSize, size * tileSize, true);
	ColorPixel color;
	std::cout << "Creating color image..." << std::endl;


	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			try {
				ColorMapping::getColor(grayScaleMap[i * size + j], &color);
			}catch (std::exception& e) {
				std::cout << e.what() << std::endl;
			}
			image.FillRegion(j * tileSize, i * tileSize, tileSize, color);
		}
	}
	image.Write(fname);
}

bool DiamondSquareBase::CheckSizeAdequate() {
	if ((size - 1 & size - 2) != 0) {
		std::cout << "Size not adequate. The map size must be a power of two plus one." << std::endl;
		return false;
	}
	return true;
}