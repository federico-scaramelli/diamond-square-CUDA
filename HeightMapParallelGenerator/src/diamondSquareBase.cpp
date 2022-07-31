#include "diamondSquareBase.h"
#include "colorMapping.h"
#include <algorithm>

DiamondSquareBase::DiamondSquareBase(const uint32_t size) {
	this->size = size;
	if (!CheckSizeAdequate())
		throw std::runtime_error("Size is not adequate!");

	this->map = new double[size * size];
	memset(map, 0.0, sizeof(double) * size * size);
	//PrintMap();
	half = 0;
}

DiamondSquareBase::~DiamondSquareBase() {
	delete[] map;
}

void DiamondSquareBase::ExecuteDiamondSquare() {
	InitializeDiamondSquare(size - 1);
	//PrintMap();
	DiamondSquare();
	//PrintMap();
}

void DiamondSquareBase::ExecuteDiamondSquare(uint32_t initValuesDistance) {
	if (initValuesDistance == 0)
		throw std::runtime_error("Init values distance is 0!");

	InitializeDiamondSquare(initValuesDistance);
	//PrintMap();
	DiamondSquare();
	//PrintMap();
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
	grayScaleMap = new ColorPixel[size * size] {};
	double min = *std::min_element(map, map + size * size);
	double max = *std::max_element(map, map + size * size);

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			double channel = map[i * size + j];
            ColorPixel c{channel, channel, channel, min, max};
            //ColorPixel c{channel, channel, channel};
			grayScaleMap[i * size + j] = c;
		}
	}
}

void DiamondSquareBase::SaveGrayScaleImage(const char* fname, int tileSize) {
	if (grayScaleMap == nullptr) {
		std::cout << "Creating grayscale matrix..." << std::endl;
		CreateGrayScaleMap();
		//PrintGrayScaleMap();
	}

	BMP image(size * tileSize, size * tileSize, true);

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			image.FillRegion(j * tileSize, i * tileSize, tileSize, tileSize, grayScaleMap[i * size + j], 255);
		}
	}
	image.Write(fname);
}

void DiamondSquareBase::SaveColorImage(const char* fname, int tileSize) {
	if (grayScaleMap == nullptr) {
		std::cout << "Creating grayscale matrix..." << std::endl;
		CreateGrayScaleMap();
	}

	BMP image(size * tileSize, size * tileSize, true);
	ColorPixel color;

	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			try {
				ColorMapping::getColor(grayScaleMap[i * size + j].B, &color);
			}catch (std::exception& e) {
				std::cout << e.what() << std::endl;
			}
			image.FillRegion(j * tileSize, i * tileSize, tileSize, tileSize, color, 255);
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