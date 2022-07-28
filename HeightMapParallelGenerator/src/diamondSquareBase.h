#pragma once
#include <cstdint>
#include <iostream>
#include <bitset>
#include <random>

#include "bmpHandler.h"

class DiamondSquareBase
{
public:
	explicit DiamondSquareBase(const uint32_t size) {
		this->map = new double[size * size];
		memset(map, 0.0, sizeof(double) * size * size);
		this->size = size;
		half = 0;
	}

	virtual ~DiamondSquareBase() {
		delete map;
	}

	virtual void ExecuteDiamondSquare(uint32_t initValuesDistance) {
		InitializeDiamondSquare(initValuesDistance);
		//PrintMap();
		DiamondSquare();
		//PrintMap();
	}

	uint32_t GetIndex(uint32_t x, uint32_t y) {
		return (x & (size - 1)) + (y & (size - 1)) * size;		
	}

	void PrintMap() const {
		for (uint32_t i = 0; i < size; ++i)
	    {
	        for (uint32_t j = 0; j < size; ++j)
	        {
	            std::cout << map[i * size + j] << ' ';
	        }
	        std::cout << std::endl;
	    }
	    std::cout << std::endl << std::endl;
	}

	void SaveMapOnImage(const char* fname, int tileSize) const {
		BMP image(size * tileSize, size * tileSize, true);

		/*delete map;
		map = new uint8_t[size * size]{
				0,0,0,
				100,100,100,
				255,255,255
		};*/

		for (uint32_t i = 0; i < size; ++i) {
			for (uint32_t j = 0; j < size; ++j) {
				double channel = map[i * size + j];
	            Color c{channel, channel, channel};
				image.FillRegion(j * tileSize, i * tileSize, tileSize, tileSize, c, 255);
			}
	        image.Write(fname);
		}
		
		image.Write(fname);
	}

	static bool CheckSizeAdequate(const uint32_t size) {
		if ((size - 1 & size - 2) != 0) {
			std::cout << "Size not adequate. The map size must be a power of two plus one." << std::endl;
			return false;
		}
		return true;
	}

protected:
	virtual void InitializeDiamondSquare(uint32_t initValuesDistance) = 0;
	virtual void DiamondSquare() = 0;
	virtual void DiamondStep(uint32_t x, uint32_t y) = 0;
	virtual void SquareStep(uint32_t x, uint32_t y) = 0;

	double* map;
	uint32_t size;
	
	double randomScale = 1.0;

	uint32_t step;
	uint32_t half;
};