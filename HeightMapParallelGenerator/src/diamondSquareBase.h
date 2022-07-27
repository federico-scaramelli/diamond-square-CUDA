#pragma once
#include <cstdint>
#include <iostream>
#include <bitset>
#include "bmpHandler.h"

class DiamondSquareBase
{
public:
	explicit DiamondSquareBase(const uint32_t size = 9) {
		this->map = new uint8_t[size * size];
		memset(map, 0, size * size);
		this->size = size;
		step = size - 1;
		half = 0;
	}

	virtual ~DiamondSquareBase() {
		delete map;
	}

	virtual void ExecuteDiamondSquare() {
		InitializeDiamondSquare();
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
	            std::cout << static_cast<int>(map[i * size + j]) << ' ';
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
				uint8_t channel = map[i * size + j];
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
	virtual void InitializeDiamondSquare() = 0;
	virtual void DiamondSquare() = 0;
	virtual void DiamondStep(uint32_t x, uint32_t y, uint32_t step) = 0;
	virtual void SquareStep(uint32_t x, uint32_t y, uint32_t half) = 0;

	uint8_t* map;
	uint32_t size;
	int randomness = 255;

	uint32_t step;
	uint32_t half;
};