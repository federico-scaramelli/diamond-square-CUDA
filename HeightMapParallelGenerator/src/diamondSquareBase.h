#pragma once
#include <cstdint>
#include <iostream>
#include "bmpHandler.h"

class DiamondSquareBase
{
public:
	DiamondSquareBase(const uint32_t size);

	virtual ~DiamondSquareBase();

	virtual void ExecuteDiamondSquare();
	virtual void ExecuteDiamondSquare(uint32_t initValuesDistance);

	uint32_t GetIndex(uint32_t x, uint32_t y) {
		if (x >= size) x = size - 1;
		if (y >= size) y = size - 1;
		
		return x * size + y;
	}

	void PrintMap() const;
	void PrintGrayScaleMap();

	void CreateGrayScaleMap();

	void SaveMapOnImage(const char* fname, int tileSize);

	bool CheckSizeAdequate();

protected:
	virtual void InitializeDiamondSquare(uint32_t initValuesDistance) = 0;
	virtual void DiamondSquare() = 0;
	virtual void DiamondStep(uint32_t x, uint32_t y) = 0;
	virtual void SquareStep(uint32_t x, uint32_t y) = 0;

	double* map;
	ColorPixel* grayScaleMap = nullptr;

	uint32_t size;
	uint32_t step;
	uint32_t half;

	double randomScale = 8.0;
};