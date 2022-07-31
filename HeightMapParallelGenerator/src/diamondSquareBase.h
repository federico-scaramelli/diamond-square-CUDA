#pragma once
#include <map>
#include <algorithm>
#include <cstdint>

#include "bmpHandler.h"
#include "colorMapping.h"


class DiamondSquareBase
{
public:
	DiamondSquareBase(const uint32_t size);

	virtual ~DiamondSquareBase();

	virtual void ExecuteDiamondSquare();
	virtual void ExecuteDiamondSquare(uint32_t initValuesDistance);

	uint32_t GetIndex(uint32_t x, uint32_t y) const {
		if (x >= size) x = size - 1;
		if (y >= size) y = size - 1;
		
		return x * size + y;
	}

	void SetRandomScale(double randomScale) {
		this->randomScale = randomScale;
	}

	void PrintMap() const;
	void PrintGrayScaleMap();

	void CreateGrayScaleMap();

	void SaveGrayScaleImage(const char* fname, int tileSize);
	void SaveColorImage(const char* fname, int tileSize);

	bool CheckSizeAdequate();

protected:
	virtual void InitializeDiamondSquare(uint32_t initValuesDistance) = 0;
	virtual void DiamondSquare() = 0;
	virtual void DiamondStep(uint32_t x, uint32_t y) = 0;
	virtual void SquareStep(uint32_t x, uint32_t y) = 0;

	double* map;
	uint8_t* grayScaleMap = nullptr;

	uint32_t size;
	uint32_t step;
	uint32_t half;

	double randomScale = 5.0;
};