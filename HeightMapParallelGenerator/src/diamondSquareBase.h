#pragma once

//std headers
#include <map>
#include <algorithm>
#include <cstdint>

//My headers
#include "bmpHandler.h"
#include "colorMapping.h"


class DiamondSquareBase
{
public:

#pragma region Constructors

	DiamondSquareBase(const uint32_t size);

	virtual ~DiamondSquareBase();

	void DeleteDoubleMap();

#pragma endregion

#pragma region Support Functions

	void CheckSizeAdequate();
	uint32_t GetIndex(uint32_t x, uint32_t y) const;

	void PrintMap() const;
	void PrintGrayScaleMap();
	
#pragma endregion

#pragma region Setter Functions

	void SetRandomScale(double randomScale);
	void SetInitialStepSize(uint32_t initValuesDistance);

#pragma endregion

#pragma region Image Functions

void GenerateGrayScaleMap();

void SaveGrayScaleImage(const char* fname, int tileSize);
void SaveColorImage(const char* fname, int tileSize);

#pragma endregion

#pragma region Execution Functions

	virtual void ExecuteDiamondSquare();

protected:
	virtual void InitializeDiamondSquare() = 0;
	virtual void DiamondSquare() = 0;
	virtual void DiamondStep(uint32_t x, uint32_t y) = 0;
	virtual void SquareStep(uint32_t x, uint32_t y) = 0;
	
#pragma endregion

#pragma region Member Attributes

	protected:
	double* map = nullptr;
	uint8_t* grayScaleMap = nullptr;

	uint32_t size;
	uint32_t step;
	uint32_t half;

	double randomScale = 5.0;

#pragma endregion
};