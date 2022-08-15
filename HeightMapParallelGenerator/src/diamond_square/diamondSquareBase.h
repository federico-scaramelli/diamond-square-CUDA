#pragma once

//std headers
#include <map>
#include <algorithm>
#include <cstdint>

//My headers
#include "../image/bmpHandler.h"
#include "./parameters/colorMapping.h"
#include "../utils/timeMeasure.h"


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
	uint32_t GetIndexOnHost(uint32_t x, uint32_t y) const;
	double* GetExecutionTime() { return &executionTime; }

	void PrintMap() const;
	void PrintGrayScaleMap();
	void PrintIntMap ();

#pragma endregion

#pragma region Setter Functions

	void SetRandomScale(float randomScale);
	void SetInitialStepSize(uint32_t initValuesDistance);

#pragma endregion

#pragma region Image Functions

virtual void MapValuesToGrayScale();
virtual void MapValuesToIntRange (int toMin, int toMax, int* outputMap);
void GenerateGrayScaleMap();

void SaveGrayScaleImage(const char* fname, int tileSize);
void SaveColorImage(const char* fname, int tileSize);

#pragma endregion

#pragma region Execution Functions

	virtual void ExecuteDiamondSquare();

protected:
	virtual void InitializeDiamondSquare() = 0;
	virtual void DiamondSquare() = 0;
	virtual void DiamondStep() = 0;
	virtual void SquareStep() = 0;

#pragma endregion

public:
	uint32_t totalSize;

#pragma region Member Attributes

protected:
	float* map = nullptr;
	uint8_t* grayScaleMap = nullptr;
	int* intMap = nullptr;

	uint32_t size;
	uint32_t step;
	uint32_t half;

	float randomScale = 5.0;

	double executionTime;

#pragma endregion
};