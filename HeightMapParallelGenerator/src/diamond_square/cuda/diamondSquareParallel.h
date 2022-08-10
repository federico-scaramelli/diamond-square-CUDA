#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "../diamondSquareBase.h"
#include "../../utils/utils.h"

class DiamondSquareParallel : public DiamondSquareBase {
public:
	DiamondSquareParallel(uint32_t size) : DiamondSquareBase(size) {}

	void InitializeDiamondSquare() override;
	void CopyMapToDevice();
	void CalculateBlockGridSizes ();
	void PrintRandoms();
	void GenerateRandomNumbers();

	void getRandom(float* value);

	void DiamondSquare() override;
	void DiamondStep() override;
	void SquareStep() override;

	float* GetExecutionTimeCuda() { return &executionTimeCuda; }

	void CleanUp();

private:
	float* randoms = nullptr;

    float* dev_Randoms = nullptr;
    float* dev_Map = nullptr;

	/* 2^k -> k = loop step [0, n-1] */
	uint32_t threadAmount = 1;

	uint32_t blockSizeDiamond = 0;
	uint32_t blockXSizeSquare = 0;
	uint32_t blockYSizeSquare = 0;

	uint32_t gridSizeDiamond = 0;
	uint32_t gridSizeXSquare = 0;

	float executionTimeCuda;
};