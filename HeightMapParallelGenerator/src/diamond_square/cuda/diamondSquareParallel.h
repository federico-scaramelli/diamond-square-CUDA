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

	void PrintRandoms();
	void GenerateRandomNumbers();

	bool getRandom(float* value);

	void DiamondSquare() override;

	void DiamondStep() override;
	
	void SquareStep() override;

	void CleanUp();

private:
	float* randoms = nullptr;

    float* dev_Randoms = nullptr;
    float* dev_Map = nullptr;

	uint32_t blockSizeDiamond = 1;
	uint32_t blockSizeSquare = 4;
};