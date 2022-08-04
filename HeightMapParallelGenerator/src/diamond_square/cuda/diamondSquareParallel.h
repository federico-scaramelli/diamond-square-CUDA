#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "../diamondSquareBase.h"

class DiamondSquareParallel : public DiamondSquareBase {
public:
	DiamondSquareParallel(uint32_t size) : DiamondSquareBase(size) {}

	void InitializeDiamondSquare() override;

	void DiamondSquare() override;

	__global__ void DiamondStep() override;
	
	__global__ void SquareStep() override;

	__global__ void InitializeDiamondSquareParallel();
};