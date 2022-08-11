#pragma once
#include <stdio.h>

#include "../diamondSquareBase.h"
#include "../../utils/utils.h"

#pragma region CheckCUDACalls

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#pragma endregion

class DiamondSquareParallel : public DiamondSquareBase {
public:
	DiamondSquareParallel(uint32_t size) : DiamondSquareBase(size) {}

	void InitializeDiamondSquare() override;
	void CalculateBlockGridSizes ();
	void PrintRandoms();
	void GenerateRandomNumbers();

	void DiamondSquare() override;
	void DiamondStep() override;
	void SquareStep() override;

	float* GetExecutionTimeCuda() { return &executionTimeCuda; }

	void CleanUp();

protected:
    float* dev_Map = nullptr;

	/* 2^k -> k = loop step [0, n-1] */
	uint32_t threadAmount = 1;

	uint32_t blockSizeDiamond = 0;
	uint32_t blockXSizeSquare = 0;
	uint32_t blockYSizeSquare = 0;

	uint32_t gridSizeDiamond = 0;
	uint32_t gridSizeXSquare = 0;

	float executionTimeCuda = 0;
};