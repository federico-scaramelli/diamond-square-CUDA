#include "DiamondSquareParallel_Constant.h"
#include "../parameters/applicationSettings.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct Constant
{
	uint32_t dev_Step;
	float dev_RandomScale;
};
Constant constant = {};
__constant__ Constant dev_Constant[1];
__constant__ uint32_t dev_Size[1];

__device__ __forceinline__ float GetRandomOnDevice (float const value)
{
	bool cond = static_cast<int> (value * 10) / 1 & 0x01;
	return value * (-1) * cond + value * !cond;
}

__device__ __forceinline__ uint32_t GetIndex (uint32_t x, uint32_t y)
{
	x = x >= *dev_Size ? *dev_Size - 1 : x;
	y = y >= *dev_Size ? *dev_Size - 1 : y;

	return x * *dev_Size + y;
}

__global__ void DiamondStepParallel (float* map)
{
	uint32_t x = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
	x = x * dev_Constant->dev_Step + (dev_Constant->dev_Step / 2);
	y = y * dev_Constant->dev_Step + (dev_Constant->dev_Step / 2);
	
	float val = map[GetIndex (x - (dev_Constant->dev_Step / 2), y - (dev_Constant->dev_Step / 2))] +
				map[GetIndex (x + (dev_Constant->dev_Step / 2), y - (dev_Constant->dev_Step / 2))] +
				map[GetIndex (x - (dev_Constant->dev_Step / 2), y + (dev_Constant->dev_Step / 2))] +
				map[GetIndex (x + (dev_Constant->dev_Step / 2), y + (dev_Constant->dev_Step / 2))];

	val /= 4.0f;

	map[GetIndex (x, y)] = GetRandomOnDevice(map[GetIndex (x, y)]) * dev_Constant->dev_RandomScale + val;
}

__global__ void SquareStepParallel (float* map)
{
	uint32_t thd_X = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	uint32_t x =  thd_X * dev_Constant->dev_Step  * (y % 2 == 0) +
				  y * (dev_Constant->dev_Step / 2) * (y % 2 != 0);
			 y = (y * (dev_Constant->dev_Step / 2) + (dev_Constant->dev_Step / 2)) * (y % 2 == 0) +
				  thd_X * dev_Constant->dev_Step  * (y % 2 != 0);

	if (x > *dev_Size || y > *dev_Size) {
		return;
	}
	
	float val = map[GetIndex (x - (dev_Constant->dev_Step / 2), y)] +
				map[GetIndex (x + (dev_Constant->dev_Step / 2), y)] +
				map[GetIndex (x, y - (dev_Constant->dev_Step / 2))] +
				map[GetIndex (x, y + (dev_Constant->dev_Step / 2))];

	val /= 4.0f;

	map[GetIndex (x, y)] = GetRandomOnDevice(map[GetIndex (x, y)]) * dev_Constant->dev_RandomScale + val;
}

void DiamondSquareParallel_Constant::InitializeDiamondSquare ()
{
	DiamondSquareParallel::InitializeDiamondSquare();
	constant.dev_Step = step;
	constant.dev_RandomScale = randomScale;
	cudaMemcpyToSymbol(dev_Constant, &constant, sizeof(constant));
	cudaMemcpyToSymbol(dev_Size, &size, sizeof(uint32_t));
}

void DiamondSquareParallel_Constant::DiamondSquare ()
{
	#if CUDA_EVENTS_TIMING
	cudaEvent_t start, stop;
	CHECK (cudaEventCreate(&start))
	CHECK (cudaEventCreate(&stop))
	CHECK (cudaEventRecord( start, 0 ))
#endif

	while (step > 1) {
		ComputeBlockGridSizes();
		DiamondStep();
		CHECK (cudaDeviceSynchronize())

#if PRINT_DIAMOND_STEP_CUDA
		CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
		PrintMap();
#endif

		SquareStep();
		CHECK (cudaDeviceSynchronize())

#if PRINT_SQUARE_STEP_CUDA
		CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
		PrintMap();
#endif

		randomScale /= 2.0f;
		step /= 2;
		
		constant.dev_Step = step;
		constant.dev_RandomScale = randomScale;
		cudaMemcpyToSymbol(dev_Constant, &constant, sizeof(constant));

		/* 2^k */
		threadAmount *= 2;
	}

#if COPY_RESULT_ON_HOST
	CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
#endif

	CleanUp();

#if CUDA_EVENTS_TIMING
	CHECK (cudaEventRecord( stop, 0 ))
	CHECK (cudaEventSynchronize( stop ))

	CHECK (cudaEventElapsedTime( &executionTimeCuda, start, stop ))
	CHECK (cudaEventDestroy( start ))
	CHECK (cudaEventDestroy( stop ))
#endif
}

void DiamondSquareParallel_Constant::DiamondStep ()
{
	dim3 blockDimension (blockSizeDiamond, blockSizeDiamond, 1);
	dim3 gridDimension (gridSizeDiamond, gridSizeDiamond, 1);
	DiamondStepParallel<<<gridDimension, blockDimension>>> (dev_Map);
}

void DiamondSquareParallel_Constant::SquareStep ()
{
	dim3 blockDimension (blockXSizeSquare, blockYSizeSquare, 1);
	dim3 gridDimension (gridSizeXSquare, gridSizeYSquare, 1);
	SquareStepParallel<<<gridDimension, blockDimension>>> (dev_Map);
}