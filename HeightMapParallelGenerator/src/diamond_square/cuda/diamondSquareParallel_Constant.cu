#include "DiamondSquareParallel_Constant.h"
#include "../parameters/applicationSettings.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Data contained in this struct change on every algorithm cycle
struct Constant
{
	uint32_t dev_Step;
	float dev_RandomScale;
};

Constant constant = {};
__constant__ Constant dev_Constant[1];
// Size is constant and never change during the algorithm execution
__constant__ uint32_t dev_Size[1];

__device__ __forceinline__ uint32_t GetIndexOnDevice (uint32_t x, uint32_t y)
{
	x = x >= *dev_Size ? *dev_Size - 1 : x;
	y = y >= *dev_Size ? *dev_Size - 1 : y;

	return x * *dev_Size + y;
}


void DiamondSquareParallel_Constant::InitializeDiamondSquare ()
{
	std::cout << " ==== CONSTANT MEMORY ====\n";
	DiamondSquareParallel::InitializeDiamondSquare();
	constant.dev_Step = step;
	constant.dev_RandomScale = randomScale;
	cudaMemcpyToSymbol (dev_Constant, &constant, sizeof(constant));
	cudaMemcpyToSymbol (dev_Size, &size, sizeof (uint32_t));
}


#pragma region Execution

void DiamondSquareParallel_Constant::DiamondSquare ()
{
#if EVENTS_TIMING
	cudaEvent_t start, stop;
	CHECK (cudaEventCreate(&start))
	CHECK (cudaEventCreate(&stop))
	CHECK (cudaEventRecord( start, 0 ))
#endif

	while (step > 1)
	{
		ComputeBlockGridSizes();

		DiamondStep();
		CHECK (cudaDeviceSynchronize())

#if PRINT_DIAMOND_STEP_CUDA
		CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
		PrintFloatMap();
#endif

		SquareStep();
		CHECK (cudaDeviceSynchronize())

#if PRINT_SQUARE_STEP_CUDA
		CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
		PrintFloatMap();
#endif

		randomScale /= 2.0f;
		step /= 2;

		constant.dev_Step = step;
		constant.dev_RandomScale = randomScale;
		cudaMemcpyToSymbol (dev_Constant, &constant, sizeof(constant));

		/* 2^k */
		threadAmount *= 2;
	}

#if COPY_RESULT_ON_HOST
	CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
#endif

#if EVENTS_TIMING
	CHECK (cudaEventRecord( stop, 0 ))
	CHECK (cudaEventSynchronize( stop ))

	CHECK (cudaEventElapsedTime( &executionTimeCuda, start, stop ))
	CHECK (cudaEventDestroy( start ))
	CHECK (cudaEventDestroy( stop ))
#endif
}

__global__ void DiamondStepParallel (float* map)
{
	uint32_t x = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
	x = x * dev_Constant->dev_Step + (dev_Constant->dev_Step / 2);
	y = y * dev_Constant->dev_Step + (dev_Constant->dev_Step / 2);

	float val = map[GetIndexOnDevice (x - (dev_Constant->dev_Step / 2), y - (dev_Constant->dev_Step / 2))] +
		map[GetIndexOnDevice (x + (dev_Constant->dev_Step / 2), y - (dev_Constant->dev_Step / 2))] +
		map[GetIndexOnDevice (x - (dev_Constant->dev_Step / 2), y + (dev_Constant->dev_Step / 2))] +
		map[GetIndexOnDevice (x + (dev_Constant->dev_Step / 2), y + (dev_Constant->dev_Step / 2))];

	val /= 4.0f;
	val += (-1.0f + map[GetIndexOnDevice (x, y)] * 2.0f) * dev_Constant->dev_RandomScale;

	map[GetIndexOnDevice (x, y)] = val;
}

__global__ void SquareStepParallel (float* map)
{
	uint32_t thd_X = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	uint32_t x = thd_X * dev_Constant->dev_Step * (y % 2 == 0)
		+ y * (dev_Constant->dev_Step / 2) * (y % 2 != 0);

	y = (y * (dev_Constant->dev_Step / 2) + (dev_Constant->dev_Step / 2)) * (y % 2 == 0)
		+ thd_X * dev_Constant->dev_Step * (y % 2 != 0);

	if (x > *dev_Size || y > *dev_Size) { return; }

	float val = (static_cast<int> (x - (dev_Constant->dev_Step / 2)) >= 0
		            && (x - (dev_Constant->dev_Step / 2)) < *dev_Size
		            && (y < *dev_Size))
		            ? map[(x - (dev_Constant->dev_Step / 2)) * *dev_Size + y]
		            : 0;

	int count = 1 * (static_cast<int> (x - (dev_Constant->dev_Step / 2)) >= 0
		&& (x - (dev_Constant->dev_Step / 2)) < *dev_Size
		&& (y < *dev_Size));

	val += (x + (dev_Constant->dev_Step / 2)) < *dev_Size
	       && (y < *dev_Size)
		       ? map[(x + (dev_Constant->dev_Step / 2)) * *dev_Size + y]
		       : 0;
	count += 1 * (x + (dev_Constant->dev_Step / 2)) < *dev_Size
		&& (y < *dev_Size);

	val += (y + (dev_Constant->dev_Step / 2) < *dev_Size)
		       ? map[x * *dev_Size + y + (dev_Constant->dev_Step / 2)]
		       : 0;
	count += 1 * y + (dev_Constant->dev_Step / 2) < *dev_Size;

	val += (static_cast<int> (y - (dev_Constant->dev_Step / 2))) >= 0
		       ? map[x * *dev_Size + y - (dev_Constant->dev_Step / 2)]
		       : 0;
	count += 1 * (static_cast<int> (y - (dev_Constant->dev_Step / 2))) >= 0;

	/*float val = map[GetIndexOnDevice (x - (dev_Constant->dev_Step / 2), y)] +
		map[GetIndexOnDevice (x + (dev_Constant->dev_Step / 2), y)] +
		map[GetIndexOnDevice (x, y - (dev_Constant->dev_Step / 2))] +
		map[GetIndexOnDevice (x, y + (dev_Constant->dev_Step / 2))];*/

	val /= count;
	val += (-1.0f + map[GetIndexOnDevice (x, y)] * 2.0f) * dev_Constant->dev_RandomScale;

	map[GetIndexOnDevice (x, y)] = val;
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

#pragma endregion
