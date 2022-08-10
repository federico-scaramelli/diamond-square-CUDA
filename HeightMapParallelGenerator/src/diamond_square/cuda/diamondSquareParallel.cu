#include "diamondSquareParallel.h"
#include "../parameters/applicationSettings.h"

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

#pragma region Random Generator

#include "curand.h"

void DiamondSquareParallel::PrintRandoms ()
{
	/*randoms = new float[totalSize];
	CHECK(cudaMemcpy(randoms, dev_Randoms, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
	auto count = 0;
	/* Show result #1#
	for (int i = 0; i < totalSize; i++) {
		count = getRandom(randoms + i) ? count + 1 : count;
		//std::cout << randoms[i];
	}
	std::cout << count << " negativi" << std::endl;
	std::cout << totalSize - count << " positivi" << std::endl;
	delete[] randoms;*/
}

void DiamondSquareParallel::GenerateRandomNumbers ()
{
	int seed = RandomIntUniform();
	curandGenerator_t generator;
	CHECK_CURAND (curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937))
	CHECK_CURAND (curandSetPseudoRandomGeneratorSeed(generator, seed))

	/* Allocate n floats on device */
	CHECK (cudaMalloc((void **)&dev_Map, totalSize * sizeof(float)))

	/* Generate n floats on device */
	CHECK_CURAND (curandGenerateUniform(generator, dev_Map, totalSize))

	//PrintRandoms();

	/* Cleanup */
	CHECK_CURAND (curandDestroyGenerator(generator))
}

inline void DiamondSquareParallel::getRandom (float* const value)
{
	bool cond = static_cast<int> (*value * 10) / 1 & 0x01;
	*value = *value * (-1) * cond + *value * !cond;
}

__device__ __forceinline__ float getRandomOnDevice (float const value)
{
	bool cond = static_cast<int> (value * 10) / 1 & 0x01;
	return value * (-1) * cond + value * !cond;
}

#pragma endregion

#if CONSTANT_MEMORY

struct Constant
{
	uint32_t dev_Size;
	uint32_t dev_Step;
	float dev_RandomScale;
};
Constant constant = {};
__constant__ Constant dev_Constant[1];

__device__ __forceinline__ uint32_t GetIndex (uint32_t x, uint32_t y)
{
	x = x >= dev_Constant->dev_Size ? dev_Constant->dev_Size - 1 : x;
	y = y >= dev_Constant->dev_Size ? dev_Constant->dev_Size - 1 : y;

	return x * dev_Constant->dev_Size + y;
}

__global__ void InitializeDiamondSquareParallel (float* map, const float* randoms)
{
	uint32_t thd_X = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t thd_Y = blockIdx.x * blockDim.x + threadIdx.x;
	thd_X *= dev_Constant->dev_Step;
	thd_Y *= dev_Constant->dev_Step;

	map[GetIndex (thd_X, thd_Y)] = randoms[GetIndex (thd_X, thd_Y)]; 
}

#else

__device__ __forceinline__ uint32_t GetIndex (uint32_t x, uint32_t y, uint32_t size)
{
	x = x >= size ? size - 1 : x;
	y = y >= size ? size - 1 : y;

	return x * size + y;
}

#endif

void DiamondSquareParallel::InitializeDiamondSquare ()
{
  	std::cout << "==================PARALLEL DIAMOND SQUARE==================" << std::endl << std::endl;
  	std::cout << "----------INITIALIZATION----------" << std::endl;
	std::cout << "Initializing Diamond Square [" << size << " x " << size << "]..." << std::endl;

	MeasureTimeFn (nullptr, "Random number set generated in ", this, 
	  &DiamondSquareParallel::GenerateRandomNumbers);

	threadAmount = (size - 1) / step;
	uint32_t blockSize = threadAmount <= MAX_BLOCK_SIZE ? threadAmount : MAX_BLOCK_SIZE;
	uint32_t gridSize = (threadAmount + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	dim3 blockDimension (blockSize, blockSize, 1);
	dim3 gridDimension (gridSize, gridSize, 1);

#if CONSTANT_MEMORY
	constant.dev_Size = size;
	constant.dev_Step = step;
	constant.dev_RandomScale = randomScale;
	cudaMemcpyToSymbol(dev_Constant, &constant, sizeof(constant));
#endif
}

void DiamondSquareParallel::CopyMapToDevice ()
{
	/* Copy the map on the device memory */
	CHECK (cudaMalloc(&dev_Map, totalSize * sizeof(float)))
	CHECK (cudaMemcpy(dev_Map, map, totalSize * sizeof(float), cudaMemcpyHostToDevice))
}

void DiamondSquareParallel::CalculateBlockGridSizes ()
{
	/*			  2^k			  or			  MAX_BLOCK_SIZE			  */
	blockSizeDiamond = threadAmount <= MAX_BLOCK_SIZE ? threadAmount : MAX_BLOCK_SIZE;
	/*		(2^k + 1) x 2^(k+1)	  or	SQUARE_BLOCK_X_SIZE x MAX_BLOCK_SIZE
	*		        k <= 3					     k > 3						  */
	blockXSizeSquare = threadAmount <= SQUARE_BLOCK_X_SIZE ? blockSizeDiamond + 1 : SQUARE_BLOCK_X_SIZE;
	blockYSizeSquare = threadAmount <= SQUARE_BLOCK_X_SIZE ? threadAmount * 2	  : blockSizeDiamond;

	/*				  1			  or			2^k / MAX_BLOCK_SIZE		  */
	gridSizeDiamond = (threadAmount + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	/* SQUARE_BLOCK_X_SIZE x MAX_BLOCK_SIZE			block amount
	 * = (2^(k+1) / MAX_BLOCK_SIZE)  x	 (2^k / SQUARE_BLOCK_X_SIZE) + 1	  */
	gridSizeXSquare = threadAmount <= SQUARE_BLOCK_X_SIZE ? 1 : (threadAmount / SQUARE_BLOCK_X_SIZE) + 1;
}

void DiamondSquareParallel::DiamondSquare ()
{
#if CUDA_EVENTS_TIMING
	cudaEvent_t start, stop;
	CHECK (cudaEventCreate(&start))
	CHECK (cudaEventCreate(&stop))
	CHECK (cudaEventRecord( start, 0 ))
#endif

	while (step > 1) {
		CalculateBlockGridSizes();

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

#if CONSTANT_MEMORY
		constant.dev_Step = step;
		constant.dev_RandomScale = randomScale;
		cudaMemcpyToSymbol(dev_Constant, &constant, sizeof(constant));
#endif

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

#if CONSTANT_MEMORY

__global__ void DiamondStepParallel (float* map, float* randoms)
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

	map[GetIndex (x, y)] = getRandomOnDevice(map[GetIndex (x, y)]) * dev_Constant->dev_RandomScale + val;
}

__global__ void SquareStepParallel (float* map, float* randoms)
{
	uint32_t thd_X = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	uint32_t x =  thd_X * dev_Constant->dev_Step  * (y % 2 == 0) +
				  y * (dev_Constant->dev_Step / 2) * (y % 2 != 0);
			 y = (y * (dev_Constant->dev_Step / 2) + (dev_Constant->dev_Step / 2)) * (y % 2 == 0) +
				  thd_X * dev_Constant->dev_Step  * (y % 2 != 0);

	if (x > dev_Constant->dev_Size || y > dev_Constant->dev_Size) {
		return;
	}
	
	float val = map[GetIndex (x - (dev_Constant->dev_Step / 2), y)] +
				map[GetIndex (x + (dev_Constant->dev_Step / 2), y)] +
				map[GetIndex (x, y - (dev_Constant->dev_Step / 2))] +
				map[GetIndex (x, y + (dev_Constant->dev_Step / 2))];

	val /= 4.0f;

	map[GetIndex (x, y)] = getRandomOnDevice(map[GetIndex (x, y)]) * dev_Constant->dev_RandomScale + val;
}

#else

__global__ void DiamondStepParallel (float* map, float* randoms, uint32_t size, uint32_t step, float randomScale)
{
	uint32_t x = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
	x = x * step + (step / 2);
	y = y * step + (step / 2);

	float val = map[GetIndex (x - (step / 2), y - (step / 2), size)] +
		map[GetIndex (x + (step / 2), y - (step / 2), size)] +
		map[GetIndex (x - (step / 2), y + (step / 2), size)] +
		map[GetIndex (x + (step / 2), y + (step / 2), size)];

	val /= 4.0f;

	map[GetIndex (x, y, size)] = getRandomOnDevice(map[GetIndex (x, y, size)]) * randomScale + val;
}

__global__ void SquareStepParallel (float* map, float* randoms, uint32_t size, uint32_t step, float randomScale)
{
	uint32_t thd_X = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	uint32_t x =  thd_X * step * (y % 2 == 0) +
				  y * (step / 2) * (y % 2 != 0);
			 y = (y * (step / 2) + (step / 2)) * (y % 2 == 0) +
				  thd_X * step * (y % 2 != 0);

	if (x > size || y > size) {
		return;
	}
	
	float val = map[GetIndex (x - (step / 2), y, size)] +
		map[GetIndex (x + (step / 2), y, size)] +
		map[GetIndex (x, y - (step / 2), size)] +
		map[GetIndex (x, y + (step / 2), size)];

	val /= 4.0f;

	map[GetIndex (x, y, size)] = getRandomOnDevice(map[GetIndex (x, y, size)]) * randomScale + val;
}

#endif

void DiamondSquareParallel::DiamondStep ()
{
	dim3 blockDimension (blockSizeDiamond, blockSizeDiamond, 1);
	dim3 gridDimension (gridSizeDiamond, gridSizeDiamond, 1);
#if CONSTANT_MEMORY
	DiamondStepParallel<<<gridDimension, blockDimension>>> (dev_Map, dev_Randoms);
#else
	DiamondStepParallel<<<gridDimension, blockDimension>>> (dev_Map, dev_Randoms, size, step, randomScale);
#endif
}

void DiamondSquareParallel::SquareStep ()
{
	dim3 blockDimension (blockXSizeSquare, blockYSizeSquare, 1);
	dim3 gridDimension (gridSizeXSquare, (threadAmount * 2 + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE, 1);
#if CONSTANT_MEMORY
	SquareStepParallel<<<gridDimension, blockDimension>>> (dev_Map, dev_Randoms);
#else
	SquareStepParallel<<<gridDimension, blockDimension>>> (dev_Map, dev_Randoms, size, step, randomScale);
#endif
}

void DiamondSquareParallel::CleanUp ()
{
	CHECK (cudaFree(dev_Randoms))
	CHECK (cudaFree(dev_Map))
}