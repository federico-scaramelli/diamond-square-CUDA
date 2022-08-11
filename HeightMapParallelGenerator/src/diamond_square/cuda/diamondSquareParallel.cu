#include "diamondSquareParallel.h"
#include "../parameters/applicationSettings.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma region Safety Check

#if !(RANDOM_SINGLE_BATCH | RANDOM_STEP_BATCHES | RANDOM_SINGLE_VALUE)
#define RANDOM_SINGLE_BATCH 1;
#endif

#if (RANDOM_SINGLE_BATCH + RANDOM_STEP_BATCHES + RANDOM_SINGLE_VALUE) > 1
#define RANDOM_SINGLE_BATCH		1
#define RANDOM_STEP_BATCHES		0
#define RANDOM_SINGLE_VALUE		0
#endif

#pragma endregion



__device__ __forceinline__ uint32_t GetIndex (uint32_t x, uint32_t y, uint32_t size)
{
	x = x >= size ? size - 1 : x;
	y = y >= size ? size - 1 : y;

	return x * size + y;
}


#pragma region Random Generator

#include "curand.h"
#include "curand_kernel.h"

#if RANDOM_SINGLE_VALUE
  curandState* state;
#endif

inline bool getRandom (float* const value)
{
	bool cond = static_cast<int> (*value * 10) & 0x01;
	*value = *value * (-1) * cond + *value * !cond;
	return cond;
}

void DiamondSquareParallel::PrintRandoms ()
{
	map = new float[totalSize];
	CHECK(cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
	auto count = 0;
	
	for (int i = 0; i < totalSize; i++) {
		count = getRandom(map + i) ? count + 1 : count;
		//std::cout << randoms[i];
	}
	std::cout << count << " negativi" << std::endl;
	std::cout << totalSize - count << " positivi" << std::endl;
}

#if RANDOM_SINGLE_BATCH

void DiamondSquareParallel::GenerateRandomNumbers ()
{
	int seed = RandomIntUniform();
	curandGenerator_t generator;
	CHECK_CURAND (curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937))
	CHECK_CURAND (curandSetPseudoRandomGeneratorSeed(generator, seed))

	/* Generate n floats on device */
	CHECK_CURAND (curandGenerateUniform(generator, dev_Map, totalSize))

	//PrintRandoms();

	/* Cleanup */
	CHECK_CURAND (curandDestroyGenerator(generator))
}

#endif

__device__ __forceinline__ float getRandomOnDevice (float const value)
{
	bool cond = static_cast<int> (value * 10) / 1 & 0x01;
	return value * (-1) * cond + value * !cond;
}

#pragma endregion



#pragma region Initialization

__global__ void SetupRandomState (curandState* state, int seed)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int id = x * gridDim.y * blockDim.y + y * blockDim.x;
	curand_init(seed, id, id, &state[id]);
}

__global__ void InitializeValues (float* map, int step, int size, curandState* state)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	x *= step;
	y *= step;

	float r = curand_uniform (state);
	map[GetIndex (x, y, size)] = r;
}

void DiamondSquareParallel::InitializeDiamondSquare ()
{
  	std::cout << "==================PARALLEL DIAMOND SQUARE==================\n\n";
  	std::cout << "----------INITIALIZATION----------\n";
	std::cout << "Initializing Diamond Square [" << size << " x " << size << "]...\n";

	CHECK (cudaMalloc ((void**)&dev_Map, totalSize * sizeof(float)))

	threadAmount = (size - 1) / step;

#if RANDOM_SINGLE_BATCH
	MeasureTimeFn (nullptr, "Random number set generated in ", this, 
	  &DiamondSquareParallel::GenerateRandomNumbers);
#else
	int block = threadAmount + 1;
	block = block > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : block;
	dim3 blockSize (block, block, 1);
	int grid = (threadAmount + block) / block;
	dim3 gridSize (grid, grid, 1);
	CHECK (cudaMalloc((void **)&state, block * block * grid * grid * sizeof(curandState)));
	SetupRandomState<<<gridSize, blockSize>>> (state, RandomIntUniform());
	InitializeValues<<<gridSize, blockSize>>> (dev_Map, step, size, state);
	CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
	PrintMap();
#endif
}

#pragma endregion



#pragma region Execution

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

// EXECUTION KERNELS
#if RANDOM_SINGLE_VALUE

__global__ void DiamondStepParallel (float* map, uint32_t size, uint32_t step, float randomScale, int seed)
{
	uint32_t x = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;

	curandState_t curandState;
    curand_init(
        seed,     // Seed for the random generator.
        x,       // Sequence number: used to differ returned number among cores sharing the same seed.
        y, // Offset; can be zero.
        &curandState     // State of this core's random generator.   
    );

	x = x * step + (step / 2);
	y = y * step + (step / 2);

	float val = map[GetIndex (x - (step / 2), y - (step / 2), size)] +
		map[GetIndex (x + (step / 2), y - (step / 2), size)] +
		map[GetIndex (x - (step / 2), y + (step / 2), size)] +
		map[GetIndex (x + (step / 2), y + (step / 2), size)];

	val /= 4.0f;

	map[GetIndex (x, y, size)] = (-1.0f + curand_uniform(&curandState) * 2.0f) * randomScale + val;
}

__global__ void SquareStepParallel (float* map, uint32_t size, uint32_t step, float randomScale, int seed)
{
	uint32_t thd_X = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	curandState_t curandState;
    curand_init(
        seed,     // Seed for the random generator.
        thd_X,       // Sequence number: used to differ returned number among cores sharing the same seed.
        y, // Offset; can be zero.
        &curandState     // State of this core's random generator.   
    );

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

	map[GetIndex (x, y, size)] = (-1.0f + curand_uniform(&curandState) * 2.0f) * randomScale + val;
}

#else

__global__ void DiamondStepParallel (float* map, uint32_t size, uint32_t step, float randomScale)
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

__global__ void SquareStepParallel (float* map, uint32_t size, uint32_t step, float randomScale)
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
#if RANDOM_SINGLE_VALUE
	DiamondStepParallel<<<gridDimension, blockDimension>>> (dev_Map, size, step, randomScale, RandomIntUniform());
#else
	DiamondStepParallel<<<gridDimension, blockDimension>>> (dev_Map, size, step, randomScale);
#endif
}

void DiamondSquareParallel::SquareStep ()
{
	dim3 blockDimension (blockXSizeSquare, blockYSizeSquare, 1);
	dim3 gridDimension (gridSizeXSquare, (threadAmount * 2 + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE, 1);
#if RANDOM_SINGLE_VALUE
	DiamondStepParallel<<<gridDimension, blockDimension>>> (dev_Map, size, step, randomScale, RandomIntUniform());
#else
	SquareStepParallel<<<gridDimension, blockDimension>>> (dev_Map, size, step, randomScale);
#endif
}

#pragma endregion



void DiamondSquareParallel::CleanUp ()
{
	CHECK (cudaFree(dev_Map))
}