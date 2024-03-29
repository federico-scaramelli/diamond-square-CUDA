﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "diamondSquareParallel.h"
#include "../parameters/applicationSettings.h"

// Get linearized index on device (GPU)
__device__ __forceinline__ uint32_t GetIndexOnDevice (uint32_t x, uint32_t y, uint32_t size)
{
	x = x >= size ? size - 1 : x;
	y = y >= size ? size - 1 : y;

	return x * size + y;
}

#pragma region Random Generator

#include "curand.h"
#include "curand_kernel.h"

// Used only to test the distribution of positive and negative values obtained by cuRAND generation
inline bool IsRandomNegative (float* const value)
{
	*value = -1.0f + *value * 2.0f;
	return *value < 0;
}

// Used only to test the distribution of positive and negative values obtained by cuRAND generation
void DiamondSquareParallel::PrintRandoms ()
{
	map = new float[totalSize];
	CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
	auto count = 0;

	for (int i = 0; i < totalSize; i++)
	{
		count = IsRandomNegative (map + i) ? count + 1 : count;
		//std::cout << randoms[i];
	}
	std::cout << count << " negativi" << std::endl;
	std::cout << totalSize - count << " positivi" << std::endl;
}

// Generate all the random numbers (all the map) using cuRAND host API
void DiamondSquareParallel::GenerateRandomNumbers_HostAPI ()
{
	// Random seed = random number generated on the CPU
	int seed = RandomIntUniform();
	// Create and set the generator
	curandGenerator_t generator;
	CHECK_CURAND (curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937))
	CHECK_CURAND (curandSetGeneratorOrdering(generator, CURAND_ORDERING_PSEUDO_BEST));
	CHECK_CURAND (curandSetPseudoRandomGeneratorSeed(generator, seed))

	// Generate n floats on device memory
	CHECK_CURAND (curandGenerateUniform(generator, dev_Map, totalSize))

	// PrintRandoms();

	/* Cleanup */
	CHECK_CURAND (curandDestroyGenerator(generator))
}

// Kernel to setup the cuRAND Device API random generator
__global__ void SetupRandomGenerator (curandStateMRG32k3a* state, const int n, const uint32_t totalSize, const int seed)
{
	uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx * n > totalSize) return;

	// seed + idx insted of seed, idx: https://github.com/rogerallen/raytracinginoneweekendincuda/issues/2
	curand_init (seed + idx, 0, 0, &state[idx]);
}

// Kernel to generate random numbers with cuRAND Device API
__global__ void GenerateRandomNumbers (float* map, curandStateMRG32k3a* state, const int n, const uint32_t totalSize)
{
	uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx * n > totalSize) return;
	
	// Copy state to local memory for efficiency
	curandStateMRG32k3a localState = state[idx];

	// Generate pseudo-random uniforms 
	for (int i = 0; i < n; i++)
	{
		if (idx * n + i > totalSize) return;
		map[idx * n + i] = curand_uniform_double (&localState);
	}
	// Copy state back to global memory 
	state[idx] = localState;
}

// Method do call the random numbers generation kernels using cuRAND Device API
void DiamondSquareParallel::GenerateRandomNumbers_DeviceAPI ()
{
	// Define the amount of values to be generated for each thread. 
	int n;

	// 1D blocks and grid to call the kernels
	// The maximum size for the grid is defined on the Application Settings.
	// Empirically: BLOCK_SIZE_1D / 2 (128) gives better results 
	dim3 blockSize (BLOCK_SIZE_1D / 2, 1, 1);
	auto grid = (totalSize + blockSize.x - 1) / blockSize.x > MAX_GRID_SIZE_1D
										? MAX_GRID_SIZE_1D
										: (totalSize + blockSize.x - 1) / blockSize.x;
	n = grid == MAX_GRID_SIZE_1D
					? (totalSize + blockSize.x * MAX_GRID_SIZE_1D - 1) / (blockSize.x * MAX_GRID_SIZE_1D)
					: 1;

	// Allocate memory for the cuRAND states
	CHECK (cudaMalloc((void **)&dev_MRGStates, (totalSize + n - 1) / n * sizeof(curandStateMRG32k3a)))

	dim3 gridSize (grid, 1, 1);

	SetupRandomGenerator<<<gridSize, blockSize>>> (dev_MRGStates, n, totalSize, RandomIntUniform());
	CHECK (cudaDeviceSynchronize())

	GenerateRandomNumbers<<<gridSize, blockSize>>> (dev_Map, dev_MRGStates, n, totalSize);
	CHECK (cudaDeviceSynchronize())
}

#pragma endregion


#pragma region Initialization

// Allocate memory on device to contain the overall map
void DiamondSquareParallel::AllocateMapOnDevice ()
{
	CHECK (cudaMalloc ((void**)&dev_Map, totalSize * sizeof(float)))

	CHECK (cudaMalloc ((void**)&dev_Min, sizeof(float)))
	CHECK (cudaMalloc ((void**)&dev_Max, sizeof(float)))
}

// Initialization step: random numbers generation and memory allocation
void DiamondSquareParallel::InitializeDiamondSquare ()
{
	std::cout << " ==== PARALLEL DIAMOND SQUARE ====\n\n";
	std::cout << " - INITIALIZATION - \n";
	std::cout << "Initializing Diamond Square [" << size << " x " << size << "]...\n";

	MeasureTimeFn (nullptr, "Allocation time on device is ", this, &DiamondSquareParallel::AllocateMapOnDevice);

	threadAmount = (size - 1) / step;

#if !RAND_DEVICE_API
	MeasureTimeFn (nullptr, "Random number set with host API generated in ", this,
	               &DiamondSquareParallel::GenerateRandomNumbers_HostAPI);
#else
	MeasureTimeFn (nullptr, "Random number set with device API generated in ", this,
	               &DiamondSquareParallel::GenerateRandomNumbers_DeviceAPI);
#endif
}

#pragma endregion


#pragma region Execution

// Compute the blocks and grid sizes on each step of the algorithm
void DiamondSquareParallel::ComputeBlockGridSizes ()
{
	/*			  2^k			  or			  MAX_BLOCK_SIZE			  */
	blockSizeDiamond = threadAmount <= MAX_BLOCK_SIZE ? threadAmount : MAX_BLOCK_SIZE;
	/*		(2^k + 1) x 2^(k+1)	  or	SQUARE_BLOCK_X_SIZE x MAX_BLOCK_SIZE
	*		        k <= 3					     k > 3						  */
	blockXSizeSquare = threadAmount < SQUARE_BLOCK_X_SIZE ? blockSizeDiamond + 1 : SQUARE_BLOCK_X_SIZE;
	blockYSizeSquare = threadAmount <= SQUARE_BLOCK_X_SIZE ? threadAmount * 2 : blockSizeDiamond;

	/*				  1			  or			2^k / MAX_BLOCK_SIZE		  */
	gridSizeDiamond = (threadAmount + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	/* SQUARE_BLOCK_X_SIZE x MAX_BLOCK_SIZE			block amount
	 * = (2^(k+1) / MAX_BLOCK_SIZE)  x	 (2^k / SQUARE_BLOCK_X_SIZE) + 1	  */
	gridSizeXSquare = threadAmount < SQUARE_BLOCK_X_SIZE ? 1 : (threadAmount / SQUARE_BLOCK_X_SIZE) + 1;
	gridSizeYSquare = (threadAmount * 2 + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
}

// Execute the algorithm cycles
void DiamondSquareParallel::DiamondSquare ()
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

__global__ void DiamondStepParallel (float* map, const uint32_t size, const uint32_t step, const float randomScale)
{
	uint32_t x = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
	x = x * step + (step / 2);
	y = y * step + (step / 2);

	float val = map[GetIndexOnDevice (x - (step / 2), y - (step / 2), size)] +
		map[GetIndexOnDevice (x + (step / 2), y - (step / 2), size)] +
		map[GetIndexOnDevice (x - (step / 2), y + (step / 2), size)] +
		map[GetIndexOnDevice (x + (step / 2), y + (step / 2), size)];

	val /= 4.0f;
	val += (-1.0f + map[GetIndexOnDevice (x, y, size)] * 2.0f) * randomScale;

	map[GetIndexOnDevice (x, y, size)] = val;

	// Lowers performance to 30%
	/*AtomicMinFloat(min, val);
	AtomicMaxFloat(max, val);*/
}

__global__ void SquareStepParallel (float* map, uint32_t size, uint32_t step, float randomScale)
{
	uint32_t thd_X = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	uint32_t x = thd_X * step * (y % 2 == 0)
		+ y * (step / 2) * (y % 2 != 0);

	y = (y * (step / 2) + (step / 2)) * (y % 2 == 0)
		+ thd_X * step * (y % 2 != 0);

	if (x > size || y > size) { return; }

	float val = (static_cast<int> (x - (step / 2)) >= 0
		            && (x - (step / 2)) < size
		            && (y < size))
		            ? map[(x - (step / 2)) * size + y]
		            : 0;

	int count = 1 * (static_cast<int> (x - (step / 2)) >= 0
		&& (x - (step / 2)) < size
		&& (y < size));

	val += (x + (step / 2)) < size
	       && (y < size)
		       ? map[(x + (step / 2)) * size + y]
		       : 0;
	count += 1 * (x + (step / 2)) < size
		&& (y < size);

	val += (y + (step / 2) < size)
		       ? map[x * size + y + (step / 2)]
		       : 0;
	count += 1 * y + (step / 2) < size;

	val += (static_cast<int> (y - (step / 2))) >= 0
		       ? map[x * size + y - (step / 2)]
		       : 0;
	count += 1 * (static_cast<int> (y - (step / 2))) >= 0;

	val /= count;
	val += (-1.0f + map[GetIndexOnDevice (x, y, size)] * 2.0f) * randomScale;

	map[GetIndexOnDevice (x, y, size)] = val;

	// Lowers performance to 30%
	/*AtomicMinFloat(min, val);
	AtomicMaxFloat(max, val);*/
}

void DiamondSquareParallel::DiamondStep ()
{
	dim3 blockDimension (blockSizeDiamond, blockSizeDiamond, 1);
	dim3 gridDimension (gridSizeDiamond, gridSizeDiamond, 1);

	DiamondStepParallel<<<gridDimension, blockDimension>>> (dev_Map, size, step, randomScale);
}

void DiamondSquareParallel::SquareStep ()
{
	dim3 blockDimension (blockXSizeSquare, blockYSizeSquare, 1);
	dim3 gridDimension (gridSizeXSquare, gridSizeYSquare, 1);

	SquareStepParallel<<<gridDimension, blockDimension>>> (dev_Map, size, step, randomScale);
}

#pragma endregion


#pragma region Values Mapping

// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250
__device__ __forceinline__ float AtomicMinFloat (float* addr, float value)
{
	float min;
	min = (value >= 0)
		      ? __int_as_float (atomicMin (reinterpret_cast<int*> (addr), __float_as_int (value)))
		      : __uint_as_float (atomicMax (reinterpret_cast<unsigned int*> (addr), __float_as_uint (value)));

	return min;
}

// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250
__device__ __forceinline__ float AtomicMaxFloat (float* addr, float value)
{
	float max;
	max = (value >= 0)
		      ? __int_as_float (atomicMax (reinterpret_cast<int*> (addr), __float_as_int (value)))
		      : __uint_as_float (atomicMin (reinterpret_cast<unsigned int*> (addr), __float_as_uint (value)));

	return max;
}


__global__ void ComputeMinMax (const float* map, float* myMin, float* myMax, const uint32_t size)
{
	// Shared memory to find the minimum and maximum of a single 1D block
	__shared__ float blockMin[BLOCK_SIZE_1D];
	__shared__ float blockMax[BLOCK_SIZE_1D];

	uint32_t linearIdx = blockIdx.x * blockDim.x + threadIdx.x;

	// I use FLT_MAX and FLT_MIN to avoid that values not to be considered as they are outside the map
	// influence the search for the minimum and maximum
	bool cond = linearIdx >= size * size;
	blockMin[threadIdx.x] = cond ? FLT_MAX : blockMin[threadIdx.x];
	blockMax[threadIdx.x] = cond ? -FLT_MAX : blockMax[threadIdx.x];
	if (cond) { return; }

	// Cache the values on shared memory
	blockMin[threadIdx.x] = map[linearIdx];
	blockMax[threadIdx.x] = map[linearIdx];

	// Block sync barrier
	__syncthreads();

	// Parallel reduction on blocks to push minimum and maximum in the direction of the first indices
	for (uint8_t stride = blockDim.x / 2; stride > 32; stride >>= 1)
	{
		if (threadIdx.x < stride)
			blockMin[threadIdx.x] = min (blockMin[threadIdx.x], blockMin[threadIdx.x + stride]);
		if (threadIdx.x < stride)
			blockMax[threadIdx.x] = max (blockMax[threadIdx.x], blockMax[threadIdx.x + stride]);

		__syncthreads();
	}

	// Warp unrolling
	if (threadIdx.x < 32)
	{
		volatile float* _min = blockMin;
		volatile float* _max = blockMax;

		_min[threadIdx.x] = min (_min[threadIdx.x], _min[threadIdx.x + 32]);
		_min[threadIdx.x] = min (_min[threadIdx.x], _min[threadIdx.x + 16]);
		_min[threadIdx.x] = min (_min[threadIdx.x], _min[threadIdx.x + 8]);
		_min[threadIdx.x] = min (_min[threadIdx.x], _min[threadIdx.x + 4]);
		_min[threadIdx.x] = min (_min[threadIdx.x], _min[threadIdx.x + 2]);
		_min[threadIdx.x] = min (_min[threadIdx.x], _min[threadIdx.x + 1]);

		_max[threadIdx.x] = max (_max[threadIdx.x], _max[threadIdx.x + 32]);
		_max[threadIdx.x] = max (_max[threadIdx.x], _max[threadIdx.x + 16]);
		_max[threadIdx.x] = max (_max[threadIdx.x], _max[threadIdx.x + 8]);
		_max[threadIdx.x] = max (_max[threadIdx.x], _max[threadIdx.x + 4]);
		_max[threadIdx.x] = max (_max[threadIdx.x], _max[threadIdx.x + 2]);
		_max[threadIdx.x] = max (_max[threadIdx.x], _max[threadIdx.x + 1]);
	}

	// Float atomic operations to find the global (not block-related) minimum and maximum
	if (threadIdx.x == 0) AtomicMinFloat (myMin, blockMin[0]);
	if (threadIdx.x == 0) AtomicMaxFloat (myMax, blockMax[0]);
}

__global__ void MapFloatToIntRange (const float* map, int* outputMap,
                                    const float* fromMin, const float* fromMax,
                                    const int toMin, const int toMax, const uint32_t size)
{
	uint32_t linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (linearIdx >= size * size) return;

	int value = __float2int_rd((map[linearIdx] - *fromMin) / (*fromMax - *fromMin) * (toMax - toMin) + toMin);

	outputMap[linearIdx] = value;
}

// Another method is used because the grayscale map is uint8
__global__ void MapFloatToGrayScale (const float* map, uint8_t* grayScaleMap,
                                     const float* fromMin, const float* fromMax,
                                     const int toMin, const int toMax, const uint32_t size)
{
	uint32_t linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (linearIdx >= size * size) return;

	auto value = static_cast<uint8_t> ((map[linearIdx] - *fromMin) / (*fromMax - *fromMin) * (toMax - toMin) + toMin);

	grayScaleMap[linearIdx] = value;
}

// Map all the values from float to grayscale
void DiamondSquareParallel::MapValuesToGrayScale ()
{
	std::cout << "\n - VALUES MAPPING - " << std::endl;
	std::cout << "Mapping values to gray scale..." << std::endl;

	CHECK (cudaMalloc ((void**) &dev_GrayScaleMap, totalSize * sizeof(uint8_t)))

	int gridSize = (totalSize + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

	// Kernel to find the minimum and maximum value present in the map
	ComputeMinMax<<<gridSize, BLOCK_SIZE_1D>>> (dev_Map, dev_Min, dev_Max, size);
	CHECK (cudaMemcpy (&min, dev_Min, sizeof(float), cudaMemcpyDeviceToHost))
	CHECK (cudaMemcpy (&max, dev_Max, sizeof(float), cudaMemcpyDeviceToHost))

	CHECK (cudaDeviceSynchronize());

	// Kernel to map the values to grayscale depending on the min and max values
	MapFloatToGrayScale<<<gridSize, BLOCK_SIZE_1D>>> (dev_Map, dev_GrayScaleMap, dev_Min, dev_Max, 0, 255, size);

	delete[] grayScaleMap;
	grayScaleMap = new uint8_t[totalSize]{ 0 };
	CHECK (cudaMemcpy (grayScaleMap, dev_GrayScaleMap, totalSize * sizeof(uint8_t), cudaMemcpyDeviceToHost))

	CHECK (cudaFree (dev_GrayScaleMap))
	if (DELETE_FLOAT_MAP)
	{
		DeleteFloatMap();
	}
}

void DiamondSquareParallel::MapValuesToIntRange (int toMin, int toMax)
{
	std::cout << "\n - VALUES MAPPING - " << std::endl;
	std::cout << "Mapping values to int range..." << std::endl;

	CHECK (cudaMalloc ((void**) &dev_IntMap, totalSize * sizeof(int)))

	int gridSize = (totalSize + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

	ComputeMinMax<<<gridSize, BLOCK_SIZE_1D>>> (dev_Map, dev_Min, dev_Max, size);
	CHECK (cudaMemcpy (&min, dev_Min, sizeof(float), cudaMemcpyDeviceToHost))
	CHECK (cudaMemcpy (&max, dev_Max, sizeof(float), cudaMemcpyDeviceToHost))

	CHECK (cudaDeviceSynchronize());

	MapFloatToIntRange<<<gridSize, BLOCK_SIZE_1D>>> (dev_Map, dev_IntMap, dev_Min, dev_Max, toMin, toMax, size);

	delete[] intMap;
	intMap = new int[totalSize]{ 0 };
	CHECK (cudaMemcpy (intMap, dev_IntMap, totalSize * sizeof(int), cudaMemcpyDeviceToHost))

	CHECK (cudaFree (dev_IntMap))
	if (DELETE_FLOAT_MAP)
	{
		DeleteFloatMap();
	}
}
#pragma endregion

void DiamondSquareParallel::CleanUp ()
{
	CHECK (cudaFree(dev_Map))
	CHECK (cudaFree(dev_Min))
	CHECK (cudaFree(dev_Max))
#if RAND_DEVICE_API
	CHECK (cudaFree(dev_MRGStates))
#endif
}
