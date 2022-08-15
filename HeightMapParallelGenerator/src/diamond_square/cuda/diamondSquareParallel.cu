#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "diamondSquareParallel.h"
#include "../parameters/applicationSettings.h"

__device__ __forceinline__ uint32_t GetIndexOnDevice (uint32_t x, uint32_t y, uint32_t size)
{
	x = x >= size ? size - 1 : x;
	y = y >= size ? size - 1 : y;

	return x * size + y;
}


#pragma region Random Generator

#include "curand.h"
#include "curand_kernel.h"

inline bool GetRandomOnHost (float* const value)
{
	bool cond = static_cast<int> (*value * 10) & 0x01;
	*value = *value * (-1) * cond + *value * !cond;
	return cond;
}

void DiamondSquareParallel::PrintRandoms ()
{
	map = new float[totalSize];
	CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
	auto count = 0;

	for (int i = 0; i < totalSize; i++) {
		count = GetRandomOnHost (map + i) ? count + 1 : count;
		//std::cout << randoms[i];
	}
	std::cout << count << " negativi" << std::endl;
	std::cout << totalSize - count << " positivi" << std::endl;
}

void DiamondSquareParallel::GenerateRandomNumbers_HostAPI ()
{
	int seed = RandomIntUniform();
	curandGenerator_t generator;
	CHECK_CURAND (curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937))
	CHECK_CURAND (curandSetGeneratorOrdering(generator, CURAND_ORDERING_PSEUDO_BEST));
	CHECK_CURAND (curandSetPseudoRandomGeneratorSeed(generator, seed))

	/* Generate n floats on device */
	CHECK_CURAND (curandGenerateUniform(generator, dev_Map, totalSize))

	//PrintRandoms();

	/* Cleanup */
	CHECK_CURAND (curandDestroyGenerator(generator))
}

__global__ void SetupRandomGenerator (curandStateMRG32k3a* state, int n, int totalSize, int seed)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx * n > totalSize) return;

	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init (seed + idx, 0, 0, &state[idx]);
}

__global__ void GenerateRandomNumbers (float* map, curandStateMRG32k3a* state, int n, int totalSize)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	float x;
	/* Copy state to local memory for efficiency */
	curandStateMRG32k3a localState = state[idx];

	/* Generate pseudo-random uniforms */
	for (int i = 0; i < n; i++) {
		if (idx * n + i > totalSize) return;
		x = curand_uniform_double (&localState);
		map[idx * n + i] = x;
	}
	/* Copy state back to global memory */
	state[idx] = localState;
}

void DiamondSquareParallel::GenerateRandomNumbers_DeviceAPI ()
{
	int n = 128;

	CHECK (cudaMalloc((void **)&dev_MRGStates, (totalSize + n - 1) / n * sizeof(curandStateMRG32k3a)))

	dim3 blockSize (MAX_BLOCK_SIZE * MAX_BLOCK_SIZE, 1, 1);
	dim3 gridSize ((((totalSize + n - 1) / n) + (blockSize.x - 1)) / blockSize.x, 1, 1);
	SetupRandomGenerator<<<gridSize, blockSize>>> (dev_MRGStates, n, totalSize, RandomIntUniform());
	cudaDeviceSynchronize();
	GenerateRandomNumbers<<<gridSize, blockSize>>> (dev_Map, dev_MRGStates, n, totalSize);
	cudaDeviceSynchronize();

	/*CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
	PrintMap();*/
}

__device__ __forceinline__ float GetRandomOnDevice (float const value)
{
	bool cond = static_cast<int> (value * 10) & 0x01;
	return value * (-1) * cond + value * !cond;
}

#pragma endregion


#pragma region Initialization

void DiamondSquareParallel::AllocateMapOnDevice()
{
	CHECK (cudaMalloc ((void**)&dev_Map, totalSize * sizeof(float)))
	CHECK (cudaMalloc ((void**)&dev_Min, sizeof(float)))
	CHECK (cudaMalloc ((void**)&dev_Max, sizeof(float)))
}

void DiamondSquareParallel::InitializeDiamondSquare ()
{
	std::cout << "================== PARALLEL DIAMOND SQUARE ==================\n\n";
	std::cout << "---------- INITIALIZATION ----------\n";
	std::cout << "Initializing Diamond Square [" << size << " x " << size << "]...\n";

	MeasureTimeFn (nullptr, "Allocation time on device is ", this, &DiamondSquareParallel::AllocateMapOnDevice);

	threadAmount = (size - 1) / step;

#if !CURAND_DEVICE
	MeasureTimeFn (nullptr, "Random number set with host API generated in ", this, 
	  &DiamondSquareParallel::GenerateRandomNumbers_HostAPI);
#else
	MeasureTimeFn (nullptr, "Random number set with device API generated in ", this,
	               &DiamondSquareParallel::GenerateRandomNumbers_DeviceAPI);
#endif
}

#pragma endregion



#pragma region Execution

void DiamondSquareParallel::ComputeBlockGridSizes ()
{
	/*			  2^k			  or			  MAX_BLOCK_SIZE			  */
	blockSizeDiamond = threadAmount <= MAX_BLOCK_SIZE ? threadAmount : MAX_BLOCK_SIZE;
	/*		(2^k + 1) x 2^(k+1)	  or	SQUARE_BLOCK_X_SIZE x MAX_BLOCK_SIZE
	*		        k <= 3					     k > 3						  */
	blockXSizeSquare = threadAmount <= SQUARE_BLOCK_X_SIZE ? blockSizeDiamond + 1 : SQUARE_BLOCK_X_SIZE;
	blockYSizeSquare = threadAmount <= SQUARE_BLOCK_X_SIZE ? threadAmount * 2 : blockSizeDiamond;

	/*				  1			  or			2^k / MAX_BLOCK_SIZE		  */
	gridSizeDiamond = (threadAmount + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	/* SQUARE_BLOCK_X_SIZE x MAX_BLOCK_SIZE			block amount
	 * = (2^(k+1) / MAX_BLOCK_SIZE)  x	 (2^k / SQUARE_BLOCK_X_SIZE) + 1	  */
	gridSizeXSquare = threadAmount <= SQUARE_BLOCK_X_SIZE ? 1 : (threadAmount / SQUARE_BLOCK_X_SIZE) + 1;
	gridSizeYSquare = (threadAmount * 2 + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
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

		/* 2^k */
		threadAmount *= 2;
	}

#if COPY_RESULT_ON_HOST
	CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
#endif

#if CUDA_EVENTS_TIMING
	CHECK (cudaEventRecord( stop, 0 ))
	CHECK (cudaEventSynchronize( stop ))

	CHECK (cudaEventElapsedTime( &executionTimeCuda, start, stop ))
	CHECK (cudaEventDestroy( start ))
	CHECK (cudaEventDestroy( stop ))
#endif
}

__global__ void DiamondStepParallel (float* map, uint32_t size, uint32_t step, float randomScale, float* min, float* max)
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

	//map[GetIndexOnDeviceOnDevice (x, y, size)] = GetRandomOnDevice(map[GetIndexOnDeviceOnDevice (x, y, size)]) * randomScale + val;
	map[GetIndexOnDevice (x, y, size)] = val;

	/*AtomicMinFloat(min, val);
	AtomicMaxFloat(max, val);*/
}

__global__ void SquareStepParallel (float* map, uint32_t size, uint32_t step, float randomScale, float* min, float* max)
{
	uint32_t thd_X = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	uint32_t x = thd_X * step * (y % 2 == 0) +
		y * (step / 2) * (y % 2 != 0);
	y = (y * (step / 2) + (step / 2)) * (y % 2 == 0) +
		thd_X * step * (y % 2 != 0);

	if (x > size || y > size) {
		return;
	}

	float val = map[GetIndexOnDevice (x - (step / 2), y, size)] +
		map[GetIndexOnDevice (x + (step / 2), y, size)] +
		map[GetIndexOnDevice (x, y - (step / 2), size)] +
		map[GetIndexOnDevice (x, y + (step / 2), size)];

	val /= 4.0f;
	val += (-1.0f + map[GetIndexOnDevice (x, y, size)] * 2.0f) * randomScale;

	//map[GetIndexOnDeviceOnDevice (x, y, size)] = GetRandomOnDevice(map[GetIndexOnDeviceOnDevice (x, y, size)]) * randomScale + val;
	map[GetIndexOnDevice (x, y, size)] = val;

	/*AtomicMinFloat(min, val);
	AtomicMaxFloat(max, val);*/
}

void DiamondSquareParallel::DiamondStep ()
{
	dim3 blockDimension (blockSizeDiamond, blockSizeDiamond, 1);
	dim3 gridDimension (gridSizeDiamond, gridSizeDiamond, 1);

	DiamondStepParallel<<<gridDimension, blockDimension>>> (dev_Map, size, step, randomScale, dev_Min, dev_Max);
}

void DiamondSquareParallel::SquareStep ()
{
	dim3 blockDimension (blockXSizeSquare, blockYSizeSquare, 1);
	dim3 gridDimension (gridSizeXSquare, gridSizeYSquare, 1);

	SquareStepParallel<<<gridDimension, blockDimension>>> (dev_Map, size, step, randomScale, dev_Min, dev_Max);
}

#pragma endregion


#pragma region Values Mapping

__device__ __forceinline__ float AtomicMinFloat (float * addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

        return old;
}


__device__ __forceinline__ float AtomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}


__global__ void ComputeMinMax(float* map, float* myMin, float *myMax, int size) {
    
    __shared__ float blockMin[BLOCK_SIZE_1D];
    __shared__ float blockMax[BLOCK_SIZE_1D];

    int linearIdx = blockIdx.x * blockDim.x + threadIdx.x;

	bool cond = linearIdx >= size * size;
	blockMin[threadIdx.x] = cond ? FLT_MAX : blockMin[threadIdx.x];
    blockMax[threadIdx.x] = cond ? -FLT_MAX : blockMax[threadIdx.x];
    if (cond) {
        return;
    }
	
    blockMin[threadIdx.x] = map[linearIdx];
    blockMax[threadIdx.x] = map[linearIdx];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride)
            blockMin[threadIdx.x] = min(blockMin[threadIdx.x], blockMin[threadIdx.x + stride]);
        if (threadIdx.x < stride)
            blockMax[threadIdx.x] = max(blockMax[threadIdx.x], blockMax[threadIdx.x + stride]);
        __syncthreads();
    }

	if (threadIdx.x < 32) {
        volatile float *vMin = blockMin;
        volatile float *vMax = blockMax;
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 32]);
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 16]);
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 8]);
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 4]);
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 2]);
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 1]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 32]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 16]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 8]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 4]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 2]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 1]);
    }
	
    if (threadIdx.x == 0) AtomicMinFloat(myMin, blockMin[0]); 
    if (threadIdx.x == 0) AtomicMaxFloat(myMax, blockMax[0]); 
}

__global__ void MapFloatToIntRange(float* map, int* outputMap, float* min, float *max, int toMin, int toMax, int size) {
    
    int linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIdx >= size * size) return;
	
	int value = (int)((map[linearIdx] - *min) / (*max - *min) * (toMax - toMin) + toMin);
	
    outputMap[linearIdx] = value;
}

__global__ void MapFloatToGrayScale(float* map, uint8_t* grayScaleMap, float* min, float *max, int toMin, int toMax, int size) {
    
    int linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIdx >= size * size) return;
	
	int value = (int)((map[linearIdx] - *min) / (*max - *min) * (toMax - toMin) + toMin);
	
    grayScaleMap[linearIdx] = value;
}

void DiamondSquareParallel::MapValuesToGrayScale ()
{
	std::cout << "Mapping values to gray scale..." << std::endl;
	
	CHECK (cudaMalloc ((void**) &dev_GrayScaleMap, totalSize * sizeof(uint8_t)))

	int gridSize = (totalSize + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
	ComputeMinMax<<<gridSize, BLOCK_SIZE_1D>>> (dev_Map, dev_Min, dev_Max, size);
	CHECK (cudaMemcpy (&min, dev_Min, sizeof(float), cudaMemcpyDeviceToHost))
	CHECK (cudaMemcpy (&max, dev_Max, sizeof(float), cudaMemcpyDeviceToHost))
	
	cudaDeviceSynchronize();
	MapFloatToGrayScale<<<gridSize, BLOCK_SIZE_1D>>> (dev_Map, dev_GrayScaleMap, dev_Min, dev_Max, 0, 255, size);

	grayScaleMap = new uint8_t[totalSize]{0};
	CHECK (cudaMemcpy (grayScaleMap, dev_GrayScaleMap, totalSize * sizeof(uint8_t), cudaMemcpyDeviceToHost))
}

void DiamondSquareParallel::MapValuesToIntRange (int toMin, int toMax)
{
	std::cout << "\n---------- VALUES MAPPING ----------" << std::endl;
	std::cout << "Mapping values..." << std::endl;
	
	CHECK (cudaMalloc ((void**) &dev_IntMap, totalSize * sizeof(int)))

	int gridSize = (totalSize + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
	ComputeMinMax<<<gridSize, BLOCK_SIZE_1D>>> (dev_Map, dev_Min, dev_Max, size);
	CHECK (cudaMemcpy (&min, dev_Min, sizeof(float), cudaMemcpyDeviceToHost))
	CHECK (cudaMemcpy (&max, dev_Max, sizeof(float), cudaMemcpyDeviceToHost))
	
	cudaDeviceSynchronize();
	MapFloatToIntRange<<<gridSize, BLOCK_SIZE_1D>>> (dev_Map, dev_IntMap, dev_Min, dev_Max, toMin, toMax, size);

	intMap = new int[totalSize]{0};
	CHECK (cudaMemcpy (intMap, dev_IntMap, totalSize * sizeof(int), cudaMemcpyDeviceToHost))
}

#pragma endregion 

void DiamondSquareParallel::CleanUp ()
{
	CHECK (cudaFree(dev_Map))
	CHECK (cudaFree(dev_IntMap))
	CHECK (cudaFree(dev_Min))
	CHECK (cudaFree(dev_Max))
#if CURAND_DEVICE
	CHECK (cudaFree(dev_MRGStates))
#endif
}
