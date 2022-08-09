#include "diamondSquareParallel.h"

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
	CHECK (cudaMalloc((void **)&dev_Randoms, totalSize * sizeof(float)))

	/* Generate n floats on device */
	CHECK_CURAND (curandGenerateUniform(generator, dev_Randoms, totalSize))

	//PrintRandoms();

	/* Cleanup */
	CHECK_CURAND (curandDestroyGenerator(generator))
}

/* TODO: change it to void */
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

__device__ __forceinline__ uint32_t GetIndex (uint32_t x, uint32_t y, uint32_t size)
{
	x = x >= size ? size - 1 : x;
	y = y >= size ? size - 1 : y;

	return x * size + y;
}


__global__ void InitializeDiamondSquareParallel (float* map, float* randoms, uint32_t size, uint32_t step)
{
	uint32_t thd_X = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t thd_Y = blockIdx.x * blockDim.x + threadIdx.x;
	thd_X *= step;
	thd_Y *= step;

	map[GetIndex (thd_X, thd_Y, size)] = randoms[GetIndex (thd_X, thd_Y, size)]; 
}

void DiamondSquareParallel::InitializeDiamondSquare ()
{
	MeasureTimeFn ("Parallel random numbers generation time: ", this, &DiamondSquareParallel::GenerateRandomNumbers);
	MeasureTimeFn ("Copy initial map to the device time: ", this, &DiamondSquareParallel::CopyMapToDevice);
	
	threadAmount = (size - 1) / step;
	uint32_t blockSize = threadAmount <= 16 ? threadAmount : 16;
	uint32_t gridSize = threadAmount / 16;
	dim3 blockDimension (blockSize, blockSize, 1);
	//std::cout << "Diamond step block size (" << blockSizeDiamond << ", " << blockSizeDiamond << ");\n";
	dim3 gridDimension (gridSize, gridSize, 1);
	std::cout << "Thread amount: " << threadAmount << ", blockSize: " << blockSize << ", gridSize: " << gridSize;
	//std::cout << "Diamond step grid size (" << gridSizeDiamond << ", " << gridSizeDiamond << ");\n";
	InitializeDiamondSquareParallel<<<gridDimension, blockDimension>>> (dev_Map, dev_Randoms, size, step);

	/* For now initialize on the CPU side
	 * TODO: initialize values on the GPU */
	/*for (uint32_t x = 0; x < size; x += step) {
		for (uint32_t y = 0; y < size; y += step) {
			map[GetIndex (x, y)] = RandomFloatUniform();
		}
	}
	MeasureTimeFn ("Copy initial map to the device time: ", this, &DiamondSquareParallel::CopyMapToDevice);
	*/

}

void DiamondSquareParallel::CopyMapToDevice ()
{
	/* Copy the map on the device memory */
	CHECK (cudaMalloc(&dev_Map, totalSize * sizeof(float)))
	CHECK (cudaMemcpy(dev_Map, map, totalSize * sizeof(float), cudaMemcpyHostToDevice))
}

void DiamondSquareParallel::CalculateBlockGridSizes ()
{
	/*			  2^k			  or			  16			  */
	blockSizeDiamond = threadAmount <= 16 ? threadAmount : 16;
	/*		(2^k + 1) x 2^(k+1)	  or			 9 x 16
	*		        k <= 3					     k > 3			  */
	blockXSizeSquare = threadAmount <= 8 ? blockSizeDiamond + 1 : 9;
	/*		(2^k + 1) x 2^(k+1)	  or			 9 x 16
	*			   k <= 3						 k > 3			  */
	blockYSizeSquare = threadAmount <= 8 ? threadAmount * 2 : blockSizeDiamond;

	/*				  1			  or			2^k / 16		  */
	gridSizeDiamond = threadAmount <= 8 ? 1 : threadAmount / 16;
	/* 9 x 16 block amount =  (2^k / 16)  /	  ceil(2^k / 9)		  */
	gridSizeSquare = threadAmount < 16 ? 1 : (threadAmount + 9) / 9;
}

void DiamondSquareParallel::DiamondSquare ()
{
	while (step > 1) {
		CalculateBlockGridSizes();

		DiamondStep();
		cudaDeviceSynchronize();
		//CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
		//PrintMap();

		SquareStep();
		cudaDeviceSynchronize();
		//CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))
		//PrintMap();

		randomScale /= 2.0f;
		step /= 2;

		/* 2^k */
		threadAmount *= 2;
	}

	CHECK (cudaMemcpy(map, dev_Map, totalSize * sizeof(float), cudaMemcpyDeviceToHost))

	CleanUp();
}

__global__ void DiamondStepParallel (float* map, float* randoms, uint32_t size, uint32_t step, float randomScale)
{
	uint32_t half = step / 2;

	uint32_t thd_X = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t thd_Y = blockIdx.x * blockDim.x + threadIdx.x;
	thd_X = thd_X * step + half;
	thd_Y = thd_Y * step + half;

	float val = map[GetIndex (thd_X - half, thd_Y - half, size)] +
		map[GetIndex (thd_X + half, thd_Y - half, size)] +
		map[GetIndex (thd_X - half, thd_Y + half, size)] +
		map[GetIndex (thd_X + half, thd_Y + half, size)];

	val /= 4.0f;
	val += randomScale * getRandomOnDevice(randoms[GetIndex (thd_X, thd_Y, size)]);

	map[GetIndex (thd_X, thd_Y, size)] = val;
}

void DiamondSquareParallel::DiamondStep ()
{
	dim3 blockDimension (blockSizeDiamond, blockSizeDiamond, 1);
	//std::cout << "Diamond step block size (" << blockSizeDiamond << ", " << blockSizeDiamond << ");\n";
	dim3 gridDimension (gridSizeDiamond, gridSizeDiamond, 1);
	//std::cout << "Diamond step grid size (" << gridSizeDiamond << ", " << gridSizeDiamond << ");\n";
	DiamondStepParallel<<<gridDimension, blockDimension>>> (dev_Map, dev_Randoms, size, step, randomScale);
}

__global__ void SquareStepParallel (float* map, float* randoms, uint32_t size, uint32_t step, float randomScale)
{
	uint32_t half = step / 2;

	uint32_t thd_X = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t thd_Y = blockIdx.y * blockDim.y + threadIdx.y;

	uint32_t x = thd_X * step * (thd_Y % 2 == 0) +
		thd_Y * half * (thd_Y % 2 != 0);
	uint32_t y = (thd_Y * half + half) * (thd_Y % 2 == 0) +
		thd_X * step * (thd_Y % 2 != 0);

	if (x > size || y > size) {
		return;
	}
	
	float val = map[GetIndex (x - half, y, size)] +
		map[GetIndex (x + half, y, size)] +
		map[GetIndex (x, y - half, size)] +
		map[GetIndex (x, y + half, size)];

	val /= 4.0f;
	val += randomScale * getRandomOnDevice(randoms[GetIndex (x, y, size)]);

	map[GetIndex (x, y, size)] = val;
}

void DiamondSquareParallel::SquareStep ()
{
	dim3 blockDimension (blockXSizeSquare, blockYSizeSquare, 1);
	//std::cout << "Square step block size (" << blockXSizeSquare << ", " << blockYSizeSquare << ");\n";
	dim3 gridDimension (gridSizeSquare, threadAmount * 2 / 16, 1);
	//std::cout << "Square step grid size (" << gridSizeSquare << ", " << gridSizeSquare<< ");\n";
	SquareStepParallel<<<gridDimension, blockDimension>>> (dev_Map, dev_Randoms, size, step, randomScale);
}

void DiamondSquareParallel::CleanUp ()
{
	CHECK (cudaFree(dev_Randoms))
	CHECK (cudaFree(dev_Map))
}

