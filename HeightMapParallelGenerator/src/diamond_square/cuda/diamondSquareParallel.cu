#include "diamondSquareParallel.h"

void DiamondSquareParallel::InitializeDiamondSquare() {
	//If less than epsilon do it on the CPU
	//otherwise invoke the kernel InitializeDiamondSquareParallel
}

void DiamondSquareParallel::DiamondSquare() {
	while (step > 1) {
		half = step / 2;

		//DiamondStep();

		//PrintMap();

		//SquareStep();

		//PrintMap();

		randomScale /= 2.0;
		step /= 2;
	}
}

__global__ void DiamondSquareParallel::InitializeDiamondSquareParallel() {
	
}


__global__ void DiamondSquareParallel::DiamondStep() {
	
}

__global__ void DiamondSquareParallel::SquareStep() {
	
}