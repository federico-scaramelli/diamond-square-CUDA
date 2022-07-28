#include "DiamondSquareSequential.h"

std::random_device rd;
//std::mt19937 gen(0);
std::mt19937 generator(rd());
std::uniform_real_distribution<double> unif {-1.0, 1.0};

void DiamondSquareSequential::InitializeDiamondSquare(uint32_t initValuesDistance) {
	/*for (int i = 0; i < 100; i++) {
		std::cout << unif(generator) << std::endl;
	}*/
	for (int x = 0; x < size; x += initValuesDistance) {
		if (x > size) continue;
		for (int y = 0; y < size; y += initValuesDistance) {
			if (y > size) continue;
			map[x * size + y] = unif(generator);  
		}
	}

	step = initValuesDistance;
}

void DiamondSquareSequential::DiamondSquare() {

	//step = size - 1;

	while (step > 1) {
		half = step / 2;

		for (uint32_t y = 0; y < size - 1; y += step) {
			for (uint32_t x = 0; x < size - 1; x += step) {
				DiamondStep(x, y);
			}
		}

		//PrintMap();

		for (uint32_t x = 0; x < size; x += half) {
			for (uint32_t y = (x + half) % step; y < size; y += step) {
				SquareStep(x, y);
				//map[x * size + y] = 255;
			}
		}
		
		//PrintMap();

		randomScale /= 2.0;
		step /= 2;
	}
}

void DiamondSquareSequential::DiamondStep(uint32_t x, uint32_t y) {

	double value = 0;

	value = map[x * size + y] +
		map[(x + step) * size + y] +
		map[x * size + y + step] +
		map[(x + step) * size + y + step];

	value /= 4.0;
	value += unif(generator) * randomScale;
	value = value > 1 ? 1 : value;
	value = value < -1 ? -1 : value;

	map[(y + half) * size + (x + half)] = value;
}

void DiamondSquareSequential::SquareStep(uint32_t x, uint32_t y) {

	double value = 0;
	uint8_t count = 0;
	
	if (y + half < size) {
		//std::cout  << "y + half: " << static_cast<int>(map[x * size + y + half]) << "\n";
		value += map[x * size + y + half];
		count++;
	} else
		value += map[x * size + size - 1];

	if (static_cast<int>(y - half) >= 0) {
		//std::cout << "y - half: " << static_cast<int>(map[x * size + y - half]) << "\n";
		value += map[x * size + y - half];
		count++;
	} else 
		value += map[x * size];

	if (x + half < size) {
		//std::cout  << "x + half: " << static_cast<int>(map[(x + half) * size + y]) << "\n";
		value += map[(x + half) * size + y];
		count++;
	}
	else 
		value += map[(size - 1) * size + y];

	if (static_cast<int>(x - half) >= 0) {
		//std::cout  << "x - half: " << static_cast<int>(map[(x - half) * size + y]) << "\n";
		value += map[(x - half) * size + y];
		count++;
	} else
		value += map[y];

	value /= 4.0;
	value += unif(generator) * randomScale;
	value = value > 1 ? 1 : value;
	value = value < -1 ? -1 : value;

	map[x * size + y] = value;
}
