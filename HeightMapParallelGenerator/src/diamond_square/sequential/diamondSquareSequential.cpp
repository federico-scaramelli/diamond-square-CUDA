#include "diamondSquareSequential.h"
#include "../parameters/applicationSettings.h"

#pragma region Execution Functions

void DiamondSquareSequential::InitializeDiamondSquare()
{
	std::cout << "================== SEQUENTIAL DIAMOND SQUARE ==================" << std::endl << std::endl;
  	std::cout << "---------- INITIALIZATION ----------" << std::endl;
	std::cout << "Initializing Diamond Square [" << size << " x " << size << "]..." << std::endl;
	
	for (uint32_t x = 0; x < size; x += step) {
		for (uint32_t y = 0; y < size; y += step) {
			map[GetIndexOnHost(x, y)] = RandomFloatUniform();
		}
	}
}

void DiamondSquareSequential::DiamondSquare()
{
	while (step > 1) 
	{
		half = step / 2;

		DiamondStep();

#if PRINT_DIAMOND_STEP_SEQ
		PrintMap();
#endif

		SquareStep();

#if PRINT_SQUARE_STEP_SEQ
		PrintMap();
#endif

		randomScale /= 2.f;
		step /= 2;
	}
}

void DiamondSquareSequential::DiamondStep()
{
	for (uint32_t x = half; x < size; x += step) {
		for (uint32_t y = half; y < size; y += step) {
			float value = 0;

			value = map[GetIndexOnHost(x - half, y - half)] +
				map[GetIndexOnHost(x + half, y - half)] +
				map[GetIndexOnHost(x - half, y + half)] +
				map[GetIndexOnHost(x + half, y + half)];

			value /= 4.0f;
			value += RandomFloatUniform() * randomScale;

			map[GetIndexOnHost(x, y)] = value;
		}
	}
}

void DiamondSquareSequential::SquareStep()
{
	for (int x = 0; x < size; x += half) {
		for (int y = (x + half) % step; y < size; y += step) {

			int a = 0;
			float value = 0;
			int count = 0;
			a = (x - half) * size + y;

			if (a < totalSize && a >= 0) {
				value += map[(x - half) * size + y];
				count++;
			}
			a = (x + half) * size + y;
			if (a < totalSize && a >= 0) {
				value += map[(x + half) * size + y];
				count++;
			}
			a = x * size + y - half;
			if (a < totalSize && a >= 0 && (int)(y - half) >= 0) {
				value += map[x * size + y - half];
				count++;
			}
			a = x * size + y + half ;
			if (a < totalSize && a >= 0 && y + half < size) {
				value += map[x * size + y + half];
				count++;
			}

			/*float value = map[GetIndexOnHost(x - half, y)] +
				map[GetIndexOnHost(x + half, y)] +
				map[GetIndexOnHost(x, y - half)] +
				map[GetIndexOnHost(x, y + half)];*/

			/*float value = GetMapValueOnHost(x - half, y) +
				GetMapValueOnHost(x + half, y) +
				GetMapValueOnHost(x, y - half) +
				GetMapValueOnHost(x, y + half);*/

			//value /= 4.0f;
			value /= count;
			value += RandomFloatUniform() * randomScale;

			map[x * size + y] = value;
			//std::cout << "SQUARE CAMBIA ELEMENTO: (" << x << ", " << y << ")" << std::endl;
		}
	}
}

#pragma endregion
