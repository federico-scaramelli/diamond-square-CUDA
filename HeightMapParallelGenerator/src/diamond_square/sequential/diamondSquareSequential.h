#pragma once
#include <random>
#include "../diamondSquareBase.h"

class DiamondSquareSequential : public DiamondSquareBase {
public:
	//Constructor
	DiamondSquareSequential(uint32_t size) : DiamondSquareBase(size) {}

#pragma region Execution Functions

	void InitializeDiamondSquare() override;

	void DiamondSquare() override;

	void DiamondStep() override;
	
	void SquareStep() override;

#pragma endregion
};