#pragma once
#include <random>
#include "diamondSquareBase.h"

class DiamondSquareSequential : public DiamondSquareBase {
public:
	//Constructor
	DiamondSquareSequential(uint32_t size) : DiamondSquareBase(size) {}

#pragma region Execution Functions

	void InitializeDiamondSquare() override;

	void DiamondSquare() override;

	void DiamondStep(uint32_t x, uint32_t y) override;
	
	void SquareStep(uint32_t x, uint32_t y) override;

#pragma endregion
};