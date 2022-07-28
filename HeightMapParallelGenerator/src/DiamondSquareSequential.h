#pragma once
#include "diamondSquareBase.h"

class DiamondSquareSequential : public DiamondSquareBase {
public:
	DiamondSquareSequential(uint32_t size) : DiamondSquareBase(size) {}

	void InitializeDiamondSquare(uint32_t initValuesDistance) override;

	void DiamondSquare() override;

	void DiamondStep(uint32_t x, uint32_t y) override;
	
	void SquareStep(uint32_t x, uint32_t y) override;
};