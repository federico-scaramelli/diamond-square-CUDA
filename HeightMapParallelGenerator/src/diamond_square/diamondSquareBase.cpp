#include "diamondSquareBase.h"
#include "parameters/applicationSettings.h"
#include "./cuda/diamondSquareParallel.h"

#pragma region Constructors

// Initializes data shared between all the versions of the algorithm
DiamondSquareBase::DiamondSquareBase (const uint32_t size)
{
	this->size = size;
	CheckSizeAdequate();
	this->totalSize = size * size;
	this->step = size - 1;
	this->map = new float[totalSize];
	memset (map, 0, sizeof (float) * totalSize);
	half = 0;

	executionTime = 0;
}

// Free memory
DiamondSquareBase::~DiamondSquareBase ()
{
	delete[] map;
	delete[] grayScaleMap;
	delete[] intMap;
}

// Explicitly delete the float map from memory
void DiamondSquareBase::DeleteFloatMap ()
{
	delete[] map;
	map = nullptr;
}

#pragma endregion


#pragma region Support Functions

// Check if the size is adequate for the algorithm
void DiamondSquareBase::CheckSizeAdequate ()
{
	if ((size - 1 & size - 2) != 0)
	{
		throw std::exception ("Size not adequate. The map size must be a power of two plus one.");
	}
}

// Get the linearized index
uint32_t DiamondSquareBase::GetIndexOnHost (uint32_t x, uint32_t y) const
{
	if (x >= size) x = size - 1;
	if (y >= size) y = size - 1;

	return x * size + y;
}

// Print maps methods
void DiamondSquareBase::PrintFloatMap () const
{
	for (uint32_t i = 0; i < size; ++i)
	{
		for (uint32_t j = 0; j < size; ++j) { std::cout << map[i * size + j] << ' '; }
		std::cout << std::endl;
	}
	std::cout << std::endl << std::endl;
}

void DiamondSquareBase::PrintGrayScaleMap ()
{
	if (grayScaleMap == nullptr)
		GenerateGrayScaleMap();

	for (uint32_t i = 0; i < size; ++i)
	{
		for (uint32_t j = 0; j < size; ++j) { std::cout << static_cast<int> (grayScaleMap[i * size + j]) << ' '; }
		std::cout << std::endl;
	}
	std::cout << std::endl << std::endl;
}

void DiamondSquareBase::PrintIntMap ()
{
	if (intMap == nullptr)
		throw std::runtime_error("Tried to print the int map but it has not been generated!");

	for (uint32_t i = 0; i < size; ++i)
	{
		for (uint32_t j = 0; j < size; ++j) { std::cout << intMap[i * size + j] << ' '; }
		std::cout << std::endl;
	}
	std::cout << std::endl << std::endl;
}

#pragma endregion


#pragma region Setter Functions

void DiamondSquareBase::SetRandomScale (float randomScale) { this->randomScale = randomScale; }

void DiamondSquareBase::SetInitialStepSize (uint32_t initValuesDistance)
{
	if (initValuesDistance == 0)
		throw std::runtime_error ("Init values distance is 0!");
	if (initValuesDistance > size - 1)
		throw std::runtime_error ("Init values distance is too big!");
	if ((size - 1) % initValuesDistance != 0)
		throw std::runtime_error ("Init values distance is not a multiple of the size!");

	step = initValuesDistance;
}

#pragma endregion


#pragma region Image Functions

// Mapping algorithms CPU side
void DiamondSquareBase::MapValuesToGrayScale ()
{
	std::cout << "\n - VALUES MAPPING - " << std::endl;
	std::cout << "Mapping values to grayscale..." << std::endl;


	delete[] grayScaleMap; //Safe
	grayScaleMap = new uint8_t[totalSize]{ 0 };

	auto minmax = std::minmax_element (map, map + totalSize);

	for (uint32_t i = 0; i < size; ++i)
	{
		for (uint32_t j = 0; j < size; ++j)
		{
			grayScaleMap[i * size + j] = static_cast<uint8_t> (MapValue
				(*minmax.first, *minmax.second,
				 0, 255, map[i * size + j]));
		}
	}

	if (DELETE_FLOAT_MAP)
		DeleteFloatMap();
}

void DiamondSquareBase::MapValuesToIntRange (int toMin, int toMax)
{
	std::cout << "\n - VALUES MAPPING - " << std::endl;
	std::cout << "Mapping values to int range..." << std::endl;

	delete[] intMap; //Safe
	intMap = new int[totalSize]{ 0 };

	auto minmax = std::minmax_element (map, map + totalSize);
	
	for (uint32_t i = 0; i < size; ++i)
	{
		for (uint32_t j = 0; j < size; ++j)
		{
			intMap[i * size + j] = (MapValue
				(*minmax.first, *minmax.second,
				 toMin, toMax, map[i * size + j]));
		}
	}
}

void DiamondSquareBase::GenerateGrayScaleMap ()
{
	std::cout << "\n\n - IMAGE GENERATION - " << std::endl;
	std::cout << "Mapping values to grayscale..." << std::endl;

	MeasureTimeFn (nullptr, "Grayscale map generated in ", this,
		               &DiamondSquareBase::MapValuesToGrayScale);
}

// Save image functions
void DiamondSquareBase::SaveGrayScaleImage (const char* fname, int tileSize)
{
	if (grayScaleMap == nullptr)
	{
		GenerateGrayScaleMap();
	}

	std::cout << "Creating grayscale image..." << std::endl;

	BMP image (size * tileSize, size * tileSize, true);

	for (uint32_t i = 0; i < size; ++i)
	{
		for (uint32_t j = 0; j < size; ++j)
		{
			image.FillRegion (j * tileSize, i * tileSize, tileSize, grayScaleMap[i * size + j]);
		}
	}
	image.Write (fname);
}

void DiamondSquareBase::SaveColorImage (const char* fname, int tileSize)
{
	if (grayScaleMap == nullptr)
	{
		GenerateGrayScaleMap();
	}

	ColorMapping::CacheColorsFromMapping();

	std::cout << "Creating color image..." << std::endl;

	BMP image (size * tileSize, size * tileSize, true);
	ColorPixel color;

	for (uint32_t i = 0; i < size; ++i)
	{
		for (uint32_t j = 0; j < size; ++j)
		{
			ColorMapping::GetColorLerp (grayScaleMap[i * size + j], &color);
			image.FillRegion (j * tileSize, i * tileSize, tileSize, color);
		}
	}
	image.Write (fname);
}

#pragma endregion

// Main method to start the algorithm execution (initialization + execution)
void DiamondSquareBase::ExecuteDiamondSquare ()
{
	MeasureTimeFn (&initializationTime, "Map initialized in ", this, &DiamondSquareBase::InitializeDiamondSquare);

	std::cout << "\n - EXECUTION - " << std::endl;
	std::cout << "Executing diamond square..." << std::endl;

	MeasureTimeFn (&executionTime, "Algorithm terminated in ", this, &DiamondSquareBase::DiamondSquare);

#if EVENTS_TIMING
		if (dynamic_cast<DiamondSquareParallel*>(this) != nullptr) {
			std::cout << std::endl;
			std::cout << "Diamond Square parallel time measured with CUDA Events is " << 
			  *(dynamic_cast<DiamondSquareParallel*>(this)->GetExecutionTimeCuda()) << std::endl;
			std::cout << "and the difference from time measured with CPU is equal to " << 
			  *GetExecutionTime() - *(dynamic_cast<DiamondSquareParallel*>(this)->GetExecutionTimeCuda()) << std::endl;
		}
#endif

	std::cout << "\n -- TOTAL EXECUTION TIME -- " << std::endl;
}
