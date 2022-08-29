#pragma once
#include <cstdint>
#include "../../utils/TimeMeasure.h"

// Initial setting for a single algorithm execution
struct DiamondSquareSetting
{
public:
	uint32_t size;
	uint32_t initialStepSize;
	float randomScale;
	uint8_t imageTileSize;
};

// Data structure to save performance results related to a single algorithm execution
struct DiamondSquareResult
{
public:
	DiamondSquareSetting setting;
	double totalSeq_Time;
	double totalPar_Time;
	double totalParConst_Time;
	double seqGrayscale_Time;
	double parGrayscale_Time;
	double seqCustomRangeMap_Time;
	double parCustomRangeMap_Time;

	void PrintSpeedUps()
	{
		std::cout << "\nTEST - Size [ " << setting.size << " ] \n";
		CompareTime ("Overall execution parallel speed-up is ",
		             &totalSeq_Time, &totalPar_Time);
		CompareTime ("Grayscale map conversion parallel speed-up is ",
		             &seqGrayscale_Time, &parGrayscale_Time);
	}
};

// Enum to easily recognize settings from Application
enum DiamondSquareSettings
{
	Size9_Step8_Rnd1,
	Size17_Step16_Rnd1,
	Size33_Step32_Rnd1,
	Size65_Step64_Rnd1,
	Size129_Step128_Rnd2,
	Size257_Step256_Rnd3,
	Size513_Step256_Rnd5,
	Size1025_Step512_Rnd10,
	Size2049_Step1024_Rnd10,
	Size4097_Step4096_Rnd15,
	Size4097_Step1024_Rnd15,
	Size4097_Step2048_Rnd15,
	Size8193_Step8192_Rnd15,
	Size8193_Step2048_Rnd15,
	Size8193_Step2048_Rnd20,
	Size16385_Step4096_Rnd30,
	Size16385_Step512_Rnd50
};

DiamondSquareSetting diamondSquareSettings[] = {
	{ 9, 8, 1.0, 1 }, 
	{ 17, 16, 1.0, 1 },
	{ 33, 32, 1.0, 1 },
	{ 65, 64, 1.0, 1 },
	{ 129, 128, 2.0, 1 },
	{ 257, 256, 3.0, 1 },
	{ 513, 256, 5.0, 1 },
	{ 1025, 512, 10.0, 1 },
	{ 2049, 1024, 10.0, 1 },
	{ 4097, 4096, 15.0, 1 },
	{ 4097, 1024, 15.0, 1 },
	{ 4097, 2048, 15.0, 1 },
	{ 8193, 8192, 15.0, 1 },
	{ 8193, 2048, 15.0, 1 },
	{ 8193, 2048, 20.0, 1 },
	{ 16385, 4096, 30.0, 1 },
	{ 16385, 512, 50.0, 1 }
};


DiamondSquareSetting testingDiamondSquareSettings[] = {
	{ 0, 0, 0, 0 },
	{ 9, 8, 1.0, 1 },			//1 
	{ 17, 16, 1.0, 1 },		//2
	{ 33, 32, 1.0, 1 },		//3
	{ 65, 64, 1.0, 1 },		//4
	{ 129, 128, 1.0, 1 },		//5
	{ 257, 256, 1.0, 1 },		//6
	{ 513, 512, 1.0, 1 },		//7
	{ 1025, 1024, 1.0, 1 },	//8	
	{ 2049, 2048, 1.0, 1 },	//9
	{ 4097, 1024, 10.0, 1 },	//10
	{ 8193, 8192, 1.0, 1 },	//11
	{ 16385, 16384, 1.0, 1 }   //12
};
