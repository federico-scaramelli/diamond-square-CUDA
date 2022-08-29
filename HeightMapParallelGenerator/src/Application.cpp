#include <bitset>
#include <cuda_runtime_api.h>

#include "./image/bmpHandler.h"
#include "./diamond_square/parameters/algorithmSettings.h"
#include "./diamond_square/parameters/applicationSettings.h"
#include "./diamond_square/sequential/diamondSquareSequential.h"
#include "./diamond_square/cuda/diamondSquareParallel.h"
#include "./diamond_square/cuda/DiamondSquareParallel_Constant.h"


// Setting to use if testing setting flag is not enabled
DiamondSquareSettings setting = Size16385_Step4096_Rnd30;


// Set the size variable
#if TESTING_SETTINGS
uint32_t size = testingDiamondSquareSettings[TESTING_SETTINGS].size;
#else
uint32_t size = diamondSquareSettings[setting].size;
#endif

// Instantiate an object for each version of the algorithms 
DiamondSquareParallel parDiamSquare{ size };
DiamondSquareSequential seqDiamSquare{ size };
DiamondSquareParallel_Constant parDiamSquareConstMem{ size };

// Variables to store total times of the algorithm execution (mapping not included)
double sequentialTime;
double parallelTime;
double parallelTime_const;

// Variables to store times of the mapping algorithm execution 
double seqTimeGrayscale;
double parTimeGrayscale;
double seqTimeCustomRangeMap;
double parTimeCustomRangeMap;

// Compare performance on different sizes defined by the testing settings array
void runComparator ()
{
	int settingsCount = sizeof(testingDiamondSquareSettings) / sizeof (DiamondSquareSetting);
	int startingIdx = 6;
	//settingsCount = 7;
	auto results = new DiamondSquareResult[settingsCount];

	for (auto i = startingIdx; i < settingsCount; i++)
	{
		DiamondSquareResult& currentResult = results[i];
		currentResult.setting = testingDiamondSquareSettings[i];
		uint32_t currentSize = testingDiamondSquareSettings[i].size;

		DiamondSquareParallel parallel{ currentSize };
		DiamondSquareSequential sequential{ currentSize };
		DiamondSquareParallel_Constant constantParallel{ currentSize };

		std::cout << R"( /\/\/\/\/\/\/\/\/\ TEST )" << i << " - Size [" << currentSize << R"(] /\/\/\/\/\/\/\/\/\)" << "\n";
		MeasureTimeFn (&currentResult.totalPar_Time, "Algorithm execution terminated in ",
		               &parallel, &DiamondSquareBase::ExecuteDiamondSquare);
		MeasureTimeFn (&currentResult.parCustomRangeMap_Time, "Custom range map generated in ",
		               &parallel, &DiamondSquareParallel::MapValuesToIntRange,
		               -1000, 2000);
		MeasureTimeFn (&currentResult.parGrayscale_Time, "Grayscale map generated in ",
		               &parallel, &DiamondSquareParallel::MapValuesToGrayScale);

		std::cout << "\n\n";
		MeasureTimeFn (&currentResult.totalParConst_Time, "Algorithm execution terminated in ",
		               &constantParallel, &DiamondSquareBase::ExecuteDiamondSquare);

		std::cout << "\n\n";
		MeasureTimeFn (&currentResult.totalSeq_Time, "Algorithm execution terminated in ",
		               &sequential, &DiamondSquareBase::ExecuteDiamondSquare);
		MeasureTimeFn (&currentResult.seqCustomRangeMap_Time, "Custom range map generated in ",
		               &sequential, &DiamondSquareBase::MapValuesToIntRange,
		               -1000, 2000);
		MeasureTimeFn (&currentResult.seqGrayscale_Time, "Grayscale map generated in ",
		               &sequential, &DiamondSquareBase::MapValuesToGrayScale);

		std::cout << "\n\n" << R"( /\/\/\/\/\/\/\/\/\ SPEED-UP TEST )" << i << " - Size [" << currentSize << R"(] /\/\/\/\/\/\/\/\/\)" << "\n";
		
		CompareTime ("Constant memory usage algorithm speed-up is ", constantParallel.GetExecutionTime(),
		             parallel.GetExecutionTime());
		CompareTime ("Constant memory usage overall execution speed-up is ",
		             &currentResult.totalParConst_Time, &currentResult.totalPar_Time);
		CompareTime ("Diamond Square initialization parallel speed-up is ",
		             sequential.GetInitializationTime(), parallel.GetInitializationTime());
		CompareTime ("Diamond Square algorithm parallel speed-up is ",
		             sequential.GetExecutionTime(), parallel.GetExecutionTime());
		CompareTime ("Overall execution parallel speed-up is ",
		             &currentResult.totalSeq_Time, &currentResult.totalPar_Time);
		CompareTime ("Grayscale map conversion parallel speed-up is ",
		             &currentResult.seqGrayscale_Time, &currentResult.parGrayscale_Time);
		CompareTime ("Custom range map conversion parallel speed-up is ",
		             &currentResult.seqCustomRangeMap_Time, &currentResult.parCustomRangeMap_Time);
		std::cout << "\n\n" << R"(/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\)" << "\n\n\n\n";
	}
	std::cout << "\n\n" << R"(/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\)" << "\n\n\n\n";
	for (auto i = startingIdx; i < settingsCount; i++)
	{
		results[i].PrintSpeedUps();
	}
}

// Execute the sequential version of the algorithm
void runSequential ()
{
#if !TESTING_SETTINGS
	seqDiamSquare.SetRandomScale (diamondSquareSettings[setting].randomScale);
	//seqDiamSquare.SetRandomScale(100);
	seqDiamSquare.SetInitialStepSize (diamondSquareSettings[setting].initialStepSize);
	//seqDiamSquare.SetInitialStepSize(32);
#else
	seqDiamSquare.SetRandomScale(testingDiamondSquareSettings[TESTING_SETTINGS].randomScale);
	seqDiamSquare.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

	// Execute the algorithm
	MeasureTimeFn (&sequentialTime, "Sequential algorithm execution terminated in ",
	               &seqDiamSquare, &DiamondSquareBase::ExecuteDiamondSquare);

	// Debug purposes
#if PRINT_GRAYSCALE_SEQ
	ds.PrintGrayScaleMap();
#endif

	// Map the output values
	MeasureTimeFn (&seqTimeCustomRangeMap, "Custom range map generated in ",
	               &seqDiamSquare, &DiamondSquareBase::MapValuesToIntRange,
	               -1000, 2000);
	MeasureTimeFn (&seqTimeGrayscale, "Grayscale map generated in ",
	               &seqDiamSquare, &DiamondSquareBase::MapValuesToGrayScale);

	// Image saving
#if SAVE_GRAYSCALE_IMAGE
	std::cout << "\n - GRAYSCALE IMAGE - " << std::endl;
	MeasureTimeFn(nullptr, "Grayscale image generated and saved on file in ",
				  &seqDiamSquare, &DiamondSquareBase::SaveGrayScaleImage,
	              GRAYSCALE_SEQ_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
	std::cout << "\n - COLOR IMAGE - " << std::endl;
	MeasureTimeFn (nullptr, "Color image generated and saved on file in ",
	               &seqDiamSquare, &DiamondSquareBase::SaveColorImage,
	               COLOR_SEQ_PATH, 1);
#endif
}

// Execute the parallel version of the algorithm
void runParallel ()
{
#if !TESTING_SETTINGS
	parDiamSquare.SetRandomScale (diamondSquareSettings[setting].randomScale);
	// Optionally it's possible to change the initial random scale
	parDiamSquare.SetRandomScale(100);
	parDiamSquare.SetInitialStepSize (diamondSquareSettings[setting].initialStepSize);
	// Optionally it's possible to change the initial step size
	parDiamSquare.SetInitialStepSize(256);
#else
	parDiamSquare.SetRandomScale(testingDiamondSquareSettings[TESTING_SETTINGS].randomScale);
	parDiamSquare.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

	// Execute the algorithm
	MeasureTimeFn (&parallelTime, "Parallel algorithm execution terminated in ",
	               &parDiamSquare, &DiamondSquareBase::ExecuteDiamondSquare);

	// Debug purposes
#if PRINT_GRAYSCALE_CUDA
	parDiamSquare.PrintGrayScaleMap();
#endif

	// Map the output values
	MeasureTimeFn (&parTimeGrayscale, "Grayscale map generated in ",
	               &parDiamSquare, &DiamondSquareParallel::MapValuesToGrayScale);
	MeasureTimeFn (&parTimeCustomRangeMap, "Custom range map generated in ",
	               &parDiamSquare, &DiamondSquareParallel::MapValuesToIntRange,
	               0, 255);


	// Image saving
#if SAVE_GRAYSCALE_IMAGE
	std::cout << "\n - GRAYSCALE IMAGE - " << std::endl;
	MeasureTimeFn(nullptr, "Grayscale image generated and saved on file in ",
				  &parDiamSquare, &DiamondSquareBase::SaveGrayScaleImage,
	              GRAYSCALE_CUDA_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
	std::cout << "\n - COLOR IMAGE - " << std::endl;
	MeasureTimeFn (nullptr, "Color image generated and saved on file in ",
	               &parDiamSquare, &DiamondSquareBase::SaveColorImage,
	               COLOR_CUDA_PATH, 1);
#endif
}


// Execute the parallel version of the algorithm using the constant memory
void runParallelConstantMem ()
{
#if !TESTING_SETTINGS
	parDiamSquareConstMem.SetRandomScale (diamondSquareSettings[setting].randomScale);
	// Optionally it's possible to change the initial random scale
	//parDiamSquareConstMem.SetRandomScale(100);
	parDiamSquareConstMem.SetInitialStepSize (diamondSquareSettings[setting].initialStepSize);
	// Optionally it's possible to change the initial step size
	//parDiamSquareConstMem.SetInitialStepSize(32);
#else
	parDiamSquareConstMem.SetRandomScale(30.f);
	parDiamSquareConstMem.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

	// Execute the algorithm
	MeasureTimeFn (&parallelTime_const, "Parallel algorithm execution terminated in ",
	               &parDiamSquareConstMem, &DiamondSquareBase::ExecuteDiamondSquare);

	// Debug purposes
#if PRINT_GRAYSCALE_CUDA
	parDiamSquare.PrintGrayScaleMap();
#endif

	// Map the output values
	MeasureTimeFn (nullptr, "Grayscale map generated in ",
	               &parDiamSquareConstMem, &DiamondSquareBase::MapValuesToGrayScale);

	// Image saving
#if SAVE_GRAYSCALE_IMAGE
	std::cout << "\n - GRAYSCALE IMAGE - " << std::endl;
	MeasureTimeFn(nullptr, "Grayscale generated and saved on file in ",
					  &parDiamSquareConstMem, &DiamondSquareBase::SaveGrayScaleImage,
		              GRAYSCALE_CUDA_CONST_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
	std::cout << "\n - COLOR IMAGE - " << std::endl;
	MeasureTimeFn (nullptr, "Color generated and saved on file in ",
	               &parDiamSquareConstMem, &DiamondSquareBase::SaveColorImage,
	               COLOR_CUDA_CONST_PATH, 1);
#endif
}

int main (int argc, char** argv)
{
	// Initialize the CUDA context to avoid initial overhead while measuring executions time
	cudaFree (0);

	try
	{

#if RUN_SIZES_COMPARATOR
		runComparator();
		return 0;
#endif

		runParallel();
		std::cout << std::endl << std::endl << std::endl << std::endl;

#if COMPARE_CONSTANT_MEM
		runParallelConstantMem();
		std::cout << std::endl << std::endl << std::endl << std::endl;
#endif

#if COMPARE_SEQ
		runSequential();
		std::cout << std::endl << std::endl << std::endl << std::endl;
#endif

#if COMPARE_SEQ || COMPARE_CONSTANT_MEM
		std::cout << " === SPEED-UP RESULTS === " << std::endl;
#if COMPARE_CONSTANT_MEM
		CompareTime("Constant memory usage speed-up is ", parDiamSquare.GetExecutionTime(), parDiamSquareConstMem.GetExecutionTime());
#endif
#if COMPARE_SEQ
		CompareTime ("Diamond Square algorithm parallel speed-up is ", seqDiamSquare.GetExecutionTime(), parDiamSquare.GetExecutionTime());
		CompareTime ("Initialization and algorithm parallel speed-up is ", &sequentialTime, &parallelTime);
		CompareTime ("Grayscale map conversion parallel speed-up is ", &seqTimeGrayscale, &parTimeGrayscale);
		CompareTime ("Custom range map conversion parallel speed-up is ", &seqTimeCustomRangeMap, &parTimeCustomRangeMap);
#endif
#endif
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return 0;
	}
	std::cout << std::endl << std::endl;

	return (0);
}
