#pragma once

#include <cstdint>


struct BMP {
	bmpHeader header;
	bmpInfoHeader infoHeader;
	bmpColorHeader colorHeader;
	uint8_t* data;					//Data array

	BMP(const char* fname) {
		read(fname);
	}

	void read(const char* fname) {

	}

	//hasAlpha defines if the pixel element takes 24 or 32 bits
	BMP(int32_t width, int32_t height, bool hasAlpha = true) {

	}

	void write(const char* fname) {

	}
};


//I'm forcing the alignment to 1 byte instead of 4 bytes,
//in this way my struct will take exactly 14 bytes.
#pragma pack(push, 1) 

struct bmpHeader {
	uint16_t fileType{ 0x4D42 };	//BM, the standard BMP file type
	uint32_t fileSize{ 0 };			//File size in byte
	uint16_t reserved1{ 0 };		//Reserved
	uint16_t reserved2{ 0 };		//Reserved
	uint32_t offsetData{ 0 };		//Offset (in byte) to start the pixel data, starting from the beginning of the file
};

#pragma pop

struct bmpInfoHeader {
	uint32_t sizeOfInfoHeader{ 0 };	//The size of this header, in bytes
	int32_t width{ 0 };				//Bitmap width in pixels
	int32_t height{ 0 };			//Bitmap height in pixels

	uint16_t planesAmount{ 1 };		//Number of planes, always 1
	uint16_t bitsPerPixel{ 0 };		//Bits per pixel
	uint32_t compression{ 0 };		//0 or 3 (uncompressed)
	uint32_t imageSize{ 0 };		//0 for uncompressed images
	int32_t xPixelsPerMeter{ 0 };	
	int32_t yPixelsPerMeter{ 0 };	
	uint32_t colorsUsed{ 0 };		//Number of indices in the color table. 0 = max depending on bitsPerPixel.
	uint32_t colorsImportant{ 0 };	//Number of necessary colors. 0 = all colors required.
};

struct bmpColorHeader {
	uint32_t red_mask{ 0x00ff0000 };
	uint32_t green_mask{ 0x0000ff00 };
	uint32_t blue_mask{ 0x000000ff };
	uint32_t alpha_mask{ 0xff000000 };
	int32_t color_space_type{ 0x73524742 };		// Default "sRGB" (0x73524742)
	uint32_t unused[16]{ 0 };					// Unused data for sRGB color space
};