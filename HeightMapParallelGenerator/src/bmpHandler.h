#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include "ColorPixel.h"


//I'm forcing the alignment to 1 byte instead of 4 bytes,
//in this way my struct will take exactly 14 bytes.
#pragma pack(push, 1)

struct BMPHeader {
	uint16_t fileType{0x4D42}; //BM, the standard BMP file type
	uint32_t fileSize{0}; //File size in byte
	uint16_t reserved1{0}; //Reserved
	uint16_t reserved2{0}; //Reserved
	uint32_t offsetData{0}; //Offset (in byte) to start the pixel data, starting from the beginning of the file
};

#pragma pop


struct BMPInfoHeader {
	uint32_t size{0}; //The size of this header, in bytes
	int32_t width{0}; //Bitmap width in pixels
	int32_t height{0}; //Bitmap height in pixels

	uint16_t planesAmount{1}; //Number of planes, always 1
	uint16_t bitsPerPixel{0}; //Bits per pixel
	uint32_t compression{0}; //0 or 3 (uncompressed)
	uint32_t imageSize{0}; //0 for uncompressed images
	int32_t xPixelsPerMeter{0};
	int32_t yPixelsPerMeter{0};
	uint32_t colorsUsed{0}; //Number of indices in the color table. 0 = max depending on bitsPerPixel.
	uint32_t colorsImportant{0}; //Number of necessary colors. 0 = all colors required.
};

struct BMPColorHeader {
	uint32_t redMask{0x00ff0000};
	uint32_t greenMask{0x0000ff00};
	uint32_t blueMask{0x000000ff};
	uint32_t alphaMask{0xff000000};
	int32_t colorSpaceType{0x73524742}; // Default "sRGB" (0x73524742)
	uint32_t unused[16]{0}; // Unused data for sRGB color space
};

struct BMP {
public:
	BMPHeader header;
	BMPInfoHeader infoHeader;
	BMPColorHeader colorHeader;
	std::vector<uint8_t> data; //Data array

	BMP(const char* fname) {
		Read(fname);
	}

	void Read(const char* fname) {
		//Read the file in binary mode
		std::ifstream input{fname, std::ios::binary};
		if (input) {

			//Read the header and put it on my BMPHeader instance
			input.read(reinterpret_cast<char*>(&header), sizeof(header));
			if (header.fileType != 0x4D42) {
				//If the file type is not BM, then it's not a BMP file
				throw std::runtime_error("Not a BMP file");
			}

			//Read the info header and put it on my BMPInfoHeader instance
			input.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
			// The BMPColorHeader is used only for transparent images
			if (infoHeader.bitsPerPixel == 32) {
				//Check if the file has bit mask color information
				if (infoHeader.size >= (sizeof(BMPInfoHeader) + sizeof(BMPColorHeader))) {
					input.read(reinterpret_cast<char*>(&colorHeader), sizeof(colorHeader));
					//Read the color header and put it on my BMPColorHeader instance
					CheckColorHeader(colorHeader);
				}
				else {
					std::cerr << "Warning! The file \"" << fname <<
						"\" does not seem to contain bit mask information\n";
					throw std::runtime_error("Error! Unrecognized file format!");
				}
			}

			input.seekg(header.offsetData); //Go to the start of the pixel data

			// Adjust the header fields for output.
			// Some editors will put extra info in the image file, we only save the headers and the data.
			if (infoHeader.bitsPerPixel == 32) {
				infoHeader.size = sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);
				header.offsetData = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);
			}
			else {
				infoHeader.size = sizeof(BMPInfoHeader);
				header.offsetData = sizeof(BMPHeader) + sizeof(BMPInfoHeader);
			}
			header.fileSize = header.offsetData;

			if (infoHeader.height < 0) {
				throw std::runtime_error(
					"The program can treat only BMP images with the origin in the bottom left corner!");
			}

			//Take the image size from the header and resize the data vector 
			data.resize(infoHeader.width * infoHeader.height * (infoHeader.bitsPerPixel / sizeof(uint8_t)));

			//Check if we need to take into account row padding
			//If not..
			if (infoHeader.width % 4 == 0) {
				//..Read the image data and put it on my data vector
				input.read(reinterpret_cast<char*>(data.data()), data.size());
				header.fileSize += data.size();
			}
			else {
				//..otherwise discard the padding data
				rowStride = infoHeader.width * infoHeader.bitsPerPixel / sizeof(uint8_t);
				uint32_t newStride = MakeStrideAligned(4);
				std::vector<uint8_t> paddingRow(newStride - rowStride);

				for (int y = 0; y < infoHeader.height; ++y) {
					input.read(reinterpret_cast<char*>(data.data() + rowStride * y), rowStride);
					input.read(reinterpret_cast<char*>(paddingRow.data()), paddingRow.size());
				}
				header.fileSize += data.size() + infoHeader.height * paddingRow.size();
			}
		}
		else {
			throw std::runtime_error("Unable to open the input image file.");
		}
	}

	//hasAlpha defines if the pixel element takes 24 or 32 bits
	BMP(int32_t width, int32_t height, bool hasAlpha = true) {
		if (width <= 0 || height <= 0) {
			throw std::runtime_error("The image width and height must be positive numbers.");
		}

		infoHeader.width = width;
		infoHeader.height = -height;
		if (hasAlpha) {
			infoHeader.size = sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);
			header.offsetData = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);

			infoHeader.bitsPerPixel = 32;
			infoHeader.compression = 3;
			rowStride = width * 4;
			data.resize(rowStride * height);
			header.fileSize = header.offsetData + data.size();
		}
		else {
			infoHeader.size = sizeof(BMPInfoHeader);
			header.offsetData = sizeof(BMPHeader) + sizeof(BMPInfoHeader);

			infoHeader.bitsPerPixel = 24;
			infoHeader.compression = 0;
			rowStride = width * 3;
			data.resize(rowStride * height);

			uint32_t newStride = MakeStrideAligned(4);
			header.fileSize = header.offsetData + data.size() + infoHeader.height * (newStride - rowStride);
		}
	}

	void Write(const char* fname) {
		std::ofstream output{fname, std::ios_base::binary};
		if (output) {
			//We consider only the 24 and 32 bits per pixel case
			if (infoHeader.bitsPerPixel == 32) {
				WriteHeadersAndData(output);
			}
			else if (infoHeader.bitsPerPixel == 24) {
				//Check is we need to take into account row padding
				//If not..
				if (infoHeader.width % 4 == 0) {
					WriteHeadersAndData(output);
				}
				else {
					//..otherwise Write also some padding data
					uint32_t newStride = MakeStrideAligned(4);
					std::vector<uint8_t> paddingRow(newStride - rowStride);

					WriteHeader(output);

					for (int y = 0; y < infoHeader.height; ++y) {
						output.write(reinterpret_cast<const char*>(data.data() + rowStride * y), rowStride);
						output.write(reinterpret_cast<const char*>(paddingRow.data()), paddingRow.size());
					}
				}
			}
			else {
				throw std::runtime_error("The program can treat only 24 and 32 bits per pixel BMP files");
			}
		}
		else {
			throw std::runtime_error("Unable to open the output image file.");
		}
	}

	void FillRegion(uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
	                uint8_t B, uint8_t G, uint8_t R, uint8_t A) {
		if (x0 + w > static_cast<uint32_t>(infoHeader.width) || y0 + h > static_cast<uint32_t>(infoHeader.height)) {
			throw std::runtime_error("The region does not fit in the image!");
		}

		uint32_t channels = infoHeader.bitsPerPixel / 8;
		for (uint32_t y = y0; y < y0 + h; ++y) {
			for (uint32_t x = x0; x < x0 + w; ++x) {
				data[channels * (y * infoHeader.width + x) + 0] = B;
				data[channels * (y * infoHeader.width + x) + 1] = G;
				data[channels * (y * infoHeader.width + x) + 2] = R;
				if (channels == 4)
					data[channels * (y * infoHeader.width + x) + 3] = A;
			}
		}
	}

	void FillRegion(uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, const ColorPixel& color, uint8_t A) {
		if (x0 + w > static_cast<uint32_t>(infoHeader.width) || y0 + h > static_cast<uint32_t>(infoHeader.height)) {
			throw std::runtime_error("The region does not fit in the image!");
		}

		uint32_t channels = infoHeader.bitsPerPixel / 8;
		for (uint32_t y = y0; y < y0 + h; ++y) {
			for (uint32_t x = x0; x < x0 + w; ++x) {
				data[channels * (y * infoHeader.width + x) + 0] = color.B;
				data[channels * (y * infoHeader.width + x) + 1] = color.G;
				data[channels * (y * infoHeader.width + x) + 2] = color.R;
				if (channels == 4)
					data[channels * (y * infoHeader.width + x) + 3] = A;
			}
		}
	}

private:
	uint32_t rowStride{0};

	uint32_t MakeStrideAligned(uint32_t alignment) {
		uint32_t newStride = rowStride;
		while (newStride % alignment != 0) {
			newStride++;
		}
		return newStride;
	}

	void WriteHeadersAndData(std::ofstream& output) {
		WriteHeader(output);
		output.write(reinterpret_cast<const char*>(data.data()), data.size());
	}

	void WriteHeader(std::ofstream& output) {
		output.write(reinterpret_cast<const char*>(&header), sizeof(header));
		output.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));
		if (infoHeader.bitsPerPixel == 32) {
			output.write(reinterpret_cast<const char*>(&colorHeader), sizeof(colorHeader));
		}
	}

	void CheckColorHeader(BMPColorHeader& bmpColorHeader) {
		BMPColorHeader expectedColorHeader;
		if (expectedColorHeader.redMask != bmpColorHeader.redMask ||
			expectedColorHeader.greenMask != bmpColorHeader.greenMask ||
			expectedColorHeader.blueMask != bmpColorHeader.blueMask ||
			expectedColorHeader.alphaMask != bmpColorHeader.alphaMask) {
			throw std::runtime_error(
				"Unexpected color mask format! The program expects the pixel data to be in the BGRA format.");
		}

		if (expectedColorHeader.colorSpaceType != bmpColorHeader.colorSpaceType) {
			throw std::runtime_error(
				"Unexpected color space type! The program expects the pixel data to be in the sRGB color space.");
		}
	}
};
