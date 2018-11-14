#pragma once

#include <include/jpegloader.h>
#include <include/timer.h>

class Labwork {

private:
    JpegLoader jpegLoader;
    JpegInfo *inputImage;
    char *outputImage;

public:
    void loadInputImage(std::string inputFileName);
    void saveOutputImage(std::string outputFileName);
    JpegInfo* loadImage(std::string fileName);

    void labwork1_CPU();
    void labwork1_OpenMP();

    void labwork2_GPU();

    void labwork3_GPU();

    void labwork4_GPU();

    void labwork5_CPU();
    void labwork5_GPU();
    void labwork5_GPU_optimized();

    void labwork6a_GPU(int threshold);
    void labwork6b_GPU(int brightnessChange);
    void labwork6c_GPU(float ratio, JpegInfo *inputImage2);

    void labwork7_GPU();

    void labwork8_GPU();

    void labwork9_GPU();

    void labwork10_GPU(int windowSize);
};
