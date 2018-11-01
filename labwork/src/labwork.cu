#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    switch (lwNum) {
        case 1:
            timer.start();
            labwork.labwork1_CPU();
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork1-cpu-out.jpg");
            timer.start();
            labwork.labwork1_OpenMP();
            printf("labwork 1 OpenMP ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork1-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            timer.start();
            labwork.labwork3_GPU();
            printf("labwork 3 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            timer.start();
            labwork.labwork4_GPU();
            printf("labwork 4 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            timer.start();
            labwork.labwork5_CPU();
            printf("labwork 5 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            timer.start();
            labwork.labwork5_GPU();
            printf("labwork 5 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            timer.start();
            labwork.labwork6_GPU();
            printf("labwork 6 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            timer.start();
            labwork.labwork7_GPU();
            printf("labwork 7 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            timer.start();
            labwork.labwork8_GPU();
            printf("labwork 8 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            timer.start();
            labwork.labwork9_GPU();
            printf("labwork 9 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            timer.start();
            labwork.labwork10_GPU();
            printf("labwork 10 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] +
                                          (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    #pragma omp parallel for

    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int noOfGPUs = 0;
    cudaGetDeviceCount(&noOfGPUs);

    for (int i = 0; i < noOfGPUs; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\nGPU #%d:\n", i);
        printf(" - Name: %s\n", prop.name);
        printf(" - Core info:\n");
        printf("    + Clock rate: %d\n", prop.clockRate);
        printf("    + Number of cores: %d\n", getSPcores(prop));
        printf("    + Number of multiprocessors: %d\n", prop.multiProcessorCount);
        printf("    + Warp size: %d\n", prop.warpSize);
        printf(" - Memory info:\n");
        printf("    + Clock rate: %d\n", prop.memoryClockRate);
        printf("    + Bus width: %d\n", prop.memoryBusWidth);
        printf("    + Bandwidth: %d\n", prop.memoryClockRate * prop.memoryBusWidth);
    }
}

__global__ void grayscale(char *input, char *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid * 3] = (input[tid * 3] + input[tid * 3 + 1] + input[tid * 3 + 2]) / 3;
    output[tid * 3 + 2] = output[tid * 3 + 1] = output[tid * 3];
}

void Labwork::labwork3_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlock = pixelCount / blockSize;

    for (int i = 0; i < 100; ++i)
    {
        grayscale<<<numBlock, blockSize>>>(devInput, devOutput);
    }

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void grayscale2D(char *input, char *output) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    int globalId = globalIdX + globalIdY * gridDim.x * blockDim.x;

    // int globalBlockIdx = blockIdx.x + gridDim.x * blockIdx.y;
    // int globalId = globalBlockIdx * blockDim.x * blockDim.y + (threadIdx.x + blockDim.x * threadIdx.y);

    output[globalId * 3] = (input[globalId * 3] + input[globalId * 3 + 1] + input[globalId * 3 + 2]) / 3;
    output[globalId * 3 + 2] = output[globalId * 3 + 1] = output[globalId * 3];
}

void Labwork::labwork4_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3(inputImage->width / 32 + 1, inputImage->height / 32 + 1);

    for (int i = 0; i < 100; ++i)
    {
        grayscale2D<<<gridSize, blockSize>>>(devInput, devOutput);
    }

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,
                     0, 3, 13, 22, 13, 3, 0,
                     1, 13, 59, 97, 59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97, 59, 13, 1,
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

void Labwork::labwork5_GPU() {

}

void Labwork::labwork6_GPU() {

}

void Labwork::labwork7_GPU() {

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
