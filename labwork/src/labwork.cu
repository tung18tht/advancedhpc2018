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

    // Lab 6
    int mode, param;
    float paramFloat;
    std::string inputFilename2;
    JpegInfo *inputImage2;

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
            timer.start();
            labwork.labwork5_GPU_optimized();
            printf("labwork 5 GPU optimized ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-gpu-optimized-out.jpg");
            break;
        case 6:
            mode = atoi(argv[3]);
            param = atoi(argv[4]);
            if (mode == 0) {
                timer.start();
                labwork.labwork6a_GPU(param);
                printf("labwork 6a GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
                labwork.saveOutputImage("labwork6a-gpu-out.jpg");
            } else if (mode == 1) {
                timer.start();
                labwork.labwork6b_GPU(param);
                printf("labwork 6b GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
                labwork.saveOutputImage("labwork6b-gpu-out.jpg");
            } else if (mode == 2) {
                paramFloat = atof(argv[4]);
                inputFilename2 = std::string(argv[5]);
                inputImage2 = labwork.loadImage(inputFilename2);
                timer.start();
                labwork.labwork6c_GPU(paramFloat, inputImage2);
                printf("labwork 6c GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
                labwork.saveOutputImage("labwork6c-gpu-out.jpg");
            }
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

JpegInfo* Labwork::loadImage(std::string fileName) {
    return jpegLoader.load(fileName);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) { // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = ((unsigned char) inputImage->buffer[i * 3] +
                                  (unsigned char) inputImage->buffer[i * 3 + 1] +
                                  (unsigned char) inputImage->buffer[i * 3 + 2]) / 3;
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    #pragma omp parallel for

    for (int j = 0; j < 100; j++) { // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = ((unsigned char) inputImage->buffer[i * 3] +
                                  (unsigned char) inputImage->buffer[i * 3 + 1] +
                                  (unsigned char) inputImage->buffer[i * 3 + 2]) / 3;
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

__global__ void grayscale(unsigned char *input, char *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid * 3] = (input[tid * 3] + input[tid * 3 + 1] + input[tid * 3 + 2]) / 3;
    output[tid * 3 + 2] = output[tid * 3 + 1] = output[tid * 3];
}

void Labwork::labwork3_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlock = pixelCount / blockSize;

    grayscale<<<numBlock, blockSize>>>(devInput, devOutput);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void grayscale2D(unsigned char *input, char *output, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    // int globalBlockIdx = blockIdx.x + gridDim.x * blockIdx.y;
    // int globalId = globalBlockIdx * blockDim.x * blockDim.y + (threadIdx.x + blockDim.x * threadIdx.y);

    output[globalId * 3] = (input[globalId * 3] + input[globalId * 3 + 1] + input[globalId * 3 + 2]) / 3;
    output[globalId * 3 + 2] = output[globalId * 3 + 1] = output[globalId * 3];
}

void Labwork::labwork4_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    grayscale2D<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = {0, 0,  1,  2,   1,  0,  0,
                    0, 3,  13, 22,  13, 3,  0,
                    1, 13, 59, 97,  59, 13, 1,
                    2, 22, 97, 159, 97, 22, 2,
                    1, 13, 59, 97,  59, 13, 1,
                    0, 3,  13, 22,  13, 3,  0,
                    0, 0,  1,  2,   1,  0,  0};
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
                    unsigned char gray = ((unsigned char) inputImage->buffer[tid * 3] +
                                          (unsigned char) inputImage->buffer[tid * 3 + 1] +
                                          (unsigned char) inputImage->buffer[tid * 3 + 2]) / 3;
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

__global__ void gaussianBlur(unsigned char *input, char *output, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    int weights[] = {0, 0,  1,  2,   1,  0,  0,
                     0, 3,  13, 22,  13, 3,  0,
                     1, 13, 59, 97,  59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97,  59, 13, 1,
                     0, 3,  13, 22,  13, 3,  0,
                     0, 0,  1,  2,   1,  0,  0};

    int sum = 0;
    int c = 0;
    for (int y = -3; y <= 3; y++) {
        for (int x = -3; x <= 3; x++) {
            int i = globalIdX + x;
            int j = globalIdY + y;
            if (i < 0) continue;
            if (i >= width) continue;
            if (j < 0) continue;
            if (j >= height) continue;
            int tid = j * width + i;
            unsigned char gray = (input[tid * 3] + input[tid * 3 + 1] + input[tid * 3 + 2])/3;
            int coefficient = weights[(y+3) * 7 + x + 3];
            sum = sum + gray * coefficient;
            c += coefficient;
        }
    }
    sum /= c;

    output[globalId * 3] = output[globalId * 3 + 1] = output[globalId * 3 + 2] = sum;
}

void Labwork::labwork5_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    gaussianBlur<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void gaussianBlurOptimized(unsigned char *input, char *output, int width, int height, int* weights) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    __shared__ int sharedWeights[49];

    int localId = threadIdx.x + threadIdx.y * blockDim.x;
    if (localId < 49)
    {
        sharedWeights[localId] = weights[localId];
    }

    __syncthreads();

    int sum = 0;
    int c = 0;
    for (int y = -3; y <= 3; y++) {
        for (int x = -3; x <= 3; x++) {
            int i = globalIdX + x;
            int j = globalIdY + y;
            if (i < 0) continue;
            if (i >= width) continue;
            if (j < 0) continue;
            if (j >= height) continue;
            int tid = j * width + i;
            unsigned char gray = (input[tid * 3] + input[tid * 3 + 1] + input[tid * 3 + 2])/3;
            int coefficient = sharedWeights[(y+3) * 7 + x + 3];
            sum = sum + gray * coefficient;
            c += coefficient;
        }
    }
    sum /= c;
    output[globalId * 3] = sum;
    output[globalId * 3 + 1] = sum;
    output[globalId * 3 + 2] = sum;
}

void Labwork::labwork5_GPU_optimized() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    int weights[] = {0, 0,  1,  2,   1,  0,  0,
                     0, 3,  13, 22,  13, 3,  0,
                     1, 13, 59, 97,  59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97,  59, 13, 1,
                     0, 3,  13, 22,  13, 3,  0,
                     0, 0,  1,  2,   1,  0,  0};

    int* devWeights;
    cudaMalloc(&devWeights, 49 * sizeof(int));
    cudaMemcpy(devWeights, weights, 49 * sizeof(int), cudaMemcpyHostToDevice);

    gaussianBlurOptimized<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, devWeights);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void binarization(unsigned char *input, char *output, int width, int height, int threshold) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    unsigned char binary = (input[globalId * 3] + input[globalId * 3 + 1] + input[globalId * 3 + 2]) / 3;
    // binary = (binary / threshold) > 0 ? 255 : 0;
    binary = min(binary / threshold, 1) * 255;

    output[globalId * 3] = binary;
    output[globalId * 3 + 1] = binary;
    output[globalId * 3 + 2] = binary;
}

void Labwork::labwork6a_GPU(int threshold) {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    binarization<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, threshold);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void brightness(unsigned char *input, char *output, int width, int height, int brightnessChange) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    unsigned char r = min(max(input[globalId * 3] + brightnessChange, 0), 255);
    unsigned char g = min(max(input[globalId * 3 + 1] + brightnessChange, 0), 255);
    unsigned char b = min(max(input[globalId * 3 + 2] + brightnessChange, 0), 255);

    output[globalId * 3] = r;
    output[globalId * 3 + 1] = g;
    output[globalId * 3 + 2] = b;
}

void Labwork::labwork6b_GPU(int brightnessChange) {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    brightness<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, brightnessChange);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void blend(unsigned char *input, unsigned char *input2, char *output, int width, int height, float ratio) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    unsigned char r = input[globalId * 3] * ratio + input2[globalId * 3] * (1 - ratio);
    unsigned char g = input[globalId * 3 + 1] * ratio + input2[globalId * 3 + 1] * (1 - ratio);
    unsigned char b = input[globalId * 3 + 2] * ratio + input2[globalId * 3 + 2] * (1 - ratio);

    output[globalId * 3] = r;
    output[globalId * 3 + 1] = g;
    output[globalId * 3 + 2] = b;
}

void Labwork::labwork6c_GPU(float ratio, JpegInfo *inputImage2) {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    unsigned char* devInput2;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devInput2, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(devInput2, inputImage2->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    blend<<<gridSize, blockSize>>>(devInput, devInput2, devOutput, inputImage->width, inputImage->height, ratio);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void getGreyscaleAndMaxMinIntensity(unsigned char *input, unsigned char *output, unsigned char *globalMax, unsigned char *globalMin, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    unsigned char grey = (input[globalId * 3] + input[globalId * 3 + 1] + input[globalId * 3 + 2]) / 3;
    output[globalId] = grey;

    extern __shared__ unsigned char shared[];
    unsigned char *blockMaxArray = shared;
    unsigned char *blockMinArray = &blockMaxArray[blockDim.x * blockDim.y];

    int localId = threadIdx.x + threadIdx.y * blockDim.x;
    blockMaxArray[localId] = grey;
    blockMinArray[localId] = grey;

    __syncthreads();

    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s /= 2) {
        if (localId < s) {
            blockMaxArray[localId] = max(blockMaxArray[localId], blockMaxArray[localId + s]);
            blockMinArray[localId] = min(blockMinArray[localId], blockMinArray[localId + s]);
            __syncthreads();
        }
    }

    if (localId == 0) {
        globalMax[0] = max(blockMaxArray[0], globalMax[0]);
        globalMin[0] = min(blockMinArray[0], globalMin[0]);
    }
}

__global__ void grayscaleStretch(unsigned char *input, char *output, unsigned char *max, unsigned char *min, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    unsigned char greyStretched = ((float) (input[globalId] - min[0]) / (max[0] - min[0])) * 255;

    output[globalId * 3] = greyStretched;
    output[globalId * 3 + 1] = greyStretched;
    output[globalId * 3 + 2] = greyStretched;
}

void Labwork::labwork7_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char *devInput, *devInputGrey;
    char *devOutput;
    unsigned char *devMax, *devMin;
    unsigned char tempMax = 0, tempMin = 255;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devInputGrey, pixelCount);
    cudaMalloc(&devOutput, pixelCount * 3);
    cudaMalloc(&devMax, 1);
    cudaMalloc(&devMin, 1);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(devMax, &tempMax, 1, cudaMemcpyHostToDevice);
    cudaMemcpy(devMin, &tempMin, 1, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    getGreyscaleAndMaxMinIntensity<<<gridSize, blockSize, blockX * blockY * 2>>>(devInput, devInputGrey, devMax, devMin, inputImage->width, inputImage->height);
    grayscaleStretch<<<gridSize, blockSize>>>(devInputGrey, devOutput, devMax, devMin, inputImage->width, inputImage->height);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    // unsigned char *max, *min;
    // max = (unsigned char *) malloc(1);
    // min = (unsigned char *) malloc(1);
    // cudaMemcpy(max, devMax, 1, cudaMemcpyDeviceToHost);
    // cudaMemcpy(min, devMin, 1, cudaMemcpyDeviceToHost);
    // printf("Max: %d\n", *max);
    // printf("Min: %d\n", *min);

    cudaFree(devInput);
    cudaFree(devOutput);
    cudaFree(devMax);
    cudaFree(devMin);
}

__global__ void RGB2HSV(unsigned char *input, int *hue, float *saturation, float *value, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    float floatR = input[globalId * 3] / 255.0f;
    float floatG = input[globalId * 3 + 1] / 255.0f;
    float floatB = input[globalId * 3 + 2] / 255.0f;
    float maxValue = max(max(floatR, floatG), floatB);
    float delta = maxValue - min(min(floatR, floatG), floatB);

    value[globalId] = maxValue;

    if (delta == 0.0f) {
        hue[globalId] = 0;
        saturation[globalId] = 0.0f;
    } else {
        saturation[globalId] = delta / maxValue;

        if (maxValue == floatR) {
            hue[globalId] = 60.0f * fmod((floatG - floatB) / delta, 6.0f);
        } else if (maxValue == floatG) {
            hue[globalId] = 60.0f * ((floatB - floatR) / delta + 2.0f);
        } else {
            hue[globalId] = 60.0f * ((floatR - floatG) / delta + 4.0f);
        }

        if (hue[globalId] < 0) {
            hue[globalId] = 360 + hue[globalId];
        }
    }
}

__global__ void HSV2RGB(int *hue, float *saturation, float *value, char *output, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    float c = value[globalId] * saturation[globalId];
    float x = c * (1.0f - abs(fmod(hue[globalId] / 60.0f, 2.0f) - 1.0f));
    float m = value[globalId] - c;

    if (hue[globalId] < 60) {
        output[globalId * 3] = (c + m) * 255.0f;
        output[globalId * 3 + 1] = (x + m) * 255.0f;
        output[globalId * 3 + 2] = m * 255.0f;
    } else if (hue[globalId] < 120) {
        output[globalId * 3] = (x + m) * 255.0f;
        output[globalId * 3 + 1] = (c + m) * 255.0f;
        output[globalId * 3 + 2] = m * 255.0f;
    } else if (hue[globalId] < 180) {
        output[globalId * 3] = m * 255.0f;
        output[globalId * 3 + 1] = (c + m) * 255.0f;
        output[globalId * 3 + 2] = (x + m) * 255.0f;
    } else if (hue[globalId] < 240) {
        output[globalId * 3] = m * 255.0f;
        output[globalId * 3 + 1] = (x + m) * 255.0f;
        output[globalId * 3 + 2] = (c + m) * 255.0f;
    } else if (hue[globalId] < 300) {
        output[globalId * 3] = (x + m) * 255.0f;
        output[globalId * 3 + 1] = m * 255.0f;
        output[globalId * 3 + 2] = (c + m) * 255.0f;
    } else {
        output[globalId * 3] = (c + m) * 255.0f;
        output[globalId * 3 + 1] = m * 255.0f;
        output[globalId * 3 + 2] = (x + m) * 255.0f;
    }
}

void Labwork::labwork8_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);

    unsigned char *devInput;
    int *devHue;
    float *devSaturation, *devValue;
    char *devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);
    cudaMalloc(&devHue, pixelCount * sizeof(int));
    cudaMalloc(&devSaturation, pixelCount * sizeof(float));
    cudaMalloc(&devValue, pixelCount * sizeof(float));

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    RGB2HSV<<<gridSize, blockSize>>>(devInput, devHue, devSaturation, devValue, inputImage->width, inputImage->height);

    HSV2RGB<<<gridSize, blockSize>>>(devHue, devSaturation, devValue, devOutput, inputImage->width, inputImage->height);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
    cudaFree(devHue);
    cudaFree(devSaturation);
    cudaFree(devValue);
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
