#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mpi.h>

#include <stdio.h>
#include <math.h>
#define HEAVY 10000

#define SET_A 1
#define SET_B -1
#define MINIMAL_POINTS_NUMBER 100000 
#define MAXIMUM_POINTS_NUMBER 500000 
#define MAXIMUM_DIM_SIZE 20
#define MAXIMUM_ALPHA_CHECKS 100
#define MAXIMUM_ITERATION_LIMIT 1000


#pragma warning (disable : 4996)

typedef struct {
	int coordinatesOfPoints;
	double coordinates[MAXIMUM_DIM_SIZE+1];
	int classification;
}Point;

void fillWInZeros(double* w, int size);
Point* readDataFromFile(char* fileName, Point* points, int* numberOfPoints, int* coordinatesOfPoints, double* a0, double* aMax, int* limit, double* qc);
void checkValidInputs(int numberOfPoints, int coordinatesOfPoints, double a0, double aMax, int limit);
void initPointDataType(MPI_Datatype* PointMPIDataType, int coordinatesOfPoints);
int sign(double number);
double scalarMult(double* w, Point* point);
void printWeights(double* w, int size);
double f(double* w, Point* point);
void updateWeights(double* last, Point* point, double alpha, int x);
double FindNmis(double* w, Point* points, int numOfPoints, double* results, int weightSize);
int checkQuality(double q, double qc);
int classified(double* w, Point* point);

cudaError_t calculateWithCuda(double* weights, double* results, int size, int weightSize);
cudaError_t setArraysInGPU(Point* points, double* weights, int n, int k);
void freeArraysInGPU();