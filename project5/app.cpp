#include "app.h"
#include <stdlib.h>


int main(int argc, char *argv[])
{

	//mpi settings
	int  namelen, numprocs, myid;
	int size;
	int i;
	double t1, t2;

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Comm comm;
	MPI_Status status;

	char fileName[] = "c:\\data.txt";
	MPI_Datatype PointMPIDataType;
	double* results = NULL;
	Point* points = NULL;

	Point* pointsInCuda = NULL;
	double* resultsInCuda = NULL;
	double* weightsInCuda = NULL;

	int numberOfPoints, coordinatesOfPoints=0, limit;
	double a, a0, aMax, qc, q;

	points = readDataFromFile(fileName, points, &numberOfPoints, &coordinatesOfPoints, &a0, &aMax, &limit, &qc);
	checkValidInputs(numberOfPoints, coordinatesOfPoints, a0, aMax, limit);

	a = a0;
	
	//2. create wights vector init by zeros use calloc
	double* w = (double*)malloc((coordinatesOfPoints + 1) * sizeof(double));
	
	results = (double*)malloc((numberOfPoints) * sizeof(double));
	
	setArraysInGPU(points, w, numberOfPoints, coordinatesOfPoints);

	initPointDataType(&PointMPIDataType, coordinatesOfPoints);


	t1 = MPI_Wtime();
	while(a <= aMax)
	{
		fillWInZeros(w, coordinatesOfPoints + 1);
		//3.	Cycle through all given points Xi in the order as it is defined in the input file

		//4.	For each point Xi define a sign of discriminant function f(Xi) = WT Xi. If the values of vector W
		//		is chosen properly then all points belonging to set A will have positive value of f(Xi) and all points
		//		belonging to set B will have negative value of f(Xi).
		//		The first point P that does not satisfies this criterion will cause to stop the check and immediate
		//		redefinition of the vector W: W = W + [a*sign(f(P))] P

		//5.	 Loop through stages 3-4 till one of following satisfies:
		//			a.All given points are classified properly
		//			b.The number maximum iterations LIMIT is reached
		
		int iteration = 0;
		int x;
		while (iteration < limit)
		{
			for (i = 0; i < numberOfPoints; i++)
			{
				x = points[i].classification - sign(scalarMult(w, &points[i]));
				if (x) 
				{
					updateWeights(w, &points[i], a, (int)sign(x));   
				}
			}
			
			q = FindNmis(w,points,numberOfPoints,results,coordinatesOfPoints+1);
			//All given points are classified properly 
			int qIsLessThenQc = checkQuality(q, 0);
			if (qIsLessThenQc)
				break;

			iteration++;
		}
		// 6.	Find Nmis - the number of points that are wrongly classified, meaning that the value of f(Xi)
		//		is not correct for those points. Calculate a Quality of Classifier q according the formula
		//		q = Nmis / N
		
		q = FindNmis(w, points, numberOfPoints, results, coordinatesOfPoints + 1);

		// 7.	Check if the Quality of Classifier is reached 
		int qIsLessThenQc = checkQuality(q, qc);
		
		// 8.	Stop if q < QC.
		if (qIsLessThenQc)
			break;

		// 9.	Increment the value of a
		a += a0;


	}
	//10.	Loop through stages 2-9
	t2 = MPI_Wtime();

	printWeights(w, (coordinatesOfPoints + 1));
	printf("data : %s\nthe quality is %f \nqc: %f \nalpha %f\ntime: %f",fileName, q, qc, a,t2-t1);

	//free the data allocation
	free(points);
	free(w);
	freeArraysInGPU();

	MPI_Finalize();
	return 0;

}

void fillWInZeros(double* w, int size)
{
	int i;
	for (i = 0; i < size; i++)
		w[i] = 0;
}

Point* readDataFromFile(char* fileName, Point* points, int* numberOfPoints, int* coordinatesOfPoints, double* a0, double* aMax, int* limit, double* qc)
{
	FILE* file = fopen(fileName, "r");


	if (file == NULL)
	{
		printf("file null pointer");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

	}

	//reading settings
	fscanf(file, "%d %d %lf %lf %d %lf", numberOfPoints, coordinatesOfPoints, a0, aMax, limit, qc);
	//create points array

	points = (Point*)malloc(sizeof(Point)*(*numberOfPoints));


	int i, j;
	//start reading points
	for (i = 0; i < *numberOfPoints; i++)
	{
		points[i].coordinatesOfPoints = *coordinatesOfPoints;
		for (j = 0; j < *coordinatesOfPoints; j++)
		{
			fscanf(file, "%lf ", &(points[i].coordinates[j]));
		}
		points[i].coordinates[*coordinatesOfPoints] = 1;
		fscanf(file, "%d\n", &(points[i].classification));
	}



	fclose(file);

	return points;

}

void checkValidInputs(int numberOfPoints, int coordinatesOfPoints, double a0, double aMax, int limit)
{
	int flag = 0;
	if (numberOfPoints< MINIMAL_POINTS_NUMBER || numberOfPoints > MAXIMUM_POINTS_NUMBER)
	{
		printf("in valid points number\n");
		flag = 1;
	}
	else if (coordinatesOfPoints > coordinatesOfPoints)
	{
		printf("invalid dim number\n");
		flag = 1;
	}
	else if (aMax / a0 > MAXIMUM_ALPHA_CHECKS)
	{
		printf("invalid alpha range\n");
		flag = 1;
	}
	else if (limit > MAXIMUM_ITERATION_LIMIT)
	{
		printf("invalid limit range\n");
		flag = 1;
	}

	if (flag)
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

}

void initPointDataType(MPI_Datatype* PointMPIDataType, int coordinatesOfPoints)
{
	Point point;
	MPI_Datatype type[3] = { MPI_INT,MPI_DOUBLE ,MPI_INT };
	int blockLen[3] = { 1,coordinatesOfPoints,1 };
	MPI_Aint disp[3];

	// Create MPI user data type for partical
	disp[0] = (char *)&point.coordinatesOfPoints - (char *)&point;
	disp[1] = (char *)&point.coordinates - (char *)&point;
	disp[2] = (char *)&point.classification - (char *)&point;

	MPI_Type_create_struct(3, blockLen, disp, type, PointMPIDataType);

	MPI_Type_commit(PointMPIDataType);
}

int sign(double number)
{

	return number >= 0 ? 1 : -1;

}

double scalarMult(double* w, Point* point)
{
	int i;
	double sum = 0;

	for (i = 0; i < point->coordinatesOfPoints + 1; i++)
	{
		sum += (w[i] * point->coordinates[i]);
	}
	return sum;
}

void printWeights(double* w, int size)
{
	int i = 0;
	printf("\n[");
	for (i = 0; i < size; i++)
	{
		printf(" %d ", (int)w[i]);
	}
	printf("]\n");

}

double f(double* w, Point* point)
{
	return scalarMult(w, point);
}

void updateWeights(double* last, Point* point, double alpha, int x)
{

	int i;
	double fx = f(last, point);
	for (i = 0; i < point->coordinatesOfPoints + 1; i++)
	{
		last[i] = last[i] + alpha * sign(x)  * (point->coordinates[i]);
	}
}

double FindNmis(double* w, Point* points, int numOfPoints, double* results, int weightSize)
{

	int i, j;
	double value = 0;
	int notCorrectPoints = 0;

	cudaError_t cudaStatus = calculateWithCuda(w, results, numOfPoints, weightSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

#pragma omp parallel for private(value) reduction(+: notCorrectPoints)
	for (i = 0; i < numOfPoints; i++)
	{
		if ((points[i].classification == SET_A && results[i] <= 0) || (points[i].classification == SET_B && results[i] >= 0))
		{
			notCorrectPoints++;
		}
	}

	return (double)notCorrectPoints / numOfPoints;

}

int checkQuality(double q, double qc)
{
	return q <= qc ? 1 : 0;
}

int classified(double* w, Point* point)
{
	double  fx = scalarMult(w, point);

	if (point->classification == SET_A && fx >= 0)
		return 1;

	if (point->classification == SET_B && fx < 0)
		return 1;
	else
		return 0;

}