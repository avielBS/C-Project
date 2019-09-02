#include "app.h"
#include <stdlib.h>

#define INPUT "c:\\data.txt"
#define OUTPUT "c:\\output.txt"



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

	MPI_Datatype PointMPIDataType;
	double* results = NULL;
	Point* points = NULL;


	int numberOfPoints, coordinatesOfPoints=0, limit;
	double a, a0, aMax, qc, q ,otherProcessAlpha ,otherProcessQ;

	initPointDataType(&PointMPIDataType);

	double x, fx;
	int iteration = 0;

	if (numprocs != 2)
	{	
		printf("please run this program with 2 proccess");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}


	//  read and send data to second computer
	if (myid == 0)
	{
		points = readDataFromFile(INPUT, points, &numberOfPoints, &coordinatesOfPoints, &a0, &aMax, &limit, &qc);
		checkValidInputs(numberOfPoints, coordinatesOfPoints, a0, aMax, limit);
		t1 = MPI_Wtime();	

		MPI_Send(&numberOfPoints, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Send(points, numberOfPoints, PointMPIDataType, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&coordinatesOfPoints, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&aMax, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&limit, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&qc, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&a0, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
	}
	else //recive the data 
	{
		MPI_Recv(&numberOfPoints, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,&status);
		points = (Point*)malloc(sizeof(Point)*(numberOfPoints));
		
		MPI_Recv(points, numberOfPoints, PointMPIDataType, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&coordinatesOfPoints, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&aMax, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&limit, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&qc, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&a0, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		a0 += a0;
	}
	
	//1. a = a0
	a = a0;
	
	//2. create wights vector
	double* w = (double*)malloc((coordinatesOfPoints + 1) * sizeof(double));

	setArraysInGPU(points, w, numberOfPoints, coordinatesOfPoints);
	//results array for cuda
	results = (double*)malloc((numberOfPoints) * sizeof(double));
	


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
		iteration = 0;
		while (iteration < limit)
		{
			for (i = 0; i < numberOfPoints; i++)
			{
				fx = f(w, &points[i]);
				x = points[i].classification - sign(fx);
				if(x)
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
		if (myid == 0)
			a += (a0 * 2);
		else
			a += a0;

	}
	//10.	Loop through stages 2-9

	if (myid == 0) //recive other computer result
	{
		MPI_Recv(&otherProcessAlpha, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&otherProcessQ, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
		double* weightFromOtherProcess = (double*)malloc((coordinatesOfPoints + 1) * sizeof(double));
		MPI_Recv(weightFromOtherProcess, coordinatesOfPoints+1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
		t2 = MPI_Wtime();

		FILE* output = fopen(OUTPUT, "w");

		if (a < otherProcessAlpha && checkQuality(q, qc))
		{
			writeWeights(w, (coordinatesOfPoints + 1),output);
			fprintf(output,"data : %s\nthe quality is %f \nqc: %f \nalpha %f\ntime: %f", INPUT, q, qc, a, t2 - t1);
		}
		else if (otherProcessAlpha < a && checkQuality(otherProcessAlpha, qc))
		{
			writeWeights(weightFromOtherProcess, (coordinatesOfPoints + 1),output);
			fprintf(output,"data : %s\nthe quality is %f \nqc: %f \nalpha %f\ntime: %f", INPUT, otherProcessQ, qc, otherProcessAlpha, t2 - t1);
		}
		else if(!checkQuality(q,qc) && !checkQuality(otherProcessQ,qc))
			fprintf(output,"Alpha is not found");
		
		free(weightFromOtherProcess);
		fclose(output);
		printf("Output file is ready\n");
	}
	else //send to main process my result
	{
		MPI_Send(&a, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&q, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&w, coordinatesOfPoints+1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}


	//free data allocation
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
		printf("in valid points number");
		flag = 1;
	}
	else if (coordinatesOfPoints > coordinatesOfPoints)
	{
		printf("invalid dim number");
		flag = 1;
	}
	else if (aMax / a0 > MAXIMUM_ALPHA_CHECKS)
	{
		printf("invalid alpha range");
		flag = 1;
	}
	else if (limit > MAXIMUM_ITERATION_LIMIT)
	{
		printf("invalid limit range");
		flag = 1;
	}

	if (flag)
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

}

void initPointDataType(MPI_Datatype* PointMPIDataType)
{
	Point point;
	MPI_Datatype type[3] = { MPI_INT,MPI_DOUBLE,MPI_INT };
	int blockLen[3] = { 1,MAXIMUM_DIM_SIZE + 1,1 };
	MPI_Aint disp[3];

	// Create MPI user data type for partical
	disp[0] = (char *)&point.coordinatesOfPoints - (char *)&point;
	disp[1] = (char *)point.coordinates - (char *)&point;
	disp[2] = (char *)&point.classification - (char *)&point;

	MPI_Type_create_struct(3, blockLen, disp, type, PointMPIDataType);

	MPI_Type_commit(PointMPIDataType);
}

void printDataAsIs(Point* points, int numberOfPoints, int coordinatesOfPoints, double a0, double aMax, int limit, double qc)
{
	int counter = 0;

	printf("%d %d %f %f %d %f\n", numberOfPoints, coordinatesOfPoints, a0, aMax, limit, qc);
	int i, j;
	for (i = 0; i < numberOfPoints; i++)
	{
		for (j = 0; j < coordinatesOfPoints + 1; j++)
		{
			printf("%f ", points[i].coordinates[j]);
		}
		printf("%d\n", points[i].classification);
		counter++;
	}
	printf("total points %d", counter);
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

void writeWeights(double* w, int size,FILE* file)
{
	int i = 0;
	fprintf(file,"[");
	for (i = 0; i < size; i++)
	{
		fprintf(file," %d ", (int)w[i]);
	}
	fprintf(file,"]\n");
}

double f(double* w, Point* point)
{
	return scalarMult(w, point);
}

void updateWeights(double* last, Point* point, double alpha, int x)
{
	int i;
	for (i = 0; i < point->coordinatesOfPoints + 1; i++)
	{
		last[i] = last[i] + alpha * sign(x)  * (point->coordinates[i]);
	}
}

double FindNmis(double* w, Point* points, int numOfPoints, double* results, int weightSize)
{

	int i, j;
	int notCorrectPoints = 0;

	cudaError_t cudaStatus = calculateWithCuda(w, results, numOfPoints, weightSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	#pragma omp parallel for reduction(+: notCorrectPoints)
	for (i = 0; i < numOfPoints; i++)
	{
		if ((points[i].classification == SET_A && results[i] <= 0) || (points[i].classification == SET_B && results[i] >= 0))
		{
			notCorrectPoints++;
		}
	}

	return (double)notCorrectPoints / (double)numOfPoints;

}

int checkQuality(double q, double qc)
{
	return q <= qc ? 1 : 0;
}

double classified(double* w, Point* point)
{
	double  fx = scalarMult(w, point);

	if (point->classification == SET_A && fx >= 0)
		return fx;

	if (point->classification == SET_B && fx < 0)
		return fx;
	else
		return 0;

}
