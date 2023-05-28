#include<iostream>
#include<omp.h>
#include<vector>

using namespace std;

int parallelMin(vector<int>& arr){
	int minVal = arr[0];
	#pragma omp parallel for reduction(min: minVal)
	for(int i=0; i<arr.size(); i++){
		if(arr[i]<minVal){
			minVal = arr[i];
		}
	}
	return minVal;
}

int parallelMax(vector<int>& arr){
	int maxVal = arr[0];
	#pragma omp parallel for reduction(max: maxVal)
	for(int i=0; i<arr.size(); i++){
		if(arr[i]>maxVal){
			maxVal = arr[i];
		}
	}
	return maxVal;
}

int parallelSum(vector<int>& arr){
	int sum = 0;
	#pragma omp parallel for reduction(+: sum)
	for(int i=0; i<arr.size(); i++){
		sum+=arr[i];
	}
	return sum;
}

double parallelAvg(vector<int>& arr){
	int sum = parallelSum(arr);
	int n = arr.size();
	double avg = static_cast<double>(sum) / n;
    return avg;
}

int main(){
	vector<int> arr = {2, 1, 5, 7, 4, 9, 6, 8, 3, 10};

	cout<<"Minimum: "<<parallelMin(arr)<<endl;
	cout<<"Maximum: "<<parallelMax(arr)<<endl;
	cout<<"Sum: "<<parallelSum(arr)<<endl;
	cout<<"Average: "<<parallelAvg(arr)<<endl;

}


// The reduction operation combines multiple values into a single value by iteratively applying the operation in a tree-like structure. The key idea is to divide the input data into smaller chunks and perform the reduction operation concurrently on these chunks. The intermediate results are then combined until a final result is obtained.

// OpenMP, a popular API for parallel programming in shared-memory systems, provides a reduction clause that simplifies the implementation of parallel reduction. By using the reduction clause in an OpenMP parallel construct, each thread maintains a private copy of a reduction variable and performs the reduction operation on its private copy. Finally, the private copies are combined using the specified reduction operation (e.g., sum, min, max) to obtain the final result.

// #pragma omp: This is a compiler directive that informs the compiler to interpret the following code using OpenMP directives.

// parallel for: This specifies that the loop should be executed in parallel, with iterations divided among multiple threads.

// reduction(min : minVal): This clause specifies the reduction operation to be performed on the variable minVal. In this case, it is a minimum reduction (min). The minVal variable is shared among all threads, and each thread maintains its private copy of minVal. At the end of the parallel loop, the minimum value among all threads is assigned to the shared minVal variable.

// The reduction operation is a way to combine the results of computations performed by multiple threads into a single result. In the case of min, each thread compares its local value of minVal with the current value of minVal shared among all threads, and the minimum value is assigned to the shared variable.

