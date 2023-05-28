#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

using namespace std;

// Sequential merge sort algorithm
void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        vector<int> temp(right - left + 1);
        int i = left, j = mid + 1, k = 0;

        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j])
                temp[k++] = arr[i++];
            else
                temp[k++] = arr[j++];
        }

        while (i <= mid)
            temp[k++] = arr[i++];
        
        while (j <= right)
            temp[k++] = arr[j++];

        for (int p = 0; p < k; p++)
            arr[left + p] = temp[p];
    }
}

// Parallel merge sort algorithm using OpenMP
void parallelMergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, left, mid);

            #pragma omp section
            parallelMergeSort(arr, mid + 1, right);
        }

        vector<int> temp(right - left + 1);
        int i = left, j = mid + 1, k = 0;

        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j])
                temp[k++] = arr[i++];
            else
                temp[k++] = arr[j++];
        }

        while (i <= mid)
            temp[k++] = arr[i++];
        
        while (j <= right)
            temp[k++] = arr[j++];

        for (int p = 0; p < k; p++)
            arr[left + p] = temp[p];
    }
}

int main() {
    int size = 1000000;
    vector<int> arr(size);
    vector<int> arrCopy(size);

    // Initialize the array with random values
    srand(time(0));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 1000;
        arrCopy[i] = arr[i];
    }

    // Sequential merge sort
    clock_t startTime = clock();
    mergeSort(arr, 0, size - 1);
    clock_t endTime = clock();
    double sequentialTime = double(endTime - startTime) / CLOCKS_PER_SEC;

    // Parallel merge sort
    startTime = clock();
    parallelMergeSort(arrCopy, 0, size - 1);
    endTime = clock();
    double parallelTime = double(endTime - startTime) / CLOCKS_PER_SEC;

    // Check if the arrays are sorted correctly
    bool isSorted = is_sorted(arr.begin(), arr.end());
    bool isParallelSorted = is_sorted(arrCopy.begin(), arrCopy.end());

    // Print results
    cout << "Sequential merge sort took " << sequentialTime << " seconds." << endl;
    cout << "Parallel merge sort took " << parallelTime << " seconds." << endl;
    cout << "Sequential merge sort " << (isSorted ? "sorted correctly." : "did not sort correctly.") << endl;
    cout << "Parallel merge sort " << (isParallelSorted ? "sorted correctly." : "did not sort correctly.") << endl;

    return 0;
}
