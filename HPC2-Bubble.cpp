#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

// Sequential Bubble Sort
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool swapped;

    for (int i = 0; i < n - 1; ++i) {
        swapped = false;

        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

// Parallel Bubble Sort using OpenMP
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool swapped;

    #pragma omp parallel
    {
        for (int i = 0; i < n - 1; ++i) {
            swapped = false;

            #pragma omp for
            for (int j = 0; j < n - i - 1; ++j) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr[j], arr[j + 1]);
                    swapped = true;
                }
            }

            #pragma omp barrier

            if (!swapped)
                break;
        }
    }
}

int main() {
    vector<int> arr = {9, 7, 1, 3, 5, 8, 2, 6, 4};
    int n = arr.size();

    // Sequential Bubble Sort
    vector<int> seqArr = arr;  // Copy of original array
    auto start = chrono::steady_clock::now();
    sequentialBubbleSort(seqArr);
    auto end = chrono::steady_clock::now();
    chrono::duration<double> seqTime = end - start;

    // Parallel Bubble Sort
    vector<int> parallelArr = arr;  // Copy of original array
    start = chrono::steady_clock::now();
    parallelBubbleSort(parallelArr);
    end = chrono::steady_clock::now();
    chrono::duration<double> parallelTime = end - start;

    cout << "Original Array: ";
    for (int num : arr)
        cout << num << " ";
    cout << endl;

    cout << "Sequential Sorted Array: ";
    for (int num : seqArr)
        cout << num << " ";
    cout << endl;

    cout << "Parallel Sorted Array: ";
    for (int num : parallelArr)
        cout << num << " ";
    cout << endl;

    cout << "Sequential Bubble Sort Time: " << seqTime.count() << " seconds" << endl;
    cout << "Parallel Bubble Sort Time: " << parallelTime.count() << " seconds" << endl;

    return 0;
}
