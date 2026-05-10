#include <iostream>
#include <omp.h>

using namespace std;

// ============================
// Minimum Value
// ============================
int minval(int arr[], int n) {

    int minval = arr[0];

    #pragma omp parallel for reduction(min:minval)
    for (int i = 0; i < n; i++) {

        if (arr[i] < minval)
            minval = arr[i];
    }

    return minval;
}

// ============================
// Maximum Value
// ============================
int maxval(int arr[], int n) {

    int maxval = arr[0];

    #pragma omp parallel for reduction(max:maxval)
    for (int i = 0; i < n; i++) {

        if (arr[i] > maxval)
            maxval = arr[i];
    }

    return maxval;
}

// ============================
// Sum
// ============================
int sum(int arr[], int n) {

    int total = 0;

    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < n; i++) {

        total += arr[i];
    }

    return total;
}

// ============================
// Average
// ============================
double average(int arr[], int n) {

    return (double)sum(arr, n) / n;
}

// ============================
// Main Function
// ============================
int main() {

    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    int arr[n];

    cout << "Enter elements:\n";

    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    cout << "\nMinimum Value = " << minval(arr, n);

    cout << "\nMaximum Value = " << maxval(arr, n);

    cout << "\nSum = " << sum(arr, n);

    cout << "\nAverage = " << average(arr, n);

    cout << endl;

    return 0;
}
