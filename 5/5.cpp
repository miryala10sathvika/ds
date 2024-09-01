#include <iostream>
#include <vector>
#include <climits>
#include <mpi.h>

using namespace std;

int matrixChainMultiplication(vector<int>& dimensions) {
    int n = dimensions.size() - 1;
    vector<vector<int>> dp(n, vector<int>(n, INT_MAX));

    // Base case: For a single matrix (i == j), the cost is 0
    for (int i = 0; i < n; ++i) {
        dp[i][i] = 0;
    }

    // Fill the dp table in a bottom-up manner
    for (int len = 2; len <= n; ++len) {
        for (int i = 0; i < n - len + 1; ++i) {
            int j = i + len - 1;
            for (int k = i; k < j; ++k) {
                int cost = dp[i][k] + dp[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1];
                dp[i][j] = min(dp[i][j], cost);
            }
        }
    }

    return dp[0][n - 1];
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n;
    vector<vector<int>> dp(n, vector<int>(n, INT_MAX));

    
    vector<int> dimensions;

    // Process 0 reads the input and broadcasts it
    if (rank == 0) {
        cin >> n;
        dimensions.resize(n + 1);
        for (int i = 0; i <= n; ++i) {
            cin >> dimensions[i];
        }
    }

    // Broadcast the input data
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(dimensions.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the diagonal assigned to this process
    int start_row = rank * (n + 1) / size;
    int end_row = (rank + 1) * (n + 1) / size - 1;

    for (int i = start_row; i <= end_row; ++i) {
        for (int j = i + 1; j < n; ++j) {
            for (int k = i; k < j; ++k) {
                int cost = dimensions[i] * dimensions[k + 1] * dimensions[j + 1];
                if (dp[i][k] + dp[k + 1][j] + cost < dp[i][j]) {
                    dp[i][j] = dp[i][k] + dp[k + 1][j] + cost;
                }
            }
        }
    }

    // Gather the results from all processes
    MPI_Gather(dp.data(), n * n, MPI_INT, dp.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

    // Process 0 prints the result
    if (rank == 0) {
        cout << dp[0][n - 1] << endl;
    }

    MPI_Finalize();
    return 0;
}