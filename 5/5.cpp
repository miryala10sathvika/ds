#include <iostream>
<<<<<<< HEAD
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
=======
#include <mpi.h>
#include <vector>
#include <fstream>

const unsigned long long MAXI = 18446744073709551615ULL;


void singleProcess(unsigned long long &level, unsigned long long &N, std::vector<unsigned long long> &matric_dim, std::vector<std::vector<unsigned long long>> &dp) {
    for (unsigned long long offset = level; offset < N; offset++) {
        for (unsigned long long i = 0; i + offset < N; i++) {
            unsigned long long j = i + offset;
            for (unsigned long long k = i; k < j; k++) {
                dp[i][j] = std::min(dp[i][j], dp[i][k] + dp[k + 1][j] + matric_dim[i] * matric_dim[k + 1] * matric_dim[j + 1]);
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    unsigned long long offset;
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *file_name = argv[1];
    std::ifstream file;
    unsigned long long N;
    std::vector<unsigned long long> matric_dim;
    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        file.open(file_name);
        if (!file.is_open()) {
            std::cerr << "Could not open the file: " << file_name << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        file >> N;
        matric_dim.resize(N + 1);
        for (unsigned long long i = 0; i < N + 1; i++) {
            file >> matric_dim[i];
        }
        file.close();
    }    

    // start time
    double start_time = MPI_Wtime();
    // Broadcasting the value of N to all processes
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    matric_dim.resize(N + 1);
    std::vector<std::vector<unsigned long long>> dp(N, std::vector<unsigned long long>(N, MAXI));
    MPI_Bcast(matric_dim.data(), N + 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    for (unsigned long long i = 0; i < N; i++) dp[i][i] = 0;

    for (offset = 1; offset < N; offset++) {
        unsigned long long num_elements = N - offset;
        if (num_elements < size) break;
        unsigned long long chunk_sz = num_elements / size;
        unsigned long long extra = num_elements % size;
        unsigned long long start = rank * chunk_sz;
        unsigned long long end = (rank + 1) * chunk_sz + (rank == size - 1 ? extra : 0);
        std::vector<unsigned long long> ans(num_elements);

        for (unsigned long long i = start; i < end; i++) {
            unsigned long long j = i + offset;
            for (unsigned long long k = i; k < j; k++) {
                dp[i][j] = std::min(dp[i][j], dp[i][k] + dp[k + 1][j] + matric_dim[i] * matric_dim[k + 1] * matric_dim[j + 1]);
            }
        }

        std::vector<unsigned long long> sendbuf(end - start);
        for (unsigned long long i = start; i < end; i++) {
            sendbuf[i - start] = dp[i][i + offset];
        }

        std::vector<int> counts(size);
        std::vector<int> displacements(size);
        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                counts[i] = chunk_sz + (i == size - 1 ? extra : 0);
                displacements[i] = i * chunk_sz;
            }
        }

        MPI_Gatherv(sendbuf.data(), end - start, MPI_UNSIGNED_LONG_LONG, ans.data(), counts.data(), displacements.data(), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

        MPI_Bcast(ans.data(), num_elements, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        for (unsigned long long i = 0; i < num_elements; i++) {
            dp[i][i + offset] = ans[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        singleProcess(offset, N, matric_dim, dp); // remaining table entries
    }
    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << dp[0][N - 1] << "\n";
        std::cerr << "Time: " << end_time - start_time << "s\n";
    }
    MPI_Finalize();
    return 0;
}
>>>>>>> 5239088 (haha)
