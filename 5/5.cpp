#include <iostream>
#include <mpi.h>
#include <vector>
#include <fstream>

const unsigned long long MAXI = 18446744073709551615ULL;


void singleProcess(unsigned long long &level, unsigned long long &N, std::vector<unsigned long long> &matric_dim, std::vector<std::vector<unsigned long long>> &dp) {
    for (unsigned long long offset = level; offset < N; offset++) {
        for (unsigned long long i = 0; i + offset < N; i++) {
            unsigned long long j = i + offset;
            for (unsigned long long k = i; k < j; k++) {
                if (dp[i][j] == 0) dp[i][j] = MAXI;
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
    std::vector<std::vector<unsigned long long>> dp(N, std::vector<unsigned long long>(N, 0));
    MPI_Bcast(matric_dim.data(), N + 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    for (offset = 1; offset < N; offset++) {
        unsigned long long num_elements = N - offset;
        if (num_elements < size) break;
        unsigned long long chunk_sz = num_elements / size;
        unsigned long long extra = num_elements % size;
        unsigned long long start = rank * chunk_sz;
        unsigned long long end = (rank + 1) * chunk_sz + (rank == size - 1 ? extra : 0);
        std::vector<unsigned long long> ans(num_elements);
        std::vector<unsigned long long> sendbuf(end - start);
        std::vector<int> counts(size);
        std::vector<int> displacements(size);
        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                counts[i] = chunk_sz + (i == size - 1 ? extra : 0);
                displacements[i] = i * chunk_sz;
            }
        }
        for (unsigned long long i = start; i < end; i++) {
            unsigned long long j = i + offset;
            for (unsigned long long k = i; k < j; k++) {
                if (dp[i][j] == 0) dp[i][j] = MAXI;
                dp[i][j] = std::min(dp[i][j], dp[i][k] + dp[k + 1][j] + matric_dim[i] * matric_dim[k + 1] * matric_dim[j + 1]);
            }
            sendbuf[i - start] = dp[i][i + offset];
        }
        MPI_Gatherv(sendbuf.data(), end - start, MPI_UNSIGNED_LONG_LONG, ans.data(), counts.data(), displacements.data(), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(ans.data(), num_elements, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        for (unsigned long long i = 0; i < num_elements; i++) {
            dp[i][i + offset] = ans[i];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        singleProcess(offset, N, matric_dim, dp); 
    }
    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << dp[0][N - 1] << "\n";
        std::cout << "Time taken: " << end_time - start_time <<"seconds." <<"\n";
    }
    MPI_Finalize();
    return 0;
}
