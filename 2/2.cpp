#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int n = 0, m = 0, k = 0;
    float a, b;
    std::vector<std::vector<int>> answer;

    if (argc < 2) {
        if (world_rank == 0) {
            std::cerr << "No file name provided. Please provide a file name as an argument." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    const char* file_name = argv[1];
    std::ifstream file(file_name);
    if (!file.is_open()) {
        if (world_rank == 0) {
            std::cerr << "Could not open the file: " << file_name << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (world_rank == 0) {
        file >> n >> m >> k >> a >> b;
        if (n <= 0 || m <= 0 || k <= 0) {
            std::cerr << "Invalid dimensions read from the file." << std::endl;
            MPI_Finalize();
            return 1;
        }
        std::cout << "Read dimensions: " << n << " " << m << " " << k << std::endl;
        file.close();
    }
    //start time
    double start_time = MPI_Wtime();
    // Broadcast dimensions and parameters to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&a, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Calculate the number of rows each process will handle
    int rows_per_process = n / world_size;
    int remainder = n % world_size;
    int start_row = world_rank * rows_per_process + std::min(world_rank, remainder);
    int end_row = start_row + rows_per_process + (world_rank < remainder ? 1 : 0);
    int local_rows = end_row - start_row;

    // Create local matrix for each process
    std::vector<std::vector<int>> local_matrix(local_rows, std::vector<int>(m, 0));

    double x_min = -1.5, x_max = 1.5;
    double y_min = -1.5, y_max = 1.5;

    // Generate linearly spaced values for x and y
    std::vector<double> x(m), y(n);
    for (int i = 0; i < m; ++i) {
        x[i] = x_min + (x_max - x_min) * i / (m - 1);
    }
    for (int i = 0; i < n; ++i) {
        y[i] = y_min + (y_max - y_min) * i / (n - 1);
    }

    // Calculate Julia set for local rows
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < m; ++j) {
            std::complex<float> z(x[j], y[start_row + i]);
            int count = 0;
            while (count < k + 3 && std::norm(z) <= 4) {
                z = z * z + std::complex<float>(a, b);
                count++;
            }
            local_matrix[i][j] = (count > k + 1) ? 1 : 0;
        }
    }

    // Prepare for gathering results
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    
    for (int i = 0; i < world_size; ++i) {
        recvcounts[i] = (n / world_size + (i < remainder ? 1 : 0)) * m;
        displs[i] = (i * (n / world_size) + std::min(i, remainder)) * m;
    }

    // Flatten local matrix for gathering
    std::vector<int> local_flattened(local_rows * m);
    for (int i = 0; i < local_rows; ++i) {
        std::copy(local_matrix[i].begin(), local_matrix[i].end(), local_flattened.begin() + i * m);
    }

    // Gather results
    std::vector<int> gathered_result;
    if (world_rank == 0) {
        gathered_result.resize(n * m);
    }

    MPI_Gatherv(local_flattened.data(), local_rows * m, MPI_INT,
                gathered_result.data(), recvcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    //end time
    double end_time = MPI_Wtime();
    if (world_rank == 0) {
        std::cout << "Time taken: " << std::fixed << std::setprecision(6) << end_time - start_time << "seconds." << std::endl;
    }
    // Print the result
    if (world_rank == 0) {
        std::cout << "Julia set calculation complete. Printing matrix..." << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = m - 1; j >= 0; j--) {
                std::cout << gathered_result[i * m + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}