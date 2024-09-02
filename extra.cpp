#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = 0; // Matrix size
    int cols_per_worker = 0;
    int extra_cols = 0;
    std::vector<std::vector<float>> matrix_a, matrix_i;
    std::vector<int> cols_process;

    if (world_rank == 0) {
        // Initialize matrices and distribute work
        // ... (Keep the initialization code from the original snippet)

        n = /* set the matrix size */;
        cols_per_worker = n / (world_size - 1);
        extra_cols = n % (world_size - 1);
        matrix_a = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0));
        matrix_i = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0));
        cols_process = std::vector<int>(n, 0);

        // Initialize matrix_a and matrix_i (identity matrix)
        // ...

        for (int i = 1; i < world_size; ++i) {
            int start = (i - 1) * cols_per_worker;
            int end = start + cols_per_worker;
            if (i == world_size - 1) {
                end += extra_cols;
            }

            int num_cols = end - start;
            std::vector<float> send_buffer(num_cols * n, 0.0);

            // Send matrix_a columns
            for (int col = start; col < end; ++col) {
                for (int row = 0; row < n; ++row) {
                    send_buffer[(col - start) * n + row] = matrix_a[row][col];
                }
                cols_process[col] = i;
            }

            MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&num_cols, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(send_buffer.data(), num_cols * n, MPI_FLOAT, i, 1, MPI_COMM_WORLD);

            // Send matrix_i columns
            for (int col = start; col < end; ++col) {
                for (int row = 0; row < n; ++row) {
                    send_buffer[(col - start) * n + row] = matrix_i[row][col];
                }
            }
            MPI_Send(send_buffer.data(), num_cols * n, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
        }
    } else {
        // Worker processes
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int num_cols;
        MPI_Recv(&num_cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<float> recv_buffer(num_cols * n, 0.0);

        // Receive matrix_a columns
        MPI_Recv(recv_buffer.data(), num_cols * n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        matrix_a = std::vector<std::vector<float>>(n, std::vector<float>(num_cols, 0.0));
        for (int i = 0; i < num_cols; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix_a[j][i] = recv_buffer[i * n + j];
            }
        }

        // Receive matrix_i columns
        MPI_Recv(recv_buffer.data(), num_cols * n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        matrix_i = std::vector<std::vector<float>>(n, std::vector<float>(num_cols, 0.0));
        for (int i = 0; i < num_cols; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix_i[j][i] = recv_buffer[i * n + j];
            }
        }
    }

    // Perform Gaussian elimination
    
    // Gather the results (implementation depends on how you want to collect and use the inverted matrix)
    // ...

    MPI_Finalize();
    return 0;
}