#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <mpi.h>
#include <fstream>

void gather_result(const std::vector<float>& local_I_flat, std::vector<float>& I_flat, 
                   int N, int world_size, int world_rank, 
                   const std::vector<int>& send_counts, const std::vector<int>& displs) {
    if (world_rank == 0) {
        I_flat.resize(N * N);
    }
    MPI_Gatherv(local_I_flat.data(), send_counts[world_rank], MPI_FLOAT,
                I_flat.data(), send_counts.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);
}

void performPivotUpdate(
    std::vector<float>& local_A_flat,
    std::vector<float>& local_I_flat,
    std::vector<float>& pivot_row_A,
    std::vector<float>& pivot_row_I,
    int i, int start, int end, int N, int rows)
{
    // Normalize the pivot row
    float pivot = pivot_row_A[i];
    for (int j = 0; j < N; j++) {
        pivot_row_I[j] /= pivot;
        pivot_row_A[j] /= pivot;
    }

    // Update the matrix rows
    for (int local_row = 0; local_row < rows; local_row++) {
        float factor = local_A_flat[local_row * N + i];
        if (local_row == i - start) {
            continue;
        }
        for (int j = 0; j < N; j++) {
            local_I_flat[local_row * N + j] -= factor * pivot_row_I[j];
            local_A_flat[local_row * N + j] -= factor * pivot_row_A[j];
        }
    }

    // Update local matrices if within range
    if (i >= start && i < end) {
        int local_pivot_row = i - start;
        std::copy(pivot_row_A.begin(), pivot_row_A.end(), 
                  local_A_flat.begin() + local_pivot_row * N);
        std::copy(pivot_row_I.begin(), pivot_row_I.end(), 
                  local_I_flat.begin() + local_pivot_row * N);
    }
}

struct PivotInfo {
    int valid_pivot;
    int pivot_owner;
};

PivotInfo get_pivot_info(int world_rank, const std::vector<int>& displs, int i, 
                         int start, const std::vector<float>& local_A_flat, 
                         int N, int world_size) {
    // Determine the pivot owner based on the index i
    int pivot_owner = (i >= displs[world_size - 1] / N) ? world_size - 1 : -1;
    for (int j = 0; j < world_size - 1; j++) {
        if (i >= (displs[j] / N) && i < (displs[j + 1] / N)) {
            pivot_owner = j;
            break;
        }
    }

    // Check if the pivot is valid
    int valid_pivot = 0;
    if (world_rank == pivot_owner) {
        int local_row = i - start;
        float pivot = local_A_flat[local_row * N + i];
        valid_pivot = std::abs(pivot) >= 1e-8 ? 1 : 0;
    }

    return {valid_pivot, pivot_owner};
}




void print_matrix(const std::vector<float>& matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::fixed << std::setprecision(2) << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to find a valid pivot row
bool find_valid_pivot_row(int i, const std::vector<float>& local_A_flat, 
                           int N, int rows, int& local_pivot_row, 
                           int& new_pivot_owner, int world_rank) {
    for (int j = 0; j < rows; ++j) {
        if (std::abs(local_A_flat[j * N + i]) >= 1e-8) {
            local_pivot_row = j;
            new_pivot_owner = world_rank;
            return true;
        }
    }
    return false;
}

void distribute_matrix(const std::vector<float>& A_flat, std::vector<float>& local_A_flat, 
                       int N, int world_size, int world_rank, 
                       std::vector<int>& send_counts, std::vector<int>& displs) {
    int rows_per_process = N / world_size;
    int remainder = N % world_size;

    for (int i = 0; i < world_size; ++i) {
        send_counts[i] = (rows_per_process + (i < remainder ? 1 : 0)) * N;
        displs[i] = (i > 0) ? displs[i-1] + send_counts[i-1] : 0;
    }

    int local_rows = send_counts[world_rank] / N;
    local_A_flat.resize(local_rows * N);

    MPI_Scatterv(A_flat.data(), send_counts.data(), displs.data(), MPI_FLOAT,
                 local_A_flat.data(), local_rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void initialize_identity(std::vector<float>& local_I_flat, std::vector<float>& I_flat,int N, int start, int local_rows) {
    local_I_flat.assign(local_rows * N, 0.0);
    for (int i = 0; i < local_rows; ++i) {
        local_I_flat[i * N + (i + start)] = 1.0;
    }
    I_flat.resize(N * N);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int N;
    std::vector<float> A_flat;
    const char* file_name = argv[1];
    std::ifstream file(file_name);
    
    if (world_rank == 0) {
    if (!file.is_open()) {
        if (world_rank == 0) {
            std::cerr << "Could not open the file: " << file_name << std::endl;
        }
        MPI_Finalize();
        return 1;
        }
        file >> N;
        A_flat.resize(N * N);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                file >> A_flat[i * N + j];
            }
        }
        std::vector<std::vector<float>> A(N, std::vector<float>(N));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = A_flat[i * N + j];
            }
        }
    }
    // start time
    double start_time = MPI_Wtime();
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // divide rows
    int rows_per_process = N / world_size;
    int remainder = N % world_size;

    int start = world_rank * rows_per_process + std::min(world_rank, remainder);
    int end = start + rows_per_process + (world_rank < remainder ? 1 : 0);
    int rows = end - start;

    std::vector<int> send_counts(world_size);
    std::vector<int> displs(world_size, 0);

    MPI_Gather(&rows, 1, MPI_INT, send_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(send_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<float> local_A_flat(rows * N);
    std::vector<float> local_I_flat(rows * N, 0.0);
    std::vector<float> I_flat;

    distribute_matrix(A_flat, local_A_flat, N, world_size, world_rank, send_counts, displs);
    initialize_identity(local_I_flat,I_flat, N, displs[world_rank] / N, send_counts[world_rank] / N);
    for (int i = 0; i < N; i++) {
        PivotInfo pivot_info = get_pivot_info(world_rank, displs, i, start, local_A_flat, N, world_size);
        int valid_pivot = pivot_info.valid_pivot;
        int pivot_owner = pivot_info.pivot_owner;
        // int valid_pivot = is_valid_pivot(world_rank, pivot_owner, i, start, local_A_flat, N);
        MPI_Bcast(&valid_pivot, 1, MPI_INT, pivot_owner, MPI_COMM_WORLD);
        std::vector<float> pivot_row_A(N), pivot_row_I(N);
        if (valid_pivot) {
            if (world_rank == pivot_owner) {
                int local_row = i - start;

                MPI_Bcast(&local_A_flat[local_row * N], N, MPI_FLOAT, pivot_owner, MPI_COMM_WORLD);
                MPI_Bcast(&local_I_flat[local_row * N], N, MPI_FLOAT, pivot_owner, MPI_COMM_WORLD);

                std::copy(local_A_flat.begin() + local_row * N, local_A_flat.begin() + (local_row + 1) * N, pivot_row_A.begin());
                std::copy(local_I_flat.begin() + local_row * N, local_I_flat.begin() + (local_row + 1) * N, pivot_row_I.begin());
            } else {
                MPI_Bcast(pivot_row_A.data(), N, MPI_FLOAT, pivot_owner, MPI_COMM_WORLD);
                MPI_Bcast(pivot_row_I.data(), N, MPI_FLOAT, pivot_owner, MPI_COMM_WORLD);
            }
        } else {
            int local_pivot_row = -1;
            int new_pivot_owner = -1;
            bool found_pivot = find_valid_pivot_row(i, local_A_flat, N, rows, local_pivot_row, new_pivot_owner, world_rank);
            int local_result[2], global_result[2];
            local_result[0] = found_pivot;
            local_result[1] = world_rank;
            MPI_Allreduce(&local_result, &global_result, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
            if (global_result[0]) {
                new_pivot_owner = global_result[1];
                // Broadcasting pivot row data if this process is the new pivot owner
                if (world_rank == new_pivot_owner) {
                    // Send the pivot row
                    std::copy(local_A_flat.begin() + local_pivot_row * N, 
                            local_A_flat.begin() + (local_pivot_row + 1) * N, 
                            pivot_row_A.begin());
                    std::copy(local_I_flat.begin() + local_pivot_row * N, 
                            local_I_flat.begin() + (local_pivot_row + 1) * N, 
                            pivot_row_I.begin());
                    MPI_Bcast(local_A_flat.data() + local_pivot_row * N, N, MPI_FLOAT, new_pivot_owner, MPI_COMM_WORLD);
                    MPI_Bcast(local_I_flat.data() + local_pivot_row * N, N, MPI_FLOAT, new_pivot_owner, MPI_COMM_WORLD);
                } else {
                    // Receive the pivot row data if not the new pivot owner
                    MPI_Bcast(pivot_row_A.data(), N, MPI_FLOAT, new_pivot_owner, MPI_COMM_WORLD);
                    MPI_Bcast(pivot_row_I.data(), N, MPI_FLOAT, new_pivot_owner, MPI_COMM_WORLD);
                }

                // Send the local row if within the range
                if (i >= start && i < end) {
                    int local_row = i - start;
                    MPI_Send(local_A_flat.data() + local_row * N, N, MPI_FLOAT, new_pivot_owner, 0, MPI_COMM_WORLD);
                    MPI_Send(local_I_flat.data() + local_row * N, N, MPI_FLOAT, new_pivot_owner, 1, MPI_COMM_WORLD);
                }
                // Update local data if within the range
                if (i >= start && i < end) {
                    int local_row = i - start;
                    std::copy(pivot_row_A.begin(), pivot_row_A.end(), local_A_flat.begin() + local_row * N);
                    std::copy(pivot_row_I.begin(), pivot_row_I.end(), local_I_flat.begin() + local_row * N);
                }
            }

        }
        performPivotUpdate(local_A_flat, local_I_flat, pivot_row_A, pivot_row_I, i, start, end, N, rows);
    }
    gather_result(local_I_flat, I_flat, N, world_size, world_rank, send_counts, displs);
        if (world_rank == 0) {
            print_matrix(I_flat, N);
        }
    // end time
    double end_time = MPI_Wtime();
    if (world_rank == 0) {
        std::cout << "Time taken: " << end_time - start_time << " seconds." << std::endl;
    }
    MPI_Finalize();
    return 0;
}