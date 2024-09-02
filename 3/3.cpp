#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

// Function to compute local prefix sum
vector<double> compute_process(vector<double>& chunk) {
    if (chunk.empty()) {
        return chunk;  // Return empty vector if chunk is empty
    }
    
    for (size_t i = 1; i < chunk.size(); ++i) {
        chunk[i] += chunk[i - 1];
    }
    return chunk;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    long long N;
    vector<double> input_arr;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // File input handling
    if (rank == 0) {
        ifstream infile("input.txt");
        if (!infile) {
            cerr << "Unable to open file" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        infile >> N;
        input_arr.resize(N);
        for (long long i = 0; i < N; ++i) {
            infile >> input_arr[i];  
        }
        infile.close();
    }

    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    long long window = (N + size - 1) / size;  // Each process gets at most window elements

    // Manually send chunks from root to all other processes
    vector<double> combined(window);  
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            long long start = i * window;
            long long end = min(N, (i + 1) * window);
            long long count = end - start;

            if (count > 0) {
                MPI_Send(&input_arr[start], count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
        }
        long long local_size = min(window, N);
        std::copy(input_arr.begin(), input_arr.begin() + local_size, combined.begin());

    } else {
        long long start = rank * window;
        long long end = min(N, (rank + 1) * window);
        long long count = end - start;

        if (count > 0) {
            MPI_Recv(combined.data(), count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    vector<double> process = compute_process(combined);

    double start_time = MPI_Wtime();

    double temp_var;
    if (process.empty()) {
        temp_var = 0.0;
    } else {
        temp_var = process.back();
    }
    vector<double> p_sum(size, 0.0);
    
    MPI_Gather(&temp_var, 1, MPI_DOUBLE, p_sum.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            p_sum[i] = p_sum[i] + p_sum[i - 1];
        }
    }

    MPI_Bcast(p_sum.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank > 0 && !process.empty()) {
        for (long long i = 0; i < process.size(); ++i) {
            process[i] = process[i] + p_sum[rank - 1];
        }
    }

    if (rank == 0) {
        for (long long i = 0; i < process.size(); ++i) {
            // cout << process[i] << " ";
        }

        for (int i = 1; i < size; ++i) {
            long long start = i * window;
            long long end = min(N, (i + 1) * window);
            long long count = end - start;

            if (count > 0) {
                vector<double> final(count);
                MPI_Recv(final.data(), count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // for (long long i = 0; i < final.size(); ++i) {
                //     cout << final[i] << " ";
                // }

            }
        }

        // cout << endl;
    } else {
        if (!process.empty()) MPI_Send(process.data(), process.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        
    }

    // End timing
    double end_time = MPI_Wtime();
    if (rank == 0) {
        cout << "Time taken: " << (end_time - start_time) << "seconds." << endl;
    }

    MPI_Finalize();
    return 0;
}
