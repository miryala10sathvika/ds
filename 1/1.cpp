#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

// Function to find the K nearest neighbors of a query point using a min-heap
void KNearestNeighbors(float* point_coordinates_x, float* point_coordinates_y, int* point_indices, int total_points, float query_point_x, float query_point_y, int num_neighbors, int* nearest_neighbor_indices) {
    // Min-heap (priority queue) to store distances and indices
    priority_queue<pair<float, int>> distance_heap;

    for (int point_index = 0; point_index < total_points; point_index++) {
        // Calculate the Euclidean distance between the query point and the current point
        float distance = sqrt((point_coordinates_x[point_index] - query_point_x) * (point_coordinates_x[point_index] - query_point_x) + 
                              (point_coordinates_y[point_index] - query_point_y) * (point_coordinates_y[point_index] - query_point_y));
        
        // Push the distance and index as a pair into the heap
        distance_heap.push(make_pair(distance, point_indices[point_index]));
        
        // If the heap size exceeds num_neighbors, remove the farthest element
        if (distance_heap.size() > num_neighbors) {
            distance_heap.pop();
        }
    }

    // Extract the results from the heap and store them in nearest_neighbor_indices
    vector<int> temp_indices;
    while (!distance_heap.empty()) {
        temp_indices.push_back(distance_heap.top().second);
        distance_heap.pop();
    }

    // The nearest_neighbor_indices array will be in reverse order, so reverse it
    reverse(temp_indices.begin(), temp_indices.end());
    copy(temp_indices.begin(), temp_indices.end(), nearest_neighbor_indices);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int process_rank, total_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    if (total_processes < 1 || total_processes > 12) {
        if (process_rank == 0) {
            cerr << "Number of processes should be between 1 and 12." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int num_points, num_queries, num_neighbors;

    // File input handling
    if (process_rank == 0) {
        ifstream input_file("input.txt");
        if (!input_file) {
            cerr << "Unable to open file" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        input_file >> num_points >> num_queries >> num_neighbors;
        input_file.close();
    }

    // Broadcast num_points, num_queries, and num_neighbors to all processes
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_queries, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_neighbors, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Arrays for points and queries
    float *point_coordinates_x = new float[num_points];
    float *point_coordinates_y = new float[num_points];
    int *point_indices = new int[num_points];
    float *query_coordinates_x = new float[num_queries];
    float *query_coordinates_y = new float[num_queries];

    if (process_rank == 0) {
        ifstream input_file("input.txt");
        input_file.ignore(numeric_limits<streamsize>::max(), '\n'); // Skip num_points, num_queries, num_neighbors line
        for (int point_index = 0; point_index < num_points; point_index++) {
            input_file >> point_coordinates_x[point_index] >> point_coordinates_y[point_index];
            point_indices[point_index] = point_index; // Store the original index of the point
        }
        for (int query_index = 0; query_index < num_queries; query_index++) {
            input_file >> query_coordinates_x[query_index] >> query_coordinates_y[query_index];
        }
        input_file.close();
    }
    
    double start_time = MPI_Wtime();
    // Broadcast points & queries to all processes
    MPI_Bcast(point_coordinates_x, num_points, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(point_coordinates_y, num_points, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(point_indices, num_points, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(query_coordinates_x, num_queries, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(query_coordinates_y, num_queries, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Start timing


    // Distribute queries among processes
    int queries_per_process = num_queries / total_processes;
    int start_query_index = process_rank * queries_per_process + min(process_rank, num_queries % total_processes);
    int end_query_index = start_query_index + queries_per_process + (process_rank < num_queries % total_processes);

    int *local_knn_results = new int[queries_per_process * num_neighbors];
    int result_position = 0;

    for (int query_index = start_query_index; query_index < end_query_index; ++query_index) {
        KNearestNeighbors(point_coordinates_x, point_coordinates_y, point_indices, num_points, query_coordinates_x[query_index], query_coordinates_y[query_index], num_neighbors, &local_knn_results[result_position]);
        result_position += num_neighbors;
    }

    vector<int> query_counts_per_process(total_processes);
    vector<int> displacement_indices(total_processes);

    int local_result_count = result_position;
    MPI_Gather(&local_result_count, 1, MPI_INT, query_counts_per_process.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (process_rank == 0) {
        displacement_indices[0] = 0;
        for (int process_index = 1; process_index < total_processes; process_index++) {
            displacement_indices[process_index] = displacement_indices[process_index - 1] + query_counts_per_process[process_index - 1];
        }
    }

    int *global_knn_results = nullptr;
    if (process_rank == 0) {
        global_knn_results = new int[num_queries * num_neighbors];
    }

    // Gather all results from processes
    MPI_Gatherv(local_knn_results, local_result_count, MPI_INT, global_knn_results, query_counts_per_process.data(), displacement_indices.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // End timing
    double end_time = MPI_Wtime();

    // Print results and timing
    if (process_rank == 0) {
        for (int query_index = 0; query_index < num_queries; query_index++) {
            for (int neighbor_index = 0; neighbor_index < num_neighbors; neighbor_index++) {
                int point_index = global_knn_results[query_index * num_neighbors + neighbor_index];
                //cout << point_coordinates_x[point_index] << " " << point_coordinates_y[point_index] << endl;
            }
        }
        cout << "Time taken: " << (end_time - start_time) << "seconds." << endl;
        delete[] global_knn_results;
    }

    delete[] point_coordinates_x;
    delete[] point_coordinates_y;
    delete[] point_indices;
    delete[] query_coordinates_x;
    delete[] query_coordinates_y;
    delete[] local_knn_results;

    MPI_Finalize();
    return 0;
}
