#!/bin/bash

# Load necessary modules
module load openmpi

# Compile the MPI program
mpicxx -o mpi_4 mpi_4.cpp  # Compile mpi_4.cpp into an executable named mpi_4

# Create or clear the output file
output_file="output.txt"
echo "TestCase Processes TimeTaken" > 

# Loop through test cases (test_case_1.txt to test_case_10.txt)
for testcase in {1..10}
do
    # Loop through the number of processes (1 to 12)
    for np in {1..12}
    do
        echo "Running test_case_${testcase}.txt with -np $np"
        
        # Use the time command to measure the real time of the execution
        start_time=$(date +%s%N)  # Start time in nanoseconds
        
        # Run the MPI program and append output to output.txt
        mpirun --oversubscribe -np $np ./mpi_4 ./test_case_${testcase}.txt >> output_${testcase}_${np}.txt
        
        # Calculate and store the time taken
        end_time=$(date +%s%N)  # End time in nanoseconds
        time_taken=$((($end_time - $start_time)/1000000))  # Time in milliseconds
        
        # Echo the results 
        echo "test_case_${testcase}.txt $np $time_taken ms" 
        
    done
done

# Print completion message
echo "All jobs completed"
