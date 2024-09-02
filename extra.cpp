# Function to generate test cases and save them
def generate_test_cases(num_cases=10, min_size=1, max_size=500, filename_prefix="test_case_"):
    for case_num in range(0, num_cases ):
        # Generate a random size for the matrix chain
        sizes = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
        size=sizes[case_num]
        # Generate random dimensions between 1 and 1000 for each matrix
        p = [random.randint(1, 1000) for _ in range(size + 1)]
        
        # Save the input in a file
        with open(f"{filename_prefix}{case_num+1}_input.txt", "w") as input_file:
            input_file.write(f"{size}\n")
            input_file.write(" ".join(map(str, p)) + "\n")

# Generate the test cases
generate_test_cases(num_cases=10) 