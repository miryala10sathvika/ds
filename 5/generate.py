import random
import sys

def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            m[i][j] = sys.maxsize
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k
    
    return m[0][n - 1]

# Function to generate test cases and save them
def generate_test_cases(num_cases=10, min_size=1, max_size=500, filename_prefix="test_case_"):
    for case_num in range(1, num_cases + 1):
        # Generate a random size for the matrix chain
        size = random.randint(min_size, max_size)
        # Generate random dimensions between 1 and 1000 for each matrix
        p = [random.randint(1, 1000) for _ in range(size + 1)]
        
        # Calculate the minimum number of multiplications needed
        solution = matrix_chain_order(p)
        
        # Save the input in a file
        with open(f"{filename_prefix}{case_num}_input.txt", "w") as input_file:
            input_file.write(f"{size}\n")
            input_file.write(" ".join(map(str, p)) + "\n")
        
        # Save the solution in a file
        with open(f"{filename_prefix}{case_num}_output.txt", "w") as output_file:
            output_file.write(f"{solution}\n")

# Generate the test cases
generate_test_cases(num_cases=10)  # Adjust the number of test cases as needed
