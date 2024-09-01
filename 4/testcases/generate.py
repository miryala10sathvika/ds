import numpy as np
import sys

def generate_non_singular_matrix(size):
    while True:
        matrix = np.round(np.random.rand(size, size) * 10 - 5, 2)
        if np.linalg.det(matrix) != 0:
            inverse_matrix = np.round(np.linalg.inv(matrix), 2)
            return matrix, inverse_matrix

def save_matrix_to_file(matrix, filename):
    with open(filename, 'w') as f:
        size = matrix.shape[0]
        if filename != "expected.txt":
            f.write(f"{size}\n")
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    # Loop through the desired sizes
    for size in range(100, 1100, 100):
        # Generate the non-singular matrix and its inverse
        matrix, inverse_matrix = generate_non_singular_matrix(size)

        # Define the filenames
        matrix_filename = f"test_case_{size}.txt"
        inverse_filename = f"expected_{size}.txt"

        # Save the generated matrix to the file
        save_matrix_to_file(matrix, matrix_filename)

        # Save the inverse matrix to the file
        save_matrix_to_file(inverse_matrix, inverse_filename)

        print(f"Generated matrix and inverse for size {size} saved to {matrix_filename} and {inverse_filename}.")