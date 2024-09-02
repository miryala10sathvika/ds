import random

def generate_points(n):
    """Generate n points with each point having 2 random float values."""
    points = []
    for _ in range(n):
        x = round(random.uniform(-100, 100), 2)
        y = round(random.uniform(-100, 100), 2)
        points.append(f"{x} {y}")
    return points

def generate_queries(q):
    """Generate q queries with each query having 2 random float values."""
    queries = []
    for _ in range(q):
        x = round(random.uniform(-100, 100), 2)
        y = round(random.uniform(-100, 100), 2)
        queries.append(f"{x} {y}")
    return queries

# Define the list of n and q values
n_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
q_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
k = 5  # Number of nearest neighbors

# Generate the files
for i, (n, q) in enumerate(zip(n_values, q_values), start=1):
    file_name = f"test_case_{i}.txt"
    with open(file_name, "w") as file:
        # Write the first line with n, q, k
        file.write(f"{n} {q} {k}\n")
        
        # Generate and write the points
        points = generate_points(n)
        for point in points:
            file.write(point + "\n")
        
        # Generate and write the queries
        queries = generate_queries(q)
        for query in queries:
            file.write(query + "\n")

    print(f"Generated {file_name}")
