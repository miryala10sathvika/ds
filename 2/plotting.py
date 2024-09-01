import matplotlib.pyplot as plt
import pandas as pd
import io

# Data in CSV format
data = """
Group,Index,Value
1,1,0.418399
1,2,0.271621
1,3,0.530665
1,4,0.432962
1,5,0.556229
1,6,0.486830
1,7,0.428529
1,8,0.466052
1,9,0.502719
1,10,0.483175
1,11,0.489395
1,12,0.460729
2,1,1.672469
2,2,1.084312
2,3,1.944141
2,4,1.743174
2,5,2.074971
2,6,1.952832
2,7,2.068818
2,8,1.929439
2,9,1.905748
2,10,1.901101
2,11,1.959889
2,12,1.949509
3,1,3.762579
3,2,2.484539
3,3,4.097256
3,4,3.860489
3,5,4.652269
3,6,4.379219
3,7,4.713424
3,8,4.339425
3,9,4.583908
3,10,4.381725
3,11,4.533711
3,12,4.427797
4,1,6.709179
4,2,4.346900
4,3,7.968688
4,4,7.007444
4,5,8.268361
4,6,7.629407
4,7,8.428111
4,8,7.721466
4,9,8.266927
4,10,7.872602
4,11,7.951818
4,12,8.187583
5,1,10.439870
5,2,6.790893
5,3,12.512330
5,4,10.732692
5,5,13.221910
5,6,12.120233
5,7,13.153631
5,8,12.063286
5,9,12.330760
5,10,12.307764
5,11,12.302895
5,12,12.371549
6,1,15.047542
6,2,9.783156
6,3,16.607919
6,4,15.422435
6,5,18.966136
6,6,17.366866
6,7,18.840645
6,8,17.216053
6,9,17.870908
6,10,17.765248
6,11,17.883103
6,12,17.714719
7,1,20.549823
7,2,13.315444
7,3,25.130237
7,4,21.028130
7,5,25.627594
7,6,23.765100
7,7,25.773281
7,8,23.667072
7,9,25.204430
7,10,24.077170
7,11,23.214946
7,12,23.964617
8,1,26.721856
8,2,17.386029
8,3,31.289020
8,4,27.413314
8,5,33.695078
8,6,31.180747
8,7,33.023521
8,8,30.882872
8,9,31.324458
8,10,31.263512
8,11,32.057683
8,12,30.783369
9,1,33.835550
9,2,22.014147
9,3,40.542649
9,4,34.714486
9,5,42.831312
9,6,39.453186
9,7,42.634636
9,8,39.257146
9,9,40.746743
9,10,39.988665
9,11,39.358910
9,12,39.100902
10,1,41.746663
10,2,27.175386
10,3,48.830116
10,4,43.358064
10,5,53.085431
10,6,48.718575
10,7,52.993408
10,8,48.272170
10,9,51.000888
10,10,47.486321
10,11,49.184575
10,12,50.080222
"""

# Read the data into a pandas DataFrame
df = pd.read_csv(io.StringIO(data))

# Plotting
plt.figure(figsize=(12, 8))
for group in df['Group'].unique():
    subset = df[df['Group'] == group]
    plt.plot(subset['Index'], subset['Value'], marker='o', label=f'Group {group}')

plt.xlabel('Number of processes')
plt.ylabel('Value (in seconds)')
plt.title('Line Graph of Values by input values')
plt.legend(loc='upper left')
plt.grid(True)

# Adjust layout to make room for the legend
plt.tight_layout()

plt.show()
