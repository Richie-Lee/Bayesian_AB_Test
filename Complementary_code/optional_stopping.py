import pandas as pd

aaa_results_grouped_by_sample = []

# Assuming all tests share the sample length (fixed horizon), when disabling early stopping
for i in range(len(results_interim_tests[0])):
    # Initialise
    bfs_at_time_t = []
    
    # Collect all bfs
    for test in results_interim_tests:
        bfs_at_time_t.append(test[i])
    
    # Store
    aaa_results_grouped_by_sample.append(bfs_at_time_t)

# Remove sample's -> only keep bfs (need to track)

aaa_samples = [a for a, b in results_interim_tests[0]]

for t in aaa_results_grouped_by_sample:
    t[:] = [b for a, b in t]


# get summarised results (fixed horizon)
aaa_summary = []
for i in range(len(aaa_samples)):
    t = aaa_samples[i]    
    count_reject = sum(1 for element in aaa_results_grouped_by_sample[i] if element < 1/19)
    count_accept = sum(1 for element in aaa_results_grouped_by_sample[i] if element > 19)
    count_inconclusive = len(aaa_results_grouped_by_sample[i]) - count_reject - count_accept

    aaa_summary.append([t, count_reject, count_accept, count_inconclusive])

aaa_summary = pd.DataFrame(aaa_summary, columns = ["n", "H0", "H1", "inconclusive"])


import matplotlib.pyplot as plt

def plot_dataframe(df):
    plt.figure(figsize=(10, 6))
    
    # Plot each column
    # plt.plot(df['n'], df['H0'], label='H0')
    plt.plot(df['n'], df['H1'], label='H1')
    # plt.plot(df['n'], df['inconclusive'], label='Inconclusive')

    # Adding labels and title
    plt.xlabel('n')
    plt.ylabel('outcome')
    plt.title('BF over time')
    
    # Show legend
    plt.legend()

    # Show the plot
    plt.show()

# Assuming df is your DataFrame
plot_dataframe(aaa_summary)
