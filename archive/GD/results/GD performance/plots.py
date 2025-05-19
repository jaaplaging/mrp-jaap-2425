import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_style('whitegrid')

with open('init_fill.pkl', 'rb') as file:
    init_fill = pickle.load(file)

with open('final_fill.pkl', 'rb') as file:
    final_fill = pickle.load(file)

with open('runtime.pkl', 'rb') as file:
    runtime = pickle.load(file)

with open('histories.pkl', 'rb') as file:
    histories = pickle.load(file)


#####################
# Fill factor hists #
#####################

n_bins_final=6
bin_width_init = (np.max(final_fill)-np.min(final_fill))/n_bins_final
n_bins_init = len(set(init_fill))

print(bin_width_init)

sns.histplot(init_fill, bins=20, binrange=(0.5,1.0), label='Initial fill factor')
sns.histplot(init_fill, bins=20, binrange=(0.0,0.5), label='Initial fill factor')
sns.histplot(final_fill, bins=n_bins_final, label='Final fill factor')
plt.ylabel('Number')
plt.xlabel('Fill factor')
plt.xlim(np.min(init_fill)-0.1)
plt.title('Gradient Descent method fill factor')
plt.legend()
plt.savefig('mrp-jaap-2425/GD/results/GD performance/gd_hist.pdf')
plt.show()

################
# Runtime hist #
################
sns.histplot(runtime, label='GD runtime')
plt.ylabel('Number')
plt.xlabel('Runtime')
plt.xlim(0,100)
plt.title('Gradient Descent runtime (seconds)')
plt.savefig('mrp-jaap-2425/GD/results/GD performance/gd-runtime.pdf')
plt.show()

#############################
# Fill factor per iteration #
#############################
df = pd.DataFrame(columns=['Iteration', 'Fill Factor'])

for run_ind, run in enumerate(histories):
    for it_ind, iteration in enumerate(run):
        df = df._append({'Iteration': it_ind, 'Fill Factor': iteration}, ignore_index=True)

plt.subplot(1,2,1)
sns.lineplot(data=df, x='Iteration', y='Fill Factor')
plt.ylabel('Fill factor')
plt.xlabel('Iteration')
plt.xlim(-5,100)

plt.subplot(1,2,2)
sns.lineplot(data=df, x='Iteration', y='Fill Factor')
plt.ylabel('Fill factor')
plt.xlabel('Iteration')
plt.xlim(-50,2000)

plt.tight_layout()
plt.savefig('mrp-jaap-2425/GD/results/GD performance/gd_histories.pdf')
plt.show()

