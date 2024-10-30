import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('random_search_results.pkl', 'rb') as f:
    results_1 = pickle.load(f)

with open('random_search_results_2.pkl', 'rb') as f:
    results_2 = pickle.load(f)

with open('random_search_results_continuous.pkl', 'rb') as f:
    results_cont = pickle.load(f)

with open('random_search_results_continuous_2.pkl', 'rb') as f:
    results_cont_2 = pickle.load(f)

results = np.concatenate((results_1, results_2, results_cont, results_cont_2))

variables = ['init_attempts', 'w_add_object', 'w_remove_object', 
             'w_add_obs', 'w_remove_obs', 'w_replace', 'max_iter', 
             'n_sub_iter', 'add_attempts']


for i in range(9):
    plt.errorbar(results[:,i], results[:,9], yerr=results[:,10], fmt='o')
    plt.xlabel('Parameter value')
    plt.ylabel('Mean fill factor')
    plt.title(f'Paramter: {variables[i]}')
    plt.show()

# for i in range(9):
#     for j in range(i+1,9):
#         total_fill_factor = np.zeros((3,3))
#         count = np.zeros((3,3))
#         min_feature_1, max_feature_1 = np.min(results[:,i]), np.max(results[:,i])
#         min_feature_2, max_feature_2 = np.min(results[:,j]), np.max(results[:,j])
#         for sample in results:
#             if sample[i] == min_feature_1:
#                 if sample[j] == min_feature_2:
#                     index_i, index_j = 0,0
#                 elif sample[j] == max_feature_2:
#                     index_i, index_j = 0,2
#                 else:
#                     index_i, index_j = 0,1
#             elif sample[i] == max_feature_1:
#                 if sample[j] == min_feature_2:
#                     index_i, index_j = 2,0
#                 elif sample[j] == max_feature_2:
#                     index_i, index_j = 2,2
#                 else:
#                     index_i, index_j = 2,1
#             else:
#                 if sample[j] == min_feature_2:
#                     index_i, index_j = 1,0
#                 elif sample[j] == max_feature_2:
#                     index_i, index_j = 1,2
#                 else:
#                     index_i, index_j = 1,1
#             total_fill_factor[index_i, index_j] += sample[9]
#             count[index_i, index_j] += 1
#         mean_fill = total_fill_factor/count

#         plt.imshow(mean_fill)
#         plt.xlabel(variables[j])
#         plt.ylabel(variables[i])
#         plt.show()

cor = np.corrcoef(results.T)

plt.imshow(cor)
plt.show()

for sample in results:
    if sample[9] >= 0.88:
        print(sample)
