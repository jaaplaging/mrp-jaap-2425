import pickle
import numpy as np
import matplotlib.pyplot as plt

DIR = 'C:/Users/sevij/Documents/Studie/Master/MRP/code/mrp-jaap-2425/results/'

with open(DIR + 'new_results.pkl', 'rb') as f:
    results_1 = pickle.load(f)

with open(DIR + 'new_results_2.pkl', 'rb') as f:
    results_2 = pickle.load(f)

with open(DIR + 'new_results_3.pkl', 'rb') as f:
    results_3 = pickle.load(f)

with open(DIR + 'new_results_4.pkl', 'rb') as f:
    results_4 = pickle.load(f)

results = np.concatenate((results_1, results_2, results_3, results_4))

variables = ['init_attempts', 'w_add_object', 'w_remove_object', 
             'w_add_obs', 'w_remove_obs', 'w_replace', 'max_iter', 
             'n_sub_iter', 'add_attempts']


for i in range(9):

    sorted_ind = np.argsort(results[:,i])
    results_sorted = results[sorted_ind]
    mean_x = []
    mean_y = []
    std_y = []
    for j in range(len(results)-20):
        mean_x.append(np.mean(results_sorted[j:j+20,i]))
        mean_y.append(np.mean(results_sorted[j:j+20,9]))
        std_y.append(np.std(results_sorted[j:j+20,9]))

    plt.scatter(results[:,i], results[:,9])
    plt.plot(mean_x, mean_y, color='red')
    plt.plot(mean_x, np.array(mean_y)+np.array(std_y), color='orange')
    plt.plot(mean_x, np.array(mean_y)-np.array(std_y), color='orange')
    plt.xlabel('Parameter value')
    plt.ylabel('Fill factor')
    plt.title(f'Parameter: {variables[i]}')
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
    if sample[9] >= 0.92:
        print(sample)
