import pickle
import sys
import os
sys.path.append('mrp-jaap-2425/')

for file_str in os.listdir('mrp-jaap-2425/modular/results/'):
    try:
        with open(f'mrp-jaap-2425/modular/results/{file_str}', 'rb') as f:
            result = pickle.load(f)
            print(result[0].batch_size)
    except:
        pass

