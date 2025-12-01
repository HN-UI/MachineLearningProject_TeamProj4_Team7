import os
from collections import defaultdict
import numpy as np
test = True
data='snp'
models = os.listdir(f'./logs/{data}')
for model in models:
    files = os.listdir(f'./logs/{data}/{model}')
    hype_dict = defaultdict(list)
    for seed in [10,20,30]:
        for file in files:
            if test and 'drop' not in file and 'aug' not in file:
                with open(f'./logs/{data}/{model}/{file}', 'r') as f:
                    for line in f:
                        if '[Test]  metric (with vol-cap): ' in line:
                            line = line.strip()
                            met = float(line.split('[Test]  metric (with vol-cap): ')[-1])
                            hype_dict[0].append(met)
            if f'{seed}_' in file:
                with open(f'./logs/{data}/{model}/{file}', 'r') as f:
                    hps = '_'.join(file.split('_')[1:])
                    for line in f:
                        if 'Mean CV' in line:
                            line = line.strip()
                            met = float(line.split('metric: ')[-1])
                            hype_dict[hps].append(met)
    best_key, best_met, std_met = None, 0.0, 0.0
    for k, v in hype_dict.items():
        mean_met = sum(v) / len(v)
        variance = sum((x - mean_met) ** 2 for x in v) / (len(v) - 1)  # sample variance
        std_met  = variance ** 0.5
        if mean_met > best_met:
            best_met = mean_met
            best_key = k
    if test: 
        if 'base' in model:
            print(f'Model:{model},\tBest_Tst:{round(best_met, 4)} ± {round(std_met,4)}')
        else:
            print(f'Model:{model},\t\tBest_Tst:{round(best_met, 4)} ± {round(std_met,4)}')
    else: print(f'Model:{model}, Best_Val:{best_met}, Hyp:{best_key}')
        
                            