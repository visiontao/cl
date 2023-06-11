import os
import csv
import numpy as np
from utils.metrics import forgetting, backward_transfer

class CsvLogger:
    def __init__(self, root='results'):
        
        self.root = root           
        if not os.path.exists(self.root):
            os.makedirs(self.root)
    
    def write(self, setting, results, args):
        params = ['buffer_size', 'alpha', 'beta', 'lr', 'batch_size', 'n_epochs']
        
        columns = [f'task {i+1}' for i in range(args.n_tasks)] + ['forgetting', 'backward_transfer'] + params   
                
        row = {}        
        for t in range(args.n_tasks):
            row[f'task {t+1}'] = round(np.mean(results[t]), 2)
        
        row['forgetting'] = round(forgetting(results), 2)        
        row['backward_transfer'] = round(backward_transfer(results), 2)
        
        row['buffer_size'] = args.buffer_size  
        
        row['alpha'] = args.alpha
        row['beta'] = args.beta
        row['lr'] = args.lr
        row['batch_size'] = args.batch_size
        row['n_epochs'] = args.n_epochs

        csv_log_name = os.path.join(self.root, f'{args.model}_{args.dataset}_{args.n_tasks}_{setting}.csv')
        csv_log_exists = os.path.exists(csv_log_name)
        
        writer = csv.DictWriter(open(csv_log_name, 'a'), fieldnames=columns)
        if not csv_log_exists:
            writer.writeheader()
        writer.writerow(row)



