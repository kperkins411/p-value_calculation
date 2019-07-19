from unittest import TestCase
import pandas as pd
from numpy.random import randint
import numpy as np
from MEM.mem import MEMs
import random
from pvals import pValue
import matplotlib.pyplot as plt

NUMB_ROWS=1000
SPLIT=int(.8*NUMB_ROWS)
VERBOSE=False

class Test_pvals(TestCase):
    def setUp(self):
        bin=[0]*(NUMB_ROWS//2)+[1]*(NUMB_ROWS//2)
        not_bin = [0 if x is 1 else 1 for x in bin]
        unbal_bin = [0]*SPLIT+[1]*(NUMB_ROWS-SPLIT)
        float_val=[10*random.random() for _ in range(NUMB_ROWS)]
        dep=bin.copy()
        random.shuffle(dep)
        data={'bin':bin, 'not_bin':not_bin, 'unbal_bin':unbal_bin,'float_val':float_val,'dep':dep}
        self.df = pd.DataFrame(data,columns=['bin','not_bin','unbal_bin','float_val','dep'])

        #split out dep var
        self.df_y=self.df['dep'].copy()
        self.df.drop(columns=['dep'],axis=1,inplace=True)

    def test_get_pval_bin(self):
        columns=['bin','not_bin','unbal_bin','float_val']
        pv = pValue(self.df, self.df_y,columns,verbose=VERBOSE)
        res=pv.get_all_pvals()
        self.draw_histograms(res, 2, 2)
        pass

    def draw_histograms(self,res, n_rows, n_cols):
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(30, 60))
        fig.subplots_adjust(hspace=0.6, wspace=0.6)
        for i in range(n_rows):
            for j in range(n_cols):
                try:
                    tmp = res[((i) * n_cols) + j]
                except:
                    pass

                ax[i, j].set_title(f'{tmp.col}, pval={tmp.get_pval()}')
                ax[i, j].hist(tmp.permuted_preds, bins=50, ec='red', label='permuted')
                ax[i, j].axvline(tmp.correct_pred, color='k', linestyle='dashed', linewidth=1, label='not-permuted')
        plt.show()
