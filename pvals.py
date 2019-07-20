'''
Calculates the p-values using bootstrapping and random forest
'''
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#this directory contains symlink created at command line like this
# ln -s ../Marginal_Effects_at_Means ./Marginal_Effects_at_Means
#it allows this directory to find Marginal_Effects_at_Means, a directory 1 above this one
#this dir contains a file called mem.py which contains MEMs
from Marginal_Effects_at_Means.mem import MEMs

#random forest params
NUMB_ITERATIONS=500
NUMB_ESTIMATORS=100
MIN_SAMPLES_LEAF=10

class pValInfo():
    '''
    hold permuted list and non-permuted initial val
    returns p-value
    '''

    def __init__(self, col):
        """
        param col: column we are operating on
        """
        self.col = col
        self.correct_pred = []
        self.permuted_preds = []
        self.pval=None

    def get_pval(self):
        '''
        lazy returns the percentage of values in the permuted preds that are larger than the
        non permuted pred
        '''
        if self.pval is None:#lazy evaluation    
            # if either of the above are empty throws exception
            n = sum(i > self.correct_pred for i in self.permuted_preds)
            val= (n / len(self.permuted_preds)).item(0)

            #cheesy but it doesn't matter if they are all above or below
            self.pval=val if val<0.5 else (1-val)
        return self.pval
    
    #these are needed for sorting
    def __eq__(self, other):
        return self.col == other.col #TODO should verify that correct_pred and permuted+preds are almost equal too
    
    def __lt__(self, other):
        return self.get_pval() < other.get_pval()


class pValue():
    def __init__(self,trn,trn_y, all_columns, numb_iter = NUMB_ITERATIONS,verbose=False):
        '''
        :param trn: dataframe to operate on
        :param trn_y: correct classes for above dataframe
        :param all_columns: all the columns in trn to calculate the pVals for
        :param numb_iter:  number of times to create rf out of permuted data in trn
        '''
        self.trn=trn.copy()
        self.trn_y= trn_y.copy()
        self.numb_iter=numb_iter
        self.all_columns=all_columns
        self.mem = MEMs(self.trn)
        self.res=[]
        self.verbose = verbose

    def get_all_pvals(self):
        '''
        Get all the column p-values
        :return: list of pValInfo objects
        '''
        self.res.clear()
        for col in self.all_columns:
            trncpy = self.trn.copy()
            print(f'Column {col}')
            self.res.append(self._get_col_pval(col, trncpy))
        return self.res

    def _get_col_pval(self,col, trn_tmp):
        '''
        Get a single columns p-value
        :param col: the column we are operating on
        :return: numbiter MEM calcs
        '''
        val = pValInfo(col)    #holderS

        #get correct prediction on unpermuted data
        val.correct_pred = self._get_MEM(col,trn_tmp,numb_iter=1)

        #permute column, then get self.numb_iter permuted predictions
        trn_tmp[col]=np.random.permutation(trn_tmp[col].values)
        val.permuted_preds = self._get_MEM(col, trn_tmp, numb_iter=self.numb_iter)

        return val

    def _get_MEM(self,col,trn_tmp, numb_iter):
        '''
         Returns numb_iter MEM calculations
        :param col: the column we are operating on
        :param trn_tmp:
        :param numb_iter:  how many MEM to generate on permuted data, used to produce prediction normal distribution
        :return: numbiter MEM calcs
        '''
        res = []
        cnt=0
        for _ in range(numb_iter):
            # create and train a random forest object
            m_rf = RandomForestClassifier(n_estimators=NUMB_ESTIMATORS, n_jobs=-1, oob_score=True, max_features='auto',
                                          min_samples_leaf=MIN_SAMPLES_LEAF, verbose=False);
            _ = m_rf.fit(trn_tmp, self.trn_y)

            prob_change = self.mem.getMEM_avgplusoneSimple_Probability_Change(m_rf, col)
            res.append(prob_change)

            cnt=(cnt+1)%10
            if (cnt==0):
                # if(self.verbose): print(f'Prob change={prob_change}')
                print(".",end='',flush=True)
        print()
        return res





