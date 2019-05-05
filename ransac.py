import numpy as np
from scipy import dot
from scipy.linalg import lstsq

def ransac(data,model,n,k,t,d,debug=False,return_all=False, avg_err=None):
    """
    Given:
        data - a set of observed data points
        model - a model that can be fitted to data points
        n - the minimum number of data values required to fit the model
        k - the maximum number of iterations allowed in the algorithm
        t - a threshold value for determining when a data point fits a model
        d - the number of close data values required to assert that a model fits well to data
        avg_err - the average error that a model is rejected if it passes. None means no such rejection.
    Return:
        bestfit - model parameters which best fit the data (or nil if no good model is found)
    iterations = 0
    bestfit = nil
    besterr = something really large
    while iterations < k {
        maybeinliers = n randomly selected values from data
        maybemodel = model parameters fitted to maybeinliers
        alsoinliers = empty set
        for every point in data not in maybeinliers {
            if point fits maybemodel with an error smaller than t
                 add point to alsoinliers
        }
        if the number of elements in alsoinliers is > d {
            % this implies that we may have found a good model
            % now test how good it is
            bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
            thiserr = a measure of how well model fits these points
            if thiserr < besterr {
                bestfit = bettermodel
                besterr = thiserr
            }
        }
        increment iterations
    }
    return bestfit
    }}}
    """
    
    iterations = 0
    bestfit = model.fit(data[:8, :])
    besterr = np.inf
    best_inlier_idxs = np.arange(data.shape[0])
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error( test_points, maybemodel)
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]
        if debug:
            print('test_err.min()',test_err.min())
            print('test_err.max()',test_err.max())
            print('numpy.mean(test_err)',np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d'%(iterations,len(alsoinliers)))
        if len(alsoinliers) > d:
            betterdata = np.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = np.mean( better_errs )
            if (avg_err is None or thiserr <= avg_err) and thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) )
        iterations+=1
    if bestfit is None:
        #raise ValueError("did not meet fit acceptance criteria")
        return bestfit, {'inliers': best_inlier_idxs}
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit

def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange( n_data )
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LinearLeastSquaresModel:
    """linear system solved using linear least squares
    This class serves as an example that fulfills the model interface
    needed by the ransac() function.
    
    """
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        x,resids,rank,s = lstsq(A,B)
        return x
    def get_error( self, data, model):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        B_fit = dot(A,model)
        err_per_point = np.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point
        
def test():
    # generate perfect input data

    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20*np.random.random((n_samples,n_inputs) )
    perfect_fit = 60*np.random.normal(size=(n_inputs,n_outputs) ) # the model
    B_exact = dot(A_exact,perfect_fit)
    assert B_exact.shape == (n_samples,n_outputs)

    # add a little gaussian noise (linear least squares alone should handle this well)
    A_noisy = A_exact + np.random.normal(size=A_exact.shape )
    B_noisy = B_exact + np.random.normal(size=B_exact.shape )

    if 1:
        # add some outliers
        n_outliers = 100
        all_idxs = np.arange( A_noisy.shape[0] )
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        non_outlier_idxs = all_idxs[n_outliers:]
        A_noisy[outlier_idxs] =  20*np.random.random((n_outliers,n_inputs) )
        B_noisy[outlier_idxs] = 50*np.random.normal(size=(n_outliers,n_outputs) )

    # setup model

    all_data = np.hstack( (A_noisy,B_noisy) )
    input_columns = range(n_inputs) # the first columns of the array
    output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
    debug = True
    model = LinearLeastSquaresModel(input_columns,output_columns,debug=debug)
    
    linear_fit,resids,rank,s = lstsq(all_data[:,input_columns],all_data[:,output_columns])

    # run RANSAC algorithm
    ransac_fit, ransac_data = ransac(all_data,model,
                                     5, 5000, 7e4, 50, # misc. parameters
                                     debug=debug,return_all=True)
    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs] # maintain as rank-2 array

        if 1:
            pylab.plot( A_noisy[:,0], B_noisy[:,0], 'k.', label='data' )
            pylab.plot( A_noisy[ransac_data['inliers'],0], B_noisy[ransac_data['inliers'],0], 'bx', label='RANSAC data' )
        else:
            pylab.plot( A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data' )
            pylab.plot( A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fit' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,perfect_fit)[:,0],
                    label='exact system' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,linear_fit)[:,0],
                    label='linear fit' )
        pylab.legend()
        pylab.show()

if __name__=='__main__':
    test()