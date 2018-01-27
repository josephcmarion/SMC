import numpy as np
import os
import cPickle as pickle
import scipy.stats as stats
from pystan import StanModel
import time
import matplotlib.pyplot as plt


def log_weights_to_weights(log_weights, epsilon=10**-3):
    """converts a vector of log weights to normalized weights / probabilities

    parameters
    ----------
    log_weights: np.array
        weights on the log scale scale, has shape (samples, )
    epsilon: float
        minimum weight assigned to each sample, used for numerical issues

    returns
    -------
    np.array
        probabilities of each observation
    """

    weights = log_weights - log_weights.max()
    weights = np.exp(weights)+epsilon/log_weights.shape[0]

    return weights/weights.sum()


# todo: this occasionally throws a divide by 0 warning
def residual_resampling(probabilities):
    """uses residual resampling to draw a new set of particle indices.
    See "Comparison of resampling schemes for particle filtering", Douc 2005

    parameters
    ----------
    probabilities: np.array
        positive vector that sums to 0 with shape (samples, )

    returns
    -------
    np.array
        indices corresponding to resampled particles
    """

    N = probabilities.shape[0]
    n_bar = np.floor(probabilities*N).astype(int)
    residuals = probabilities*N-n_bar

    # Sample the whole parts
    integer_indices = []
    for i, n in enumerate(n_bar):
        integer_indices += [i] * n

    # Sample from the remainders
    residual_p = residuals/(N-n_bar.sum())+10**-4/N  # deal with numerical issues
    residual_p /= residual_p.sum()
    residual_inds = np.random.choice(N, N-n_bar.sum(), p=residual_p, replace=True)

    # Combine the indices
    indices = np.hstack([integer_indices, residual_inds])
    np.random.shuffle(indices)

    return indices


def effective_sample_size(probabilities):
    """computes the effective sample size of the particles.
    This is a rough measure of the quality of the particle approximation

    parameters
    ----------
    probabilities: np.array
        positive vector that sums to 0 with shape (samples, )

    returns
    -------
    float
        a number between 0 and 1, indicating the effective number of samples relative to the total number
    """

    return probabilities.mean()**2/(probabilities**2).mean()


def load_stan_model(directory, stan_file, model_code_file, model_code_text=None, load=True):
    """compiles a stan model to be used with an SMC sample.
    Tries to load a pickled model before compiling a new one.
    If a text version of the stan model can't be foudn tries to write a new one.
    Mostly this is just a big pile of error handling meant to make loading stan models simple

    parameters
    ----------
    directory: str
        directory where the stan_file and model_text_file are found. Usually the location of the package
    stan_file: str
        File name of the stan file. If the stan file does not exist, will pickle the model to directory/stan_file
    model_code_file: str
        File name of the model text file.  Can be none if the stan file exists
    model_code_text: str
       string version of the stan model. Can be none if model_code_file exists
    load:
        If true, attempts to load the model. If false, compiles the model

    returns
    -------
    StanModel
        a stan model that will later be used in markov_kernel
    """

    stan_file_name = os.path.join(directory, stan_file)
    model_code_file_name = os.path.join(directory, model_code_file)

    # load the stan model
    if os.path.exists(stan_file_name) and load:
        print('Loading Stan model...')
        with open(stan_file_name, 'rb') as f:
            stan_model = pickle.load(f)

    # if we can't find it,  try to build it
    elif load:

        # first, if the text file doesn't exist try to write it
        # todo: this next block of code should probably become its own function
        if not os.path.exists(model_code_file_name) and model_code_text is not None:
            print('Writing new Stan text file to {}'.format(model_code_file_name))
            with open(model_code_file_name, 'wb') as f:
                pickle.dump(model_code_text, f)
        elif model_code_text is None:
            raise ValueError('''Stan file not found at {}\nModel code file not found at{}\n
                                Model code text is None\nVerify locations of stan file, or model code file,
                                or provide model code text.''')

        # build the stan model
        print('Building Stan model...')

        with open(model_code_file_name, 'rb') as f:
            model_code = pickle.load(f)

        stan_model = StanModel(model_code=model_code)

        # write the model for future use
        with open(stan_file_name, 'wb') as f:
            pickle.dump(stan_model, f)

        print('Stan model written to {}'.format(stan_file_name))

    # if we don't want to load, build a fresh model
    elif not load:

        # write a new text file
        # todo: the primary difference between this block and the last is that this writes a fresh text file
        # todo: even if the stan text file exists. They also handle errors differently
        print('Writing new Stan text file to {}'.format(model_code_file_name))
        with open(model_code_file_name, 'wb') as f:
            pickle.dump(model_code_text, f)

        # build the stan model
        print('Building Stan model...')

        with open(model_code_file_name, 'rb') as f:
            model_code = pickle.load(f)

        stan_model = StanModel(model_code=model_code)

        # write the model for future use
        with open(stan_file_name, 'wb') as f:
            pickle.dump(stan_model, f)

        print('Stan model written to {}'.format(stan_file_name))

    return stan_model


def stan_model_wrapper(sample_list, data, stan_model, kernel_steps, stan_kwargs={}):
    """ wrapper for a stan model to facilitate it's inclusion in sequential monte carlo

    sample_list: list of dictionaries
        each dictionary corresponds to a single sample
    data: dict
        to be passed to StanModel.sampling as the data argument
    stan_model: StanModel
        a compiled StanModel
    kernel_steps: int >= 2
        number of hamiltonian transitions to run
    stan_kwargs: dict
        kwargs passed to SM.sampling, for example 'pars'

    returns
    -------
    np.array
        (N, dimension) array of updated samples. May need to be repackaged
    StanFit4Model
        StanFit4Model object
    """

    if kernel_steps < 2:
        raise ValueError('Number of iterations in sampler must be at least 2, change kernel_kwargs')

    fit = stan_model.sampling(
        data=data,
        init=sample_list,
        warmup=kernel_steps - 2,
        iter=kernel_steps,
        chains=len(sample_list),
        verbose=False,
        **stan_kwargs
    )

    # extract the samples
    new_samples = fit.extract(permuted=False)[-1, :, :-1]  # last last column is a convergence diagnostic

    return new_samples, fit


# todo: this should be object oriented
def plot_density(samples, plot_kwargs={}):
    """creates a simple density plot of the samples using gaussian kde

    parameters
    ----------
    samples: np.array
        one dimensional array of samples
    plot_kwargs:
        additional arguments to be passed to plot

    returns
    -------
    hopefully I will figure out object oriented plotting one day

    """

    density = stats.gaussian_kde(samples)
    xs = np.linspace(samples.min(), samples.max(), 1000)
    ys = density(xs)
    plt.plot(xs, ys, **plot_kwargs)


class ProgressBar:

    def __init__(self, loop_length):
        """Progress bar from http://prooffreaderplus.blogspot.com/2015/05/a-simple-progress-bar-for-ipython.html
        has bee slightly modified

        attributes
        ----------
        loop_length: int
            this is the number of expected iterations
        """

        self.start = time.time()
        self.increment_size = 100.0/loop_length
        self.curr_count = 0
        self.curr_pct = 0
        self.overflow = False
        print '% complete: ',

    def increment(self):
        """function to be called at the end of each loop"""
        self.curr_count += self.increment_size
        if int(self.curr_count) > self.curr_pct:
            self.curr_pct = int(self.curr_count)
            if self.curr_pct <= 100:
                if self.curr_pct%3 == 1:
                    print self.curr_pct,
            elif not self.overflow:
                print("""
                    \n* Count has gone over 100%; likely either due to:\n  - an error in the loop_length specified
                    when progress_bar was instantiated\n  - an error in the placement of the increment() function
                """)

                print('Elapsed time when progress bar full: {:0.1f} seconds.'.format(time.time() - self.start))
                self.overflow = True

    def finish(self):
        """call this method when the loop is complete"""
        if 99 <= self.curr_pct <= 100: # rounding sometimes makes the maximum count 99.
            print('\nElapsed time: {:0.1f} seconds.'.format(time.time() - self.start))
        elif self.overflow:
            print('Elapsed time after end of loop: {:0.1f} seconds.\n'.format(time.time() - self.start))
        else:
            print('\n* End of loop reached earlier than expected.\nElapsed time: {:0.1f} seconds.\n'.format(time.time() - self.start))


def unzip(zipped):
    """ converts a list of tuples to a list of lists, the opposite of the zip function

    parameters
    ----------
    zipped: list of array-like
        usually created by applying zip or by using map on a function that has multiple outputs

    returns
    -------
    list of lists
        the first list is all the 1st elements from the tuples, the second lists is the second element from
        each tuple, so on and so forth
    """
    return [map(lambda x: x[i], zipped) for i in range(len(zipped[0]))]


def gaussian_kernel(samples, mean, precision):
    """ vectorized (in the number of samples) computation of (x-mu)^t Phi (x-mu)

    parameters
    ----------
    samples: np.array
        (N,D) array of samples/observations
    mean: np.array
        (D,) mean vector
    precision: np.array
        (D,D) positive definite precision matrix

    returns
    -------
    np.array
        (N,) vector of gaussian kernels
    """

    delta = samples - mean
    kernels = (np.dot(delta, precision) * delta).sum(-1)  # this probably isn't the right way to do this
    return kernels


def load_pima_indians_data(file_name='data/pima_indians.csv'):
    """ loads pima indians data from csv and puts it in the right format for
    the logistic regression samples. Removes the variables diastolic blood pressure
    triceps_skin_thickness, and serum insulation. Adds a column of ones, and standardizes.
    The pima indians dataset and the full variable names can be found on the UCI repository

    parameters
    ----------
    file_name: str
        location of the pima indians csv

    returns
    -------
    np.array
        (N, 6) array of standardized covariates. The first column is an intercept,
        followed by NP, PGC, BMI, DP, and AGE
    np.array
        (N, ) response vector (in 0, 1)
    """
    from pandas import read_csv
    from sklearn.preprocessing import StandardScaler

    names = [
        'NP',
        'PGC',
        'diastolic_blood_pressure',
        'triceps_skin_thickness',
        'serum insulin',
        'BMI',
        'DP',
        'AGE',
        'class'
    ]
    pima_df = read_csv(file_name, names=names)

    # prepare data
    # drop unnecessary columns
    pima_df.drop([
        'diastolic_blood_pressure',
        'triceps_skin_thickness',
        'serum insulin'
    ], axis=1, inplace=True)

    # pull out the data for regression
    X = pima_df[['NP', 'PGC', 'BMI', 'DP', 'AGE']].values
    X = StandardScaler().fit_transform(X)
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # adding the intercept column
    Y = pima_df[['class']].values.flatten()

    return X, Y

def test_path(directory, fname):
    dir = os.path.dirname(__file__)
    print os.path.join(dir, directory, fname)
