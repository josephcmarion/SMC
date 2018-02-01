import numpy as np
import scipy.stats as stats
import GPy
from networkx import DiGraph
from networkx.algorithms.shortest_paths.generic import shortest_path
from samplers import NormalPathSampler, MeanFieldIsingSampler, LogisticRegressionSampler, LogisticPriorPathSampler
from utils import gaussian_kernel, unzip
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.misc import comb
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns


class PathEstimator:

    def __init__(self, Q, grid_mins, grid_maxs, grid_types):
        """ Class that does map estimation and path sampling. Not documented because I'm not sure what it does yet.

        parameters
        ----------
        Q: int
            dimension of the path space
        grid_mins: array-like
            the minimum parameter value in each of the Q dimensions
        grid_maxs: array-like
            the maximum parameter value in each of the Q dimensions
        grid_types: list of str
            the type of spacing to using in each path dimension. Must be either 'linear', 'log-increasing', or
            'log-decreasing'
        """

        self.Q = Q

        # arguments that have to do with grids
        self.grid_mins = grid_mins
        self.grid_maxs = grid_maxs
        self.grid_types = grid_types
        self._check_grid_type()  # raises an error if improper types have been specified

        # place holder stuff for the GP regression
        self.pca = None
        self.scaler = None
        self.gps = []
        self.dependent = False
        self.log_transforms = []

        # placeholder for the graph attributes
        self.graph = None
        self.offset = 0

    def _get_grids(self, grid_sizes):
        """ creates custom grids that are specified during initialization. These are used for fitting the energy map,
        establishing the graph, determining the shortest path, and plotting functions

        parameters
        ----------
        grid_sizes: list of int
            the size of the grid along each dimension

        returns
        -------
        list of np.array
            pre-specified grid in each dimension of length n
        """

        grids = []
        for size, grid_min, grid_max, grid_type in zip(grid_sizes, self.grid_mins, self.grid_maxs, self.grid_types):

            if grid_type == 'linear':
                grids.append(np.linspace(grid_min, grid_max, size))
            # todo: fill out these options
            elif grid_type == 'log-decreasing':
                pass
            elif grid_type == 'log-increasing':
                pass

        return grids

    def estimate_lambda(self, path, N, smc_kwargs={}, verbose=True):
        """ estimates lambda, the log ratio of normalizing constants between the initial distribution
         and the target distribution using a specified path. Uses the thermodynamic integration/path sampling
         method described in Gelman and Meng (1998)

        parameters
        ----------
        path: list of tuples
            parameters used at each sampling step.
        N: int
            number of particles to use
        smc_kwargs: dict
            additional arguments to smc_step. For example, resampling_method or kernel_step
        verbose: bool
            if true prints a progress bar tracking sampling progress

        returns
        -------
        float:
            the estimated log ratio of normalizing constants between the initial and
            the target distributions
        """

        # do the sampling
        output = self.sampling(path, N, smc_kwargs, save_all_samples=True, verbose=verbose)
        sample_list = output[0]

        # estimate the log ratio
        potentials = np.array([self._potential(samples, params).mean(0) for samples, params in zip(sample_list, path)])
        avg_potential = (potentials[1:] + potentials[:-1]) / 2.0
        deltas = np.array(path)[1:] - np.array(path)[:-1]
        lamda = (deltas * avg_potential).sum()

        return lamda

    def _estimate_energy(self, samples, path):
        """ estimates the energy at each point in the path using output from the model.
        Samples are drawn using self.sampling with save_all_samples=True

        parameters
        ----------
        samples: np.array
            array of samples from self.sampling
        path: list of tuples
            parameters used at each sampling step.

        return
        ------
        np.array
            (N, 4) energy estimates (beta*beta, beta*inverse_temperature, inverse_temperature*inverse_temperature)
        """

        # initialize storage and indices
        energy = np.zeros((samples.shape[0], np.int(comb(self.Q + 1, 2))))
        indices = np.triu_indices(self.Q)

        for i, (x, params) in enumerate(zip(samples, path)):
            potential = self._potential(x, params)
            energy[i] = np.cov(potential.T).reshape(self.Q, self.Q)[indices]

        return energy

    def fit_energy(self, samples_list, path_list, dependent=False, log_transforms=None, cutoff=None):
        """ fits gaussian process to estimate component energy at different parameter configurations.
        Uses samples and paths from various runs of self.sampling

        parameters
        ----------
        samples_list: list of np.array
            a list of sample outputs from multiple sequential monte carlo runs
        path_list: list of list of tuples
            a list of paths corresponding to the sample estimates
        dependent: boolean
            if true, uses PCA to fit dependent GPs.
        log_transforms: list of boolean
            for each path dimension, if true, log transforms the response prior to fitting the GP.
            This happens before using PCA. Should not be used in conjunction with cutoff. Use with PCA is
            experimental.
        cutoff: None or real>0
            if not none, turns energy values over the cuttoff to be equal to cutoff (with their original sign)
        """

        # first, convert the samples and path into energy estimates and spatial coordinates
        self.dependent = dependent
        self.log_transforms = log_transforms
        self.gps = []  # this will override previously fit gps

        xs = np.vstack(np.array(path_list))
        energy = np.vstack(map(lambda x, y: self._estimate_energy(x, y), samples_list, path_list))

        # allows for pairing of really bad values
        if cutoff is not None:
            energy[np.abs(energy) > cutoff] = cutoff  # *np.sign(energy[np.abs(energy) > cutoff])

        # log transform the response
        self.log_transforms=log_transforms
        if log_transforms is None:
            self.log_transforms = self.Q*[False]

        for q, log_transform in enumerate(self.log_transforms):
            if log_transform:
                if (energy < 0.1).any():
                    print('''Warning: log transform cannot be used with negative energy function.
                    Setting negative values to epsilon''')
                    energy[energy[:, q] < 0.1, q] = 0.001
                energy[:, q] = np.log(energy[:, q])
                print 'Transforming the {}-th coordinate'.format(q)

        # if necessary do scaling and PCA
        if self.dependent:

            # scale data
            self.scaler = StandardScaler()
            ys_scale = self.scaler.fit_transform(energy)

            # rotate data
            self.pca = PCA(n_components=energy.shape[1], whiten=True)
            ys = self.pca.fit_transform(ys_scale)

        else:
            ys = energy

        # now fit the gps
        for q in range(ys.shape[1]):
            print '\nFitting GP #{}'.format(q)
            kernel = GPy.kern.RBF(input_dim=self.Q)
            model = GPy.models.GPRegression(xs, ys[:, q].reshape(-1, 1), kernel)
            model.optimize_restarts(num_restarts=10)
            self.gps.append(model.copy())

    def predict_energy(self, param_array):
        """ predicts component-wise energy at various parameter locations using trained gps

        parameters
        ----------
        param_array: np.array
            (M, Q) array of path parameters. Each row should correspond to a path parameter for the model

        """
        self._check_gps_fit()

        # predict
        predictions = np.array([gp.predict(param_array)[0].flatten() for gp in self.gps]).T

        # transform back from dependent
        if self.dependent:
            predictions = self.scaler.inverse_transform(
                self.pca.inverse_transform(predictions)
            )

        # un-log transform
        for q, log_transform in zip(range(self.Q), self.log_transforms):
            if log_transform:
                predictions[:, q] = np.exp(predictions[:, q])

        return predictions

    def generate_weighted_graph(self, grid_sizes, directions, max_strides, offset=0.1):
        """ creates a network x graph that can be used to find the lowest energy path from one point to another.
        This graph becomes the attribute self.graph

        parameters
        ----------
        grid_sizes: list of int
            the size of the grid along each dimension
        directions: list of str
            length Q list describing the direction of possible connections for each dimension. Each element should
            be either 'forward' (only moves to larger indices), 'backward' (only moves to smaller indices) or
            'both' can move either forwards or backwards
        max_strides: list of int
            length Q list describing the maximum number of steps allowed in each dimension
        offset: float
            additional cost added to each step. prevents the path from taking really small steps
        """

        self._check_gps_fit()

        # init some things
        self.offset = offset
        grids = self._get_grids(grid_sizes)
        indices = [range(size) for size in grid_sizes]
        path = np.array(np.meshgrid(*grids)).reshape(self.Q, -1).T
        predicted_energy = self.predict_energy(path)

        # generate the edge list
        nodes, edges_list = self._generate_edge_list(indices, directions, max_strides)

        # create terrible maps, hope to find a better solution at some point
        # also, convert everything to be a tuple for so it can be used in the graph package
        nodes = [tuple(node) for node in nodes]
        edges_list = [map(tuple, edges) for edges in edges_list]
        node_to_energy = {node: energy for node, energy in zip(nodes, predicted_energy)}
        node_to_params = {node: tuple(params) for node, params in zip(nodes, path)}

        # add elements to graph
        self.graph = DiGraph()
        for node, edges in zip(nodes, edges_list):
            for edge in edges:

                u, v = node_to_params[node], node_to_params[edge]
                cost_u, cost_v = node_to_energy[node], node_to_energy[edge]

                cost = self._estimate_step_energy(u, v, cost_u, cost_v)
                cost += offset

                # vile hack
                if cost <= 0:
                    cost = 0

                self.graph.add_edge(u, v, {'cost': cost})

    @staticmethod
    def _generate_edge_list(indices, directions, max_strides):
        """ creates an array of nodes and corresponding edges for a directed graph structure specified by directions
        and max_strides. This graph is the used for path learning. Can create graphs for paths of arbitrary dimension.

        parameters
        ----------
        indices: list of np.array
            length Q list where each element corresponds to one dimension. each element should be an array of integers
            from 0 to the total number of elements in that direction
        directions: list of str
            length Q list describing the direction of possible connections for each dimension. Each element should
            be either 'forward' (only moves to larger indices), 'backward' (only moves to smaller indices) or
            'both' can move either forwards or backwards
        max_strides: list of int
            length Q list describing the maximmum number of steps allowed in each dimension

        returns
        -------
        np.array
            (#, Q) array of node indices
        list of np.array
            length # list of np.arrays. The i'th element of this list is an array containing the nodes that are
            connected to the i'th element of the first array. This is terribly written.
        """

        # creates the list of all nodes
        nodes = np.array(np.meshgrid(*indices)).reshape(len(indices), -1).T
        edges_list = []
        lengths = [len(index) for index in indices]

        # create an edge_list for each node
        for node in nodes:
            # create list of possible moves
            edges = []
            for i, (index, length, direction, stride) in enumerate(zip(indices, lengths, directions, max_strides)):

                # determine how far to move
                # the 'default' here is the both option
                min_index = max(0, node[i] - stride)
                max_index = min(length, node[i] + stride + 1)

                if direction == 'both':
                    pass
                elif direction == 'forward':
                    min_index = max(0, node[i] + 1)
                elif direction == 'backward':
                    max_index = min(length, node[i])
                else:
                    raise ValueError('''Error in the {}-th direction.
                    Must be 'forward', 'backward' or 'both' '''.format(i))

                edges.append(range(min_index, max_index))
            edges_list.append(np.array(np.meshgrid(*edges)).reshape(len(indices), -1).T)
        return nodes, edges_list

    @staticmethod
    def _estimate_step_energy(start_params, stop_params, start_energy, stop_energy):
        """ estimates the total energy used to move one set of params to another using a single step of quadriture

        parameters
        ----------
        start_params: array-like
            (Q, ) array of parameter values at the initial point
        start_params: array-like
            (Q, ) array of parameter values at the end point
        start_energy: np.array
            (Q, ) array of energy values at the initial point
        stop_energy: np.array
            (Q, ) array of energy values at the end point

        returns
        -------
        float:
            the energy required to make the transition from start_params to stop_params
            in statistical terms, this is the variance of the estimator for this step of quadriture
        """

        # compute the change in step size along each parameter
        deltas = []
        for start, stop in zip(start_params, stop_params):
            deltas.append(stop - start)

        deltas = np.array(deltas)
        indices = np.triu_indices(len(deltas))
        deltas = deltas[indices[0]] * deltas[indices[1]]  # computes the squared change in each direction

        avg_energy = (start_energy + stop_energy) / 2.0

        return (deltas * avg_energy).sum()

    def update_offset(self, new_offset):
        """ changes the offset added to the cost at each edge without recreating the graph.
        Useful for testing the effect of different offset values without recreating the graph.

        parameters
        ----------
        new_offset: float
            cost added to the energy at each edge
        """

        self._check_graph_generated()

        for edge in self.graph.edges_iter():
            self.graph[edge[0]][edge[1]]['cost'] += new_offset - self.offset
        self.offset = new_offset

    def _find_shortest_path(self, start, stop):
        """ wrapper for networkx. ... . shortest_path. Computes the shortest path across the network

        parameters
        ----------
        start: tuple
            params corresponding to the starting distribution
        stop: tuple
            params corresponding to the final distribution

        returns
        -------
        list of tuples
            params for the shortest path from the initial distribution to the target distribution
        """
        self._check_graph_generated()

        return shortest_path(self.graph, start, stop, weight='cost')

    def _create_uniform_path(self, start, stop, n_interpolate):
        """ creates a path by linearly interpolating additional steps within the optimal path

        parameters
        ----------
        start: tuple
            params corresponding to the starting distribution
        stop: tuple
            params corresponding to the final distribution
        n_interpolate: int
            number of points to interpolate at each step. steps=1 returns the best path

        returns
        -------
        list of tuples
            params for the optimal path using the uniform step rule.
        """

        short_path = self._find_shortest_path(start, stop)
        uniform_path = [short_path[0]]

        for start, stop in zip(short_path[:-1], short_path[1:]):
            params = []
            for q in range(self.Q):
                params.append(np.linspace(start[q], stop[q], n_interpolate + 1)[1:])

            params = unzip(params)  # change the index to steps
            params = map(tuple, params)  # change lists to tuples
            uniform_path += params  # save the changes

        return uniform_path

    def _create_weighted_path(self, start, stop, steps):
        """ finds the optimal path, then builds a new path by placing additional steps in sections with particularly
        high variance.

        parameters
        ----------
        start: tuple
            params corresponding to the starting distribution
        stop: tuple
            params corresponding to the final distribution
        steps: int
            number of total SMC transitions to use. May not use exactly this many due to rounding

        returns
        -------
        list of tuples
            params for the optimal path using variance weighting.
        """

        # init some things
        best_path = self._find_shortest_path(start, stop)
        weighted_path = [best_path[0]]

        # compute the weights
        energy = self.predict_energy(np.array(best_path))
        variances = np.array([self._estimate_step_energy(start, stop, start_e, stop_e)
                              for start, stop, start_e, stop_e
                              in zip(best_path[:-1], best_path[1:], energy[:-1], energy[1:])])
        weights = variances / variances.sum() * steps

        weights[weights < 1.0] = 1.0  # make sure each step gets at least 1 weight
        weights = np.round(weights).astype(int)

        # create the path
        for start, stop, weight in zip(best_path[:-1], best_path[1:], weights):
            params = []
            for q in range(self.Q):
                params.append(np.linspace(start[q], stop[q], weight + 1)[1:])

            params = unzip(params)  # change the index to steps
            params = map(tuple, params)  # change lists to tuples
            weighted_path += params

        return weighted_path

    def _check_graph_generated(self):
        """ throws an error if the graph hasn't been generated """
        if self.graph is None:
            raise RuntimeError('''Graph has not been generated. Run {}.generate_weighted_graph
                before running this function.'''.format(self.__class__.__name__))

    def _check_gps_fit(self):
        """ raises an error if the gps haven't been fit """
        # make sure the gps have been fit
        if len(self.gps) == 0:
            raise RuntimeError('''GPS muse be trained before running this function. Run
             {}.fit_energy'''.format(self.__class__.__name__))

    def _check_grid_type(self):
        """ raises an error if the grid types are misspecified """
        good_types = ['linear', 'log-increasing', 'log-decreasing']
        for grid in self.grid_types:
            if grid not in good_types:
                raise ValueError(''' grid types must be either 'linear', 'log-increasing', or 'log-decreasing'.
                 Check initialization of {}'''.format(self.__class__.__name__))


class GeometricTemperedEstimator(PathEstimator):

    def __init__(self, min_temp=0.01, beta_grid_type='linear', temp_grid_type='linear'):
        """ Path sampling class designed for geometric tempered mixtures. Provides wrappers that simplify the
        training of the map, finding the optimal solution, and customized plotting features.

        One possibly irritating feature of this class is that it assumes that the initial distribution can be sampled
        from at any temperature, assuming that beta = 0. Might have to do some work to simplify this when that
        constraint is not true.

        Throughout this class, temperature generally refers to an inverse temperature in (0,1)

        parameters
        ----------
        min_temp: float in (0,1)
            the minimum temperature to consider
        beta_grid_type: str
            denotes the kind of spacing to use for beta. Currently only supports 'linear'
        temp_grid_type: str
            denotes the kind of spacing to use for the inverse temperature. Currently only supports 'linear'
        """

        self.temp_min = min_temp
        PathEstimator.__init__(self, 2, [0.0, min_temp], [1.0, 1.0], [beta_grid_type, temp_grid_type])

        # this is a hacky way to make the paths, but it saves a lot of space
        self.uniform_path = lambda x: self._create_uniform_path((0.0, 1.0), (1.0, 1.0), x)
        self.weighted_path = lambda x: self._create_weighted_path((0.0, 1.0), (1.0, 1.0), x)
        self.linear_path = lambda x: [(beta, 1.0) for beta in np.linspace(0, 1, x)]

    def fit_energy_map(self, N, n_beta, n_temperature, gp_kwargs={}, smc_kwargs={}, verbose=True):
        """ uses sequential monte carlo to estimate the variance of the thermodynamic integrator at a variety
        of mixture/temperature combinations.

        For now, the mixture constants and (inverse) temperature constants are spaced linearly between 0 (or temp_min)
        and 1. Additional functionality should be added to incorporate a variety of logarithmic spacings.

        n_temperature runs of SMC are used. Each run uses the beta path with constant (inverse) temperature.
        After estimating the energy on the grid, GPs are fit to create the energy maps

        parameters
        ----------
        N: int
            number of samples to use during SMC
        n_beta: int > 1
            the number of beta settings on the grid
        n_temperature: int > 1
            the number of (inverse) temperature settings on the grid
        gp_kwargs: dict
            additional arguments to be passed to self.fit_energy(). i.e. log_transform, dependent, or cutoff.
        smc_kwargs: dict
            additional arguments to smc_step. For example, resampling_method or kernel_steps
        verbose: bool
            if true prints a progress bar tracking sampling progress
        """

        # generate the grids
        betas, temperatures = self._get_grids([n_beta, n_temperature])

        # init some things
        samples_list = []
        path_list = []

        # conduct the sampling
        for temp in temperatures:
            print '\nSMC with inverse temperature: {0:.2f}'.format(temp)

            path = [(beta, temp) for beta in betas]
            samples = self.sampling(path, N, smc_kwargs=smc_kwargs, verbose=verbose)[0]

            path_list.append(path)
            samples_list.append(samples)

        # fit the GPs
        self.fit_energy(samples_list, path_list, **gp_kwargs)

    def plot_energy_map(self, plot_kwargs=[{}]*3):
        """ plots the estimated energy map for the sampler on each of the three dimensions

        parameters
        ----------
        plot_kwargs: list of dict
            length three dictionary, each containing arguments to be passed to a subplot
        """

        # I might add this as an argument but it seems clumsy
        n_beta, n_temperature = 101, 101

        # create grids and estimate energy
        betas, temperatures = self._get_grids([n_beta, n_temperature])
        path_array = np.array(map(lambda x: x.flatten(), np.meshgrid(betas, temperatures))).T
        predicted_energy = self.predict_energy(path_array).reshape(101, n_beta, 3)

        # setup some stuff for the axis and titles
        beta_locs = np.linspace(0, n_beta, 11).astype(int)[:-1]
        beta_labels = ['{0:.1f}'.format(beta) for beta in betas[beta_locs]]

        temp_locs = np.linspace(0, n_temperature, 11).astype(int)[:-1]
        temp_labels = ['{0:.1f}'.format(temp) for temp in temperatures[temp_locs]]

        titles = [
            r'$\beta -\beta$ path variance',
            r'$\beta -t$ path variance',
            r'$t -t$ path variance'
        ]

        # do the plotting
        plt.rcParams['figure.figsize'] = 15, 3
        cmap = sns.cubehelix_palette(as_cmap=True)
        for q in range(3):
            plt.subplot(1, 3, q + 1)
            sns.heatmap(predicted_energy[:, :, q][::-1], cmap=cmap, vmin=0, **plot_kwargs[q])
            plt.xticks(beta_locs, beta_labels)
            plt.xlabel(r'$\beta$')

            plt.yticks(temp_locs, temp_labels)
            plt.ylabel(r'$t$')
            plt.title(titles[q])


class GeometricPathEstimator(PathEstimator):

    def __init__(self, grid_type='linear'):
        """ Path sampling class designed for geometric mixtures. Provides wrappers that simplify the
        training of the map, finding the optimal solution, and customized plotting features.


        parameters
        ----------
        grid_type: str
            denotes the kind of spacing to use for beta. Currently only supports 'linear'
        """

        PathEstimator.__init__(self, 1, [0], [1], [grid_type])

        # hack method for constructing the path methods
        self.uniform_path = lambda x: self._create_uniform_path((0.0, ), (1.0, ), x)
        self.weighted_path = lambda x: self._create_weighted_path((0.0, ), (1.0, ), x)
        self.linear_path = lambda x: [(beta, ) for beta in np.linspace(0, 1, x)]

    def fit_energy_map(self, N, n_beta, gp_kwargs={}, smc_kwargs={}, verbose=True):
        """ uses sequential monte carlo to estimate the variance of the thermodynamic integrator at a variety
        of mixture/temperature combinations.

        For now, the mixture constants and (inverse) temperature constants are spaced linearly between 0 (or temp_min)
        and 1. Additional functionality should be added to incorporate a variety of logarithmic spacings.

        n_temperature runs of SMC are used. Each run uses the beta path with constant (inverse) temperature.
        After estimating the energy on the grid, GPs are fit to create the energy maps

        parameters
        ----------
        N: int
            number of samples to use during SMC
        n_beta: int > 1
            the number of beta values to use, spaced using self._get_grids
        gp_kwargs: dict
            additional arguments to be passed to self.fit_energy(). i.e. log_transform, dependent, or cutoff.
        smc_kwargs: dict
            additional arguments to smc_step. For example, resampling_method or kernel_steps
        verbose: bool
            if true prints a progress bar tracking sampling progress
        """

        # generate the path
        betas = self._get_grids([n_beta])[0]
        path = [(beta,) for beta in betas]

        # init some things, this looks a bit silly, however it is consistent with the high dimensional methods
        samples_list = []
        path_list = []

        # get the samples
        samples = self.sampling(path, N, smc_kwargs=smc_kwargs, verbose=verbose)[0]

        path_list.append(path)
        samples_list.append(samples)

        # fit the GPs
        self.fit_energy(samples_list, path_list, **gp_kwargs)

    def plot_energy_map(self, plot_kwargs={}):
        """ plots the estimated energy map for the sampler on each of the three dimensions

        parameters
        ----------
        plot_kwargs: list of dict
            length three dictionary, each containing arguments to be passed to a subplot
        """

        # I might add this as an argument but it seems clumsy
        n_beta = 1001

        # create grids and estimate energy
        betas = self._get_grids([n_beta])[0]
        predicted_energy = self.predict_energy(betas.reshape(-1, 1))

        plt.plot(betas, predicted_energy, color='blue', label='estimated', **plot_kwargs)
        plt.xlabel(r'$\beta$')
        plt.ylabel('Energy')


class NormalPathEstimator(GeometricTemperedEstimator, NormalPathSampler):

    def __init__(self, mean1, mean2, covariance1, covariance2, path_sampler_kwargs={}):
        """Path sampling module for the geometric-tempered D-dimensional normal. Inherits SMC sampling routines from
        smc.samplers.NormalPathSampler

        attributes
        ----------
        mean1: np.array
            (D, ) mean vector for the initial normal distribution
        mean2: np.array
            (D, ) mean vector for the target normal distribution
        covariance1:  np.array
            (D,D) positive definite covariance matrix for the initial distribution
        covariance2:  np.array
            (D,D) positive definite covariance matrix for the initial distribution
        path_sampler_kwargs: dict
            additional arguments to be passed to GeometricTemperedEstimator. options inclue
            min_temp, beta_spacing and
        """

        NormalPathSampler.__init__(self, mean1, mean2, covariance1, covariance2)
        GeometricTemperedEstimator.__init__(self, **path_sampler_kwargs)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, D) array of samples
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]

        return
        ------
        np.array
            (N, 2) vector of potentials w.r.t the mixing parameter and inverse temperature parameter
        """

        beta, inverse_temperature = params

        # pre-compute the kernels
        kernel1 = gaussian_kernel(samples, self.mean1, self.precision1)
        kernel2 = gaussian_kernel(samples, self.mean2, self.precision2)

        # compute potentials
        potential_beta = -0.5*inverse_temperature*(kernel2-kernel1)
        potential_temperature = -0.5*((1.0-beta)*kernel1 + beta*kernel2)

        return np.vstack([potential_beta, potential_temperature]).T

    def _true_energy(self, path, N=10**5):
        """ uses independent gaussian sampling to estimate the true energy values at the specified parameters

        parameters
        ----------
        path: list of tuples
            parameters used at each sampling step. first value is a mixing parameter in [0,1]
            second value is inverse temperature parameter in (0,1]
        N: int
            number of samples to use in estimation

        return
        ------
        np.array
            (N, 4) energy estimates (beta*beta, beta*inverse_temperature, inverse_temperature*inverse_temperature)
        """

        # statistics
        sufficient_statistics = [self._get_normal_parameters(params) for params in path]
        samples = np.array([stats.multivariate_normal(mean=mean, cov=cov).rvs(N)
                            for mean, precision, cov in sufficient_statistics])

        return self._estimate_energy(samples, path)

    def plot_true_energy_map(self, N=10**3, plot_kwargs=[{}]*3):
        """ plots the estimated energy map for the sampler on each of the three dimensions.
        Uses a large number of independent gaussian samples to estimate the true energy

        parameters
        ----------
        N: int
            number of samples to use in estimation
        plot_kwargs: list of dict
            length three dictionary, each containing arguments to be passed to a subplot
        """

        # I might add this as an argument but it seems clumsy
        n_beta, n_temperature = 101, 101

        # create grids and estimate energy
        betas, temperatures = self._get_grids([n_beta, n_temperature])
        path_array = np.array(map(lambda x: x.flatten(), np.meshgrid(betas, temperatures))).T
        true_energy = self._true_energy(path_array, N).reshape(n_temperature, n_beta, 3)

        # setup some stuff for the axis and titles
        beta_locs = np.linspace(0, n_beta, 11).astype(int)[:-1]
        beta_labels = ['{0:.1f}'.format(beta) for beta in betas[beta_locs]]

        temp_locs = np.linspace(0, n_temperature, 11).astype(int)[:-1]
        temp_labels = ['{0:.1f}'.format(temp) for temp in temperatures[temp_locs]]

        titles = [
            r'True $\beta -\beta$ path variance',
            r'True $\beta -t$ path variance',
            r'True $t -t$ path variance'
        ]

        # do the plotting
        plt.rcParams['figure.figsize'] = 15, 3
        cmap = sns.cubehelix_palette(as_cmap=True)
        for q in range(3):
            plt.subplot(1, 3, q + 1)
            sns.heatmap(true_energy[:, :, q][::-1], cmap=cmap, vmin=0, **plot_kwargs[q])
            plt.xticks(beta_locs, beta_labels)
            plt.xlabel(r'$\beta$')

            plt.yticks(temp_locs, temp_labels)
            plt.ylabel(r'$t$')
            plt.title(titles[q])

    def true_lambda(self):
        """ returns the true log ratio of normalizing constants

        returns
        -------
        float:
            the true log ratio of normalizing constants
        """
        return 0.5*(np.linalg.slogdet(2*np.pi*self.covariance2)[1] - np.linalg.slogdet(2*np.pi*self.covariance1)[1])


class IsingPathEstimator(GeometricPathEstimator, MeanFieldIsingSampler):

    def __init__(self, dimension, alpha, path_sampler_kwargs={}):
        """ Path sampling estimator for the ising model. Uses a tempered path from the uniform distribution
        to the distribution of interest.

        attributes
        ----------
        dimension: int
            the number of sites/nodes in the graph
        alpha: float > 0
            the temperature of the model, determining the behaviour

        parameters
        ----------
        path_sampler_kwargs: dict
            arguments to be passed to GeometricPathEstimator, generally just 'grid_type'
        """

        MeanFieldIsingSampler.__init__(self, dimension, alpha)
        GeometricPathEstimator.__init__(self, **path_sampler_kwargs)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, dimension) array of samples
        params: tuple
            first value is the inverse temperature parameter in [0,1]

        return
        ------
        np.array
            (N, 1) vector of potentials w.r.t the mixing parameter and inverse temperature parameter
        """

        beta = params[0]

        # pre-compute the kernels
        potential = self.alpha/self.dimension*(samples.sum(1)**2).reshape(-1, 1)

        return potential

    def _true_energy(self, path):
        """ computes the exact energy at each step along a path

        parameters
        ----------
        path: list of tuples
            inverse temperature parameters used at each sampling step

        return
        ------
        np.array
            (N, Q) energy estimates
        """

        energies = np.zeros(len(path))

        # for each step along the path
        for i, params in enumerate(path):

            probabilities, magnetism = self._true_probabilities(params)
            energies[i] = np.cov(self.alpha/self.dimension*magnetism**2, aweights=probabilities)

        return energies

    def plot_true_energy_map(self, plot_kwargs={}):
        """ plots the estimated energy map for the sampler on each of the three dimensions

        parameters
        ----------
        plot_kwargs: list of dict
            length three dictionary, each containing arguments to be passed to a subplot
        """

        # I might add this as an argument but it seems clumsy
        n_beta = 1001

        # create grids and estimate energy
        betas = self._get_grids([n_beta])[0]
        true_energy = self._true_energy([(beta,) for beta in betas])

        plt.plot(betas, true_energy, color='red',  label='true', **plot_kwargs)
        plt.xlabel(r'$\beta$')
        plt.ylabel('Energy')

    def true_lambda(self):
        """ returns the true log ratio of normalizing constants

        returns
        -------
        float:
            the true log ratio of normalizing constants
        """

        target_z = self._true_normalizing_constant((1.0,))
        initial_z = self._true_normalizing_constant((0.0,))

        return np.log(target_z) - np.log(initial_z)


class LogisticRegressionEstimator(PathEstimator, LogisticRegressionSampler):

    def __init__(self, X, Y, prior_mean, prior_covariance):
        """ Path sampling estimator for logistic regression. Moves from the prior to model of interest via a
        tempered geometric mixture. Markov kernels are done via nuts in STAN

        attributes
        ----------
        X: np.array
            (N, D) array of covariates, should include the intercept column of ones
        Y: np.array
            (N, ) vector of class labels in {0,1}^N
        prior_mean: np.array
            (D, ) vector of prior means
        prior_covariance: np.array
            (D, D) positive definite prior covariance matrix
        """

        LogisticRegressionSampler.__init__(self, X, Y, prior_mean, prior_covariance)
        PathEstimator.__init__(self, 2)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, D) array of samples
        params: tuple
            likelihood inverse temperature in [0,1] and prior mixture parameter in [0,1]

        return
        ------
        np.array
            (N, 2) vector of potentials w.r.t the mixing parameter and inverse temperature parameter
        """

        beta, inverse_temperature = params

        # log likelihood
        mu = np.dot(samples, self.X.T)
        pi = expit(mu)

        # fix some numerical issues
        pi[pi == 1.0] = 1.0 - 10 ** -6
        pi[pi == 0.0] = 10 ** -6

        log_likelihood = (self.Y * np.log(pi) + (1.0 - self.Y) * np.log(1.0 - pi)).sum(1)

        # log prior
        log_prior = -0.5 * gaussian_kernel(samples, self.prior_mean, self.prior_precision)

        # compute potentials
        potential_beta = inverse_temperature*log_likelihood
        potential_temperature = beta*log_likelihood + log_prior

        return np.vstack([potential_beta, potential_temperature]).T


class LogisticPriorPathEstimator(PathEstimator, LogisticPriorPathSampler):

    def __init__(self, X, Y, prior_mean, prior_covariance):
        """ Path sampling estimator for logistic regression. Moves from the prior to model of interest via a
        tempered geometric mixture. Markov kernels are done via nuts in STAN

        attributes
        ----------
        X: np.array
            (N, D) array of covariates, should include the intercept column of ones
        Y: np.array
            (N, ) vector of class labels in {0,1}^N
        prior_mean: np.array
            (D, ) vector of prior means
        prior_covariance: np.array
            (D, D) positive definite prior covariance matrix

        """

        LogisticPriorPathSampler.__init__(self, X, Y, prior_mean, prior_covariance)
        PathEstimator.__init__(self, 2)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, D) array of samples
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]

        return
        ------
        np.array
            (N, 2) vector of potentials w.r.t the mixing parameter and inverse temperature parameter
        """

        beta, alpha = params

        # log likelihood
        mu = np.dot(samples, self.X.T)
        pi = expit(mu)

        # fix some numerical issues
        pi[pi == 1.0] = 1.0 - 10 ** -6
        pi[pi == 0.0] = 10 ** -6

        log_likelihood = (self.Y * np.log(pi) + (1.0 - self.Y) * np.log(1.0 - pi)).sum(1)

        # log prior
        log_prior = -0.5 * gaussian_kernel(samples, self.prior_mean, self.prior_precision)
        log_posterior = -0.5 * gaussian_kernel(samples, self.posterior_mean, self.posterior_precision)

        # compute potentials
        potential_beta = log_likelihood
        potential_alpha = log_prior + log_posterior

        return np.vstack([potential_beta, potential_alpha]).T
