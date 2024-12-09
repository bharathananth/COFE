import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

class SyntheticData:
    def __init__(self, n_features, n_samples, seed=None):
        """Constructor of the synthetic data class.

        Parameters
        ----------
        n_features : int
            Number of features in dataset
        n_samples : int
            Number of samples in dataset
        n_samples : int, or SeedSequence object, or array-like int
            Seed used for random data generation
        """
        self.p = n_features
        self.N = n_samples
        self.rng = np.random.default_rng(seed)
        self.frqs = None
        self.t = None
        
    def oscillatory_groups(self, n_groups, group_frq=None):
        """Assign self.frqs property with an optional array specifying the 
        frequencies of each group. 

        Parameters
        ----------
        n_groups : int
            Number of different rhythm frequencies in the data (excluding 
            arrhythmic) with fundamental set to 1.0 and (n_group-1) harmonics
        group_frq : array-like, optional
            Option to individually specify each group frequency with respect to 
            a reference period of 1.0 in an array, by default None, which means 
            harmonic frequencies with one non-rhythmic group (frequency=0)

        Raises
        ------
        ValueError
            if invalid frequency values are specified
        ValueError
            if number of frequncies do not match frequency list
        """
        if group_frq is not None:
            if len(group_frq) == n_groups:
                if all(isinstance(x, (float)) for x in group_frq):
                    self.frqs = group_frq
                    # Append frequency value for non-oscillatory features.
                    self.frqs.insert(0, 0.0)
                else:
                    raise ValueError('Invalid frequency value(s) specified')
            else:
                raise ValueError('Number of frqs specified does not match length of frequency list')
        else:
            # Frequency values are higher harmonics
            self.frqs = np.arange(n_groups+1) 
    
    def generate_data(self, normal_dist_times=True, st_dev=0.125, rhy_frac=0.1, 
                        spline=False, var_amp=None):
        """
        Parameters
        ----------
        normal_dist_times : bool, optional
            Option to specify how sample times are distributed. True for normal, 
            False for uniform, by default True
        st_dev : float, optional
            Standard deviation of sample collection time under normal 
            distribution over [0.0, 1.0], by default 0.125
        rhy_frac : float, optional
            Desired fraction of rhythmic features in the dataset, by default 0.1
        spline : boolean, optional
            Generate data with according to sinusoidal (False) or arbitrary 
            (True) waveforms, by default False
        var_amp : float, optional
            Should amplitudes be drawn from a Gamma distribution (var_amp,
            1/var_amp)?, by default None
        Returns
        -------
        X
            data matrix of size n_features x n_samples
        group_ind
            Dictionary with key being oscillatory group number (or 0 for 
            non-oscillatory group) and values being row indices of features 
            corresponding to key in dataset.
        rel_ampl
            Dictionary of relative amplitudes of features, if var_amp = True. 
            Otherwise, None.


        Raises
        ------
        ValueError
            if information on oscillatory_groups is not set
        """
        if self.frqs is None:
            raise ValueError('Run oscillatory_groups method before '
                             'generate_data')
        
        # Sample times are only generated from one cycle [0, 1.0]
        if normal_dist_times:
            # Sample times are not generally uniformly distributed.
            self.t = np.mod(self.rng.normal(loc=self.rng.uniform(), 
                                            scale=st_dev, 
                                            size=(self.N,)), 1.0)
        else:
            # Option to sample uniformly
            self.t = self.rng.uniform(size=self.N)
            
        X = np.zeros((self.p, self.N))
        group_ind = dict()
        rel_ampl = dict()
        shuffled_indices = self.rng.permutation(self.p)
        arrhy_indices = shuffled_indices[:int((1 - rhy_frac)*self.p)] 
        group_ind[0] = arrhy_indices
        rhy_indices = (i for i in range(self.p) if i not in arrhy_indices)
        for i in rhy_indices:
            # Determine oscillatory group of features
            g = int(np.floor(self.rng.uniform() * (len(self.frqs) - 1)) + 1)
            # Store index values of features and corresponding oscillatory group
            if g in group_ind:
                group_ind[g].append(i)
            else:
                group_ind[g]= [i]
            # generate data for all samples of feature i
            if spline:
                y = 2*self.rng.random(3)-1
                y = np.append(y, y[0])
                x = np.sort(np.append(np.array([0, 1]), self.rng.random(2)))
                cs = CubicSpline(x, y, bc_type='periodic')
                fmin = minimize_scalar(cs, bounds=[0.0, 1.0], method="bounded")
                fmax = minimize_scalar(lambda x: -cs(x), bounds=[0.0, 1.0], 
                                        method="bounded")
                y_feature = (2*cs(np.mod(self.t*self.frqs[g], 1.0)) - cs(fmin.x)
                                         - cs(fmax.x))/(cs(fmax.x) - cs(fmin.x))
            else:
                y_feature = np.sin(2*np.pi*self.frqs[g]*self.t +
                                   2*np.pi*self.rng.random())
            
            if var_amp is not None:
                A =  self.rng.gamma(shape=var_amp, scale=1/var_amp)
                rel_ampl[i] = A
                y_feature *= A
            
            # Add features row to data matrix after standardization
            X[i, :] = y_feature
        # Return n_features x n_samples matrix
        return X, group_ind, rel_ampl
            
    def corrupt(self, X, st_dev=0.05):
        """Add noise to dataset.

        Parameters
        ----------
        X : nparray
            Dataset
        st_dev : float, optional
            Standard deviation of the additive Gaussian measurement noise, by 
            default 0.05

        Returns
        -------
        nparray
            Dataset with noise.
        """
        X_noisy = np.copy(X)
        # Add gaussian noise to every data entry
        X_noisy += self.rng.normal(scale=st_dev, size=X_noisy.shape)
                
        return X_noisy
    
    def plot_X(self, X, group_ind, frqs=None, n_arrhy=0):
        """Plot features against time

        Parameters
        ----------
        X : nparray
            Dataset 
        group_ind : dictionary
            Key is oscillatory group number (or 0 for non-oscillatory group) and 
            values are row indices of features corresponding to key in dataset.
        frqs : list of tuple, optional
            Each tuple has two values. First value represents oscillatory group,
             second value represents number of features to plot from this group,
              by default None, which represents one feature per group
        n_arrhy : int, optional
            Number of non-rhythmic features to plot, by default 0

        Raises
        ------
        ValueError
            if more non-rhythmic features are requested than available
        ValueError
            if more rhythmic features are requested than available
        """
        if n_arrhy > len(group_ind[0]):
            raise ValueError('Number of non-oscilating featuress to plot '
                             'greater than actual number of non-oscilating'
                             'features.')
        
        plt.figure()
        if frqs is None:
            frqs = zip(np.arange(1,len(group_ind)),[1]*(len(group_ind)-1))

        for tup in frqs:
            if tup[1] > len(group_ind[tup[0]]):
                raise ValueError('Number of oscilating features in specified '
                                 'group is less than desired number of plots.')
            for i in range(tup[1]):
                plt.scatter(self.t, X[group_ind[tup[0]][i]], label=
                            'feature {}, group {}'.format(group_ind[tup[0]][i] 
                                                                + 1, tup[0]))
            
        for i in range(n_arrhy):
            plt.scatter(self.t, X[group_ind[0][i]], 
                        label='feature {}, group {}'.format(group_ind[0][i] 
                                                                        + 1, 0))

        plt.ylabel("features")
        plt.xlabel("scaled time within one cycle")
        plt.xlim([0.0, 1.0])
        plt.title("Example profiles from the dataset")
        plt.legend(loc="lower right")