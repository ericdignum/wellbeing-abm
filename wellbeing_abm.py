import numpy as np
import networkx as nx
from scipy.stats import genextreme, t
from scipy.spatial.distance import cdist
from sda_functions import SDA, rewire_edges


class WellbeingSim:
    """
    Simulation model of the dynamics of wellbeing in a networked population.
    """

    def __init__(self, params):
        """
        params (dict): A dictionary containing simulation parameters. Defaults:

        # Simulation
        - N (int): Number of agents in the simulation.
        - do_network_update (bool): Update network during simulation?
        - n_steps (int): Number of simulation steps.
        
        # Network
        - network_type (str): Type of network to generate ('sda', 'random',
            'preferential').
        - homophily (float): Homophily parameter for the network. When this
                parameter is below 0 it makes a random (h=-1) or preferential 
                network (h=-2, h=-3, h=-4) just to have these networks while 
                changing only one parameter.
        - avg_degree (int): Average degree of nodes in the network.
        - p_rewire (float): Probability of rewiring an edge in the network.
        - do_network_update (bool): rewire edges or not.
        - wb_in_dist (bool): dist calculations with SWB in addition to capital

        # Comparison
        - soc_comparison (bool): If social comparison should be on or not.
        - comparison_kind (str): 'quantile' or 'upward_sampling'.
        - quantile (float): The quantile one compares to in its ref. group in 
            case comparison_kind='quantile'.
        - soc_comp_w (float or np.array): weight of social comparison.     

        # Habituation / adaptation
        - alpha (float or np.array): Adaptation rate to capital shocks.
        - soc_to_wb (bool): Benefit from social capital to your SWB or not.
        - adapt_kind (str): 'adapt_lvl' or 'rfc'. THE LATTER SHOULD BE TESTED.

        # Capital settings factor
        - wb_to_cap (float or np.array): Feedback of SWB into capital
        - mu (float): first parameter of the chosen capital distribution.
        - sigma (float): second parameter of the chosen capital distribution.
        - capital_dist (str): Type of capital distribution: 'beta', 
            'homogeneous', 'uniform' or 'log-normal'.
        - growth_rate (float or np.array): deterministic growth rate of capital.
        - shock_sigma (float>0): variance of the stochasticity in capital.

        # Shock specifics
        - event_type (str): Type of event ('fixed', 'random' or 'random+fixed').
        - p (float): Probability of getting a deterministic shock
        - int_freq (int): Number of time steps between events.
        - int_size (float): Size of intervention events.
        """
        # Extract the parameters for the simulation
        self.params = params
        for key, value in params.items():
            setattr(self, key, value)
        
        try:
            self.rng = np.random.default_rng(seed=self.seed)
        except AttributeError:
            self.rng = np.random.default_rng()

        # Initialise SWB, capital, adaption levels, the social networks and events
        self.capital = self.init_capital(dist=self.capital_dist, mu=self.mu, sigma=self.sigma)
        self.al_capital = self.capital.copy()
        self.soc_comp_values = self.capital.copy()

        self.init_wellbeing() # initial wellbeing levels
        self.init_network(network_type=self.network_type)
        self.init_events()

    def init_capital(self, dist='lognormal', mu=10, sigma=0.1):
        """
        Initialises the capital of individuals.
        """
    
        capital = np.zeros((self.N, self.n_steps + 1))
        if dist =='lognormal': 
            capital[:,0] = self.rng.lognormal(mean=mu, sigma=sigma, size=self.N)

        elif dist=='beta':
            capital[:,0] = self.rng.beta(a=self.mu, b=self.sigma, size=self.N)
            
        elif dist=='homogeneous':
            capital[:,0] = 0.5

        else:
            print('Distribution not implemented!')
            raise ValueError

        capital[:,1] = capital[:,0]
        return capital
    
    def update_capital(self, step, history, wellbeing, wb_to_cap, random_shocks):
        """
        Updates the capital of individuals at each step.

        Args:
            step (int): the current simulation step.
            history (np.array): historic values of capital.
            wellbeing (np.array): values of wellbeing

        Returns:
            ndarray: updated driver of individuals.
        """
        
        wb_diff = wellbeing[:,step-1] - wellbeing[:,step-2]
        
        growth_rate = 1 + random_shocks[:,step] + self.growth_rate
        # capital = (history[:,step-1] + wb_to_cap * wb_diff) * growth_rate
        multiplier = 2 / (1 + np.exp(-wb_to_cap*wb_diff))
        capital = growth_rate * history[:,step-1] * multiplier
            
        return capital
        
    def calc_dists(self, step):
        """
        Calculates Euclidean distances between the agents their capital. This
        can be extended by taking well-being into account by setting the
        parameter wb_in_dist=True.
        """
        if self.wb_in_dist:
            points = np.vstack((self.capital[:,step], 
                                self.wellbeing[:,step-1])).T
        else:
            points = self.capital[:,step][:,np.newaxis]
        dists = cdist(points, points, metric='euclidean')
        self.dists = dists
        return dists

    def init_network(self, network_type='random'):
        """
        Initializes the network adjacency matrix.

        Args:
            network_type (str): Network type: 'random', 'sda' or 'preferential'.
        """

        # Initialise the network and calculate the distances between the agents
        N = self.N
        self.network = np.zeros((N, N, self.n_steps + 1), dtype=bool)
        dists = self.calc_dists(step=1)

        if network_type == 'random':
            self.network[:, :, 0] = nx.to_numpy_array(
                nx.fast_gnp_random_graph(N, self.avg_degree/N, directed=False))
            
        elif network_type == 'preferential':
            self.network[:, :, 0] = nx.to_numpy_array(
                nx.barabasi_albert_graph(n=self.N, m=self.avg_degree))

        elif network_type == 'sda':

            # Create other networks using the homophily parameter as well
            # such that we can easily change networks using only one parameter
            if self.homophily < 0:
                
                # Random or BA (preferential) network
                if self.homophily<=-2:
                    self.network[:, :, 0] = nx.to_numpy_array(
                        nx.barabasi_albert_graph(
                            n=self.N, m=int(self.avg_degree / 2))
                            )
                    dgrs = self.network[:,:,0].sum(axis=1)
                    srtd_econ = np.sort(self.capital[:,0])
                    
                    if self.homophily==-3:
                        dgrs_idx = dgrs.argsort()
                    elif self.homophily==-4:
                        dgrs_idx = dgrs.argsort()[::-1]
                        
                    self.capital[dgrs_idx,0] = srtd_econ
                    self.capital[dgrs_idx,1] = srtd_econ
                    self.al_capital[dgrs_idx,0] = srtd_econ
                    self.al_capital[dgrs_idx,1] = srtd_econ
                    
                    # Recalculate distances, because we switched the order
                    dists = self.calc_dists(step=1)

                else:
                    self.network[:, :, 0] = nx.to_numpy_array(
                        nx.fast_gnp_random_graph(
                            N, self.avg_degree / self.N, directed=False)
                            )
                
            else:
                
                # Regular SDA network
                sda = SDA.from_dist_matrix(D=dists, k=self.avg_degree, 
                                           alpha=self.homophily, 
                                           p_rewire=0.01, directed=False)
                self.network[:, :, 0] = sda.adjacency_matrix(sparse=False)

        else:
            raise ValueError("Network type not implemented")
        
        self.network[:,:,1] = self.network[:,:,0] # we have two initial steps
        
        # Social capital and inequality calculations if applicable
        if self.soc_to_wb!=0:
            dists = self.dists
            network = self.network[:, :, 1]
            similarities = 1 - dists / dists.max()
            nx_network = nx.from_numpy_array(
                network*similarities, edge_attr='Similarity'
                )
            self.soc_cap = np.array(
                list(nx.clustering(nx_network, weight='Similarity').values())
                )
        else:
            self.soc_cap = 0

    def update_network(self, step):
        """
        Updates the network at each step.

        Args:
            step (int): The current simulation step.

        Returns:
            ndarray: Updated network adjacency matrix.
        """

        # Start with just the old network
        new_network = self.network[:, :, step-1].copy()
        
        # Update the distances between the agents
        if self.soc_to_wb > 0 or self.do_network_update:
            dists = self.calc_dists(step=step)
            
        if self.do_network_update:
            
            if self.network_type=='random':
                new_network = rewire_edges(new_network, p=self.p_rewire, 
                                           directed=False, copy=False)

            elif self.network_type=='sda':
                
                if self.homophily < 0:
                    # Network is random or preferential, so random rewiring
                    new_network = rewire_edges(new_network, p=self.p_rewire, 
                                           directed=False, copy=False)
                else:
                    # Rewire according to homophily principles
                    b = SDA.optim_b(
                        D=dists, k=self.avg_degree, alpha=self.homophily
                        )
                    P = SDA.prob_measure(D=dists, b=b, alpha=self.homophily)

                    # This can definitely be optimised if necessary
                    p_rewire = self.p_rewire
                    rand_nrs, rand_nrs_2 = self.rng.random(size=(2,self.N))
                    for i, row in enumerate(new_network):

                        if rand_nrs[i] < p_rewire:
                            if row.sum()>1:
                                rand_nbor = self.rng.choice(np.nonzero(row)[0])
                                rand_non_nbor = self.rng.choice(
                                    np.nonzero(1-row)[0]
                                    )

                                if P[i,rand_non_nbor] > 0.00001:
                                    condition = (P[i,rand_nbor] / \
                                                 P[i,rand_non_nbor]) / 2
                                    if rand_nrs_2[i] < condition:
                                        new_network[i,rand_nbor] = 0
                                        new_network[rand_nbor,i] = 0
                                        new_network[i,rand_non_nbor] = 1
                                        new_network[rand_non_nbor,i] = 1
                                    
        # Social capital calculations if applicable
        if self.soc_to_wb!=0:
            similarities = 1 - dists / dists.max()
            nx_network = nx.from_numpy_array(
                new_network*similarities, edge_attr='Similarity'
                )
            self.soc_cap = np.array(
                list(nx.clustering(nx_network, weight='Similarity').values())
                )
            
        else:
            self.soc_cap = 0

        return new_network

    def init_wellbeing(self):
        """
        Initializes the wellbeing of individuals.
        """
        self.wellbeing = np.zeros((self.N, self.n_steps + 1))

        # SWB is partly determined by personal traits
        traits = np.random.normal(0, 1, size=self.N)
        self.traits = (traits - traits.min()) / \
            (traits.max() - traits.min())
        # Non-neutral setpoints (using the log transformation)
        swb_setpoint = np.log(self.traits + 0.25)
        
        # To close to 0 or 1 might cause problems in the calculations so 
        # normalise between 0.05 and 0.95
        self.swb_setpoint = 0.8*((swb_setpoint - swb_setpoint.min()) / \
                                 (swb_setpoint.max() - swb_setpoint.min()))+0.1

        if self.capital_dist=='homogeneous':
            self.wellbeing[:,0] = 0.5
            self.swb_setpoint = 0.5
            self.mean_shocks = 0
        else:
            self.wellbeing[:,0] = self.swb_setpoint

        self.wellbeing[:,1] = self.wellbeing[:,0]

    def rfc(self, stimuli, stimulus, min_values, max_values):
        """
        To perform Range-Frequency Compromise calculations.

        THIS IS NOT USED, BUT KEEP IT JUST IN CASE.
        """
        rfc_range = (stimuli[:,-1] - min_values) / (max_values - min_values)
        rfc_freq = np.mean(stimuli <= stimulus[:,np.newaxis], axis=1)
        rfc = self.w*rfc_range + (1-self.w)*rfc_freq
        return rfc

    def kth_closest(self, arr, targets, k, direction='right'):
        """
        Find the k-th closest element in a specified direction for each row using NumPy.
        If no valid values exist in the specified direction, return the target itself.
    
        Parameters:
            arr (np.ndarray): Input 2D array (rows are independent).
            targets (np.ndarray): 1D array of target values, one per row.
            k (int): The k-th closest element to find (1-based).
            direction (str): 'right' for greater, 'left' for less than the target.
    
        Returns:
            np.ndarray: The k-th closest element for each row, or the target itself if no valid values exist.
        """
        if len(targets) != arr.shape[0]:
            raise ValueError("The length of targets must match the number of rows in the array.")
        if direction not in {'right', 'left'}:
            raise ValueError("Direction must be 'right' or 'left'.")
    
        # Compute masks for valid values based on direction
        if direction == 'right':
            mask = arr > targets[:, None]
        elif direction == 'left':
            mask = arr < targets[:, None]
    
        # Replace invalid values with infinity
        masked_arr = np.where(mask, arr, np.inf if direction == 'right' else -np.inf)
    
        # Compute absolute distances from the target for valid values
        distances = np.abs(masked_arr - targets[:, None])
    
        # Sort distances and keep track of sorted indices
        sorted_indices = np.argsort(distances, axis=1)
        sorted_values = np.take_along_axis(masked_arr, sorted_indices, axis=1)
    
        # Count the number of valid values per row
        num_valid = np.sum(mask, axis=1)
    
        # Select the k-th closest element or handle rows with fewer valid values
        k_indices = np.clip(k - 1, 0, num_valid - 1)  # Clip indices to max valid values
        kth_values = np.take_along_axis(sorted_values, k_indices[:, None], axis=1).flatten()
    
        # If no valid values, return the target itself
        kth_values = np.where(num_valid > 0, kth_values, targets)
    
        return kth_values
    
    def update_adapt_lvl(self, values, adapt_lvl, weight, step, soc_comp_w,
                        kind='adapt_lvl'):
        
        y_t = values[:,step-1]
        
        # Social comparison component
        if self.soc_comparison:

            network = self.network[:,:,step-1]
            np.fill_diagonal(network, 1)
            cap_nbors = network*y_t
            cap_nbors[cap_nbors==0] = np.nan          
            
            if self.comparison_kind=='upward_sampling':
                
                condition = (cap_nbors>=y_t[:,np.newaxis]).astype(int)
                probs = condition / condition.sum(axis=1)[:,np.newaxis]
                soc_comp_values = [
                self.rng.choice(
                    cap_nbors[i,:], size=1, axis=0, p=probs[i,:]
                    )[0] for i in range(network.shape[1])
                ]

            elif self.comparison_kind=='kth_closest':
                if self.k <= 0:
                    k = -self.k
                    direction='left'
                else: 
                    k = self.k
                    direction='right'
                soc_comp_values = self.kth_closest(
                                                    arr=cap_nbors, 
                                                    targets=y_t, 
                                                    k=k, 
                                                    direction=direction)

            else:
                soc_comp_values = np.nanpercentile(cap_nbors, 100*self.quantile, 
                                                    axis=1, method='nearest')

        else:
            soc_comp_values = 0
        
        # Type of adaptation
        if kind=='adapt_lvl':
            # If degree is 0, take your own value to compare to
            soc_comp_values = np.where(soc_comp_values==0, y_t, soc_comp_values)
            new_adapt_lvl = weight * ((1-soc_comp_w) * values[:, step-1] + \
                                      soc_comp_w * soc_comp_values) + \
                            (1 - weight) * adapt_lvl[:, step-1]
                                
            
        elif kind=='rfc':
            # IF USED, THIS NEEDS A DOUBLE CHECK.
            new_adapt_lvl = weight * soc_comp_values + \
                (1 - weight) * adapt_lvl[:, step-1]
            
        else:
            print('Habituation function not implemented!')
            raise ValueError

        self.soc_comp_values[:,step-1] = soc_comp_values
        return new_adapt_lvl

    def update_wellbeing(self, step):
        """
        Updates the wellbeing of individuals at each step.

        Args:
            step (int): The current simulation step.

        Returns:
            ndarray: Updated wellbeing of individuals.
        """

        # Current levels of the drivers
        curr_capital = self.capital[:, step]
        prev_wellbeing = self.wellbeing[:, step-1]

        # Habituation to past and comparison group
        al_capital = self.update_adapt_lvl(
                        values=self.capital, 
                        adapt_lvl=self.al_capital,
                        weight=self.alpha, step=step, 
                        soc_comp_w=self.soc_comp_w,
                        kind=self.adapt_kind)
        self.al_capital[:,step] = al_capital
        
        # Calculate the (relative) difference with adaptation level and effect on swb
        # diff_cap = curr_capital - al_capital
        diff_cap = curr_capital - al_capital

        # Reduction due to social capital if shock is negative
        if self.soc_to_wb:
            diff_cap[diff_cap<0] *= (1-self.soc_cap) 

        # Update setpoint (EXPERIMENT)
        if self.beta>0:
            self.swb_setpoint = self.beta*prev_wellbeing + (1-self.beta)*self.swb_setpoint
        
        # Saturation of wellbeing at 0 and 1
        x0_point = self.swb_setpoint / (1-self.swb_setpoint)
        lamb = 2.25

        def sqrt_concave(x, y0, r, prev_wellbeing):
            """
            Increasing concave square-root function.
            x >= r
            y0: y-intercept at x=0
            r: x-intercept (x < 0)
            """
            a = y0 / np.sqrt(-r)
            concave = a * np.sqrt(x - r)
            # rnds = np.abs(np.random.normal(0,0.001,size=len(concave)))
            concave = np.where(concave<=0, 0.001, concave)
            concave = np.where(concave>=1, 0.999, concave)
            concave = np.nan_to_num(concave, copy=True, nan=prev_wellbeing)
            return concave

        if self.perception=='concave':
            r = -20_000
            swb = np.where(diff_cap<=0, 
                           sqrt_concave(x=diff_cap, y0=self.swb_setpoint, r=r, prev_wellbeing=prev_wellbeing),
                           sqrt_concave(x=diff_cap, y0=self.swb_setpoint, r=r * lamb, prev_wellbeing=prev_wellbeing)
                          )
        else:
            if self.N>=2:
                min_slope = 0.0001
                max_slope = 0.00001
                min_val = curr_capital.min()
                max_val = curr_capital.max()
                slope_sig = min_slope + ((curr_capital - min_val) / (max_val - min_val)) * (max_slope - min_slope)
                self.slope_sig = -1*slope_sig
            
            swb = np.where(diff_cap<0, 
                           x0_point / (x0_point + np.exp(self.slope_sig*lamb*diff_cap)),
                           x0_point / (x0_point + np.exp(self.slope_sig*diff_cap))
                          )
            
        return swb

    def init_events(self):

        event_type = self.event_type
        self.cap_event_sizes = np.zeros(self.capital.shape)

        if 'fixed' in event_type:
            T = self.n_steps
            int_freq = self.int_freq
            int_size = self.int_size
            steps = np.array(
                [step for step in range(50,T+1) if step%int_freq==0]
                ).astype(int)
            self.cap_event_steps = steps
            amounts = np.array(
                [int_size for step in range(50,T+1) if step%int_freq==0]
                )
            
            # Decide who gets a shock and who does not
            random_idx = self.rng.binomial(n=1, p=self.p, size=self.N)
            self.random_idx = random_idx
            fixed_cap_events = random_idx[:,np.newaxis] * amounts[np.newaxis,:]

            # Prefill shocks
            self.capital[:,steps] += fixed_cap_events
            self.cap_event_sizes[:,steps] += fixed_cap_events
        
        if 'random' in event_type:

            if 'gev' in event_type:
                random_cap_shocks = genextreme.rvs(
                    loc=0, c=0, scale=self.shock_sigma, 
                    size=self.capital.shape)

            elif 'student_t' in event_type:
                random_cap_shocks = t.rvs(df=self.shock_sigma, 
                    size=self.capital.shape)

            else:
                random_cap_shocks = self.rng.normal(loc=0,
                    scale=self.shock_sigma, size=self.capital.shape)
            
            # Update the drivers already and keep track of the shock sizes
            self.random_cap_shocks = random_cap_shocks

        else:
            self.random_cap_shocks = np.zeros(self.capital.shape)
                
    def update(self, step):
        self.capital[:, step] += self.update_capital(
            step, self.capital, self.wellbeing, self.wb_to_cap,
            self.random_cap_shocks
            )
        self.network[:, :, step] += self.update_network(step=step)
        self.wellbeing[:, step] += self.update_wellbeing(step=step)

    def run_simulation(self):
        """
        Runs the simulation.
        """
        for step in range(2, self.n_steps+1):
            self.update(step)


from scipy import sparse
class WellbeingSimRevised(WellbeingSim):
    """
    Revision for speedup and new functionality
    """
    
    def update_capital(self, step, history, wellbeing, wb_to_cap, random_shocks):
        """
        Updates the capital of individuals at each step.

        Args:
            step (int): the current simulation step.
            history (np.array): historic values of capital.
            wellbeing (np.array): values of wellbeing

        Returns:
            ndarray: updated driver of individuals.
        """

        old_capital = history[:,step-1]
        wb_diff = wellbeing[:,step-1] - wellbeing[:,step-2]
        # wb_diff = wellbeing[:,step-1] - self.swb_setpoint
        
        growth_rate = 1 + random_shocks[:,step] + self.growth_rate
        multiplier = 2 / (1 + np.exp(-wb_to_cap*wb_diff))
        # multiplier = (1 + wb_to_cap * wb_diff)
        new_capital = growth_rate * old_capital * multiplier
        
        if self.step==self.int_freq:
            if self.N>1:
                new_capital = np.where(self.random_idx==1, new_capital*(1-self.rel_size), new_capital)
            else:
                new_capital *= (1-self.rel_size)
        return new_capital

    def init_network(self, network_type='random'):
        """
        Initializes the network adjacency matrix.

        Args:
            network_type (str): Network type: 'random', 'sda' or 'preferential'.
        """

        # Initialise the network and calculate the distances between the agents
        N = self.N
       
        if network_type == 'random':
            network = nx.fast_gnp_random_graph(N, self.avg_degree/N, directed=False)
            
        elif network_type == 'preferential':
            network = nx.barabasi_albert_graph(n=self.N, m=self.avg_degree)

        elif network_type == 'sda':

            # Create other networks using the homophily parameter as well
            # such that we can easily change networks using only one parameter
            if self.homophily < 0:
                
                # Random or BA (preferential) network
                if self.homophily<=-2:
                    network = nx.barabasi_albert_graph(
                            n=self.N, m=int(self.avg_degree / 2))
                            
                    dgrs = network.sum(axis=1)
                    srtd_econ = np.sort(self.capital[:,0])
                    
                    if self.homophily==-3:
                        dgrs_idx = dgrs.argsort()
                    elif self.homophily==-4:
                        dgrs_idx = dgrs.argsort()[::-1]
                        
                    self.capital[dgrs_idx,0] = srtd_econ
                    self.capital[dgrs_idx,1] = srtd_econ
                    self.al_capital[dgrs_idx,0] = srtd_econ
                    self.al_capital[dgrs_idx,1] = srtd_econ
                    
                    # Recalculate distances, because we switched the order
                    dists = self.calc_dists(step=1)

                else:
                    network = nx.fast_gnp_random_graph(
                            N, self.avg_degree / self.N, directed=False)
                
            else:
                
                # Regular SDA network
                dists = self.calc_dists(step=0)
                sda = SDA.from_dist_matrix(D=dists, k=self.avg_degree, 
                                           alpha=self.homophily, 
                                           p_rewire=0.01, directed=False)
                network = sda.adjacency_matrix(sparse=False)
                network = nx.from_numpy_array(network)

        else:
            raise ValueError("Network type not implemented")

        self.network = nx.to_scipy_sparse_array(network, dtype=bool)
        self.soc_cap = 0

    def sparse_row_quantile(self, A, q, include_zeros=False):
        """
        Compute quantiles along axis=1 for a CSR sparse matrix
        without densifying.
    
        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Input sparse matrix.
        q : float in [0, 1]
            Desired quantile.
        include_zeros : bool
            Whether to include implicit zeros in the quantile computation.
    
        Returns
        -------
        quantiles : np.ndarray, shape (A.shape[0],)
            Quantile per row.
        """
        if not sparse.isspmatrix_csr(A):
            A = A.tocsr()
    
        n_rows, n_cols = A.shape
        out = np.zeros(n_rows, dtype=A.dtype)
    
        for i in range(n_rows):
            row_start, row_end = A.indptr[i], A.indptr[i + 1]
            row_data = A.data[row_start:row_end]
    
            if row_data.size == 0:
                out[i] = 0.0
                continue
    
            if not include_zeros:
                out[i] = np.quantile(row_data, q, method='nearest')
                continue
    
            nz = row_data.size
            zeros = n_cols - nz
            frac_zero = zeros / n_cols
    
            if q <= frac_zero:
                out[i] = 0.0
            else:
                adjusted_q = (q - frac_zero) / (1 - frac_zero)
                out[i] = np.quantile(row_data, adjusted_q, method='nearest')
    
        return out
    
    def update_adapt_lvl(self, values, adapt_lvl, weight, step, soc_comp_w,
                        kind='adapt_lvl'):

        prev_step = self.step-1
        prev_cap = self.capital[:,prev_step]
        prev_adapt_lvl = self.al_capital[:,prev_step]
        
        # Social comparison component
        if self.soc_comparison:

            # Calculate capital of neighbours and the quantiles
            network = self.network
            network.setdiag(1)
            nbors_cap = network.multiply(prev_cap)
            soc_comp_values = self.sparse_row_quantile(nbors_cap, 
                                                       q=self.quantile, 
                                                       include_zeros=False)
            # Calculate the new adaptation level
            comparison_part = (1-self.soc_comp_w) * prev_cap + self.soc_comp_w * soc_comp_values
            new_adapt_lvl = self.alpha * comparison_part + (1 - self.alpha) * prev_adapt_lvl

        else:
            soc_comp_values = 0
            new_adapt_lvl = weight * prev_cap + (1 - weight) * prev_adapt_lvl
        
        
        self.soc_comp_values[:,prev_step] = soc_comp_values
        return new_adapt_lvl

    def sqrt_concave(self, x, y0, r, prev_wellbeing):
        """
        Increasing concave square-root function.
        x >= r
        y0: y-intercept at x=0
        r: x-intercept (x < 0)
        """
        a = y0 / np.sqrt(-r)
        concave = a * np.sqrt(x - r)
        # rnds = np.abs(np.random.normal(0,0.001,size=len(concave)))
        concave = np.where(concave<=0, 0.001, concave)
        concave = np.where(concave>=1, 0.999, concave)
        concave = np.nan_to_num(concave, copy=True, nan=prev_wellbeing)
        return concave

    def update_wellbeing(self, step):
        """
        Updates the wellbeing of individuals at each step.

        Args:
            step (int): The current simulation step.

        Returns:
            ndarray: Updated wellbeing of individuals.
        """

        # Current levels of the drivers
        curr_capital = self.capital[:, step]
        prev_wellbeing = self.wellbeing[:, step-1]

        # Habituation to past and comparison group
        al_capital = self.update_adapt_lvl(
                        values=self.capital, 
                        adapt_lvl=self.al_capital,
                        weight=self.alpha, step=step, 
                        soc_comp_w=self.soc_comp_w,
                        kind=self.adapt_kind)
        self.al_capital[:,step] = al_capital
        
        # Calculate the (relative) difference with adaptation level and effect on swb
        diff_cap = curr_capital - al_capital

        # Saturation of wellbeing at 0 and 1
        x0_point = self.swb_setpoint / (1-self.swb_setpoint)
        lamb = 2.25

        if self.perception=='concave':
            r = -20_000
            swb = np.where(diff_cap<=0, 
                           self.sqrt_concave(x=diff_cap, y0=self.swb_setpoint, r=r, prev_wellbeing=prev_wellbeing),
                           self.sqrt_concave(x=diff_cap, y0=self.swb_setpoint, r=r * lamb, prev_wellbeing=prev_wellbeing)
                          )
        else:
            if self.N>10:
                min_slope = 0.0001
                max_slope = 0.00001
                min_val = curr_capital.min()
                max_val = curr_capital.max()
                slope_sig = min_slope + ((curr_capital - min_val) / (max_val - min_val)) * (max_slope - min_slope)
                self.slope_sig = -1*slope_sig
            swb = np.where(diff_cap<0, 
                           x0_point / (x0_point + np.exp(self.slope_sig*lamb*diff_cap)),
                           x0_point / (x0_point + np.exp(self.slope_sig*diff_cap))
                          )
            
        return swb

   
                
    def update(self, step):
        self.capital[:, step] += self.update_capital(
            step, self.capital, self.wellbeing, self.wb_to_cap,
            self.random_cap_shocks
            )
        self.wellbeing[:, step] = self.update_wellbeing(step=step)

    def run_simulation(self):
        """
        Runs the simulation.
        """
        for step in range(2, self.n_steps+1):
            self.step = step
            self.update(step)
        

if __name__=='__main__':

    N = 50
    model = WellbeingSim(params={
    # Network params
    'N': N, 'p': 0.5, 'do_network_update':False, 'network_type': 'sda', 
    'homophily': -1, 'avg_degree': 10, 'p_rewire': 0.1, 'wb_in_dist':False,

    # Simulation params
    'n_steps': 99,  'alpha': np.random.uniform(0.05,0.15,size=N), 
    'soc_comp_w': 0.5, 'shock_sigma':0.005, 'soc_to_wb':0,  'adapt_kind': 'adapt_lvl',

    # Capital params
    'wb_to_cap': 0, 'mu': 1, 'sigma': 1, 'capital_dist': 'beta', 
    'soc_comparison':True, 'comparison_kind':'quantile',
    'quantile':0.7, 'growth_rate':0,  
    
    # Event params
    'event_type':'fixed+random',
    'int_freq':50, 'int_size':-0.25
    })
    model.run_simulation()
    print(model.wellbeing)