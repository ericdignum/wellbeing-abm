import ray
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import statsmodels.api as sm
import matplotlib.colors as mcolors
from wellbeing_abm import WellbeingSim
from sklearn.preprocessing import KBinsDiscretizer

def calc_longterm_change(swb):
    '''
    Longterm change is defined as the average swb before the shock 
    (t=30 until t=49) minus that after the schock (t=80 until t=99)
    '''
    before_shock = swb[:,30:50].mean(axis=1)
    # before_shock = swb[:,0]
    after_shock = swb[:,80:100].mean(axis=1)
    longterm_change = after_shock - before_shock
    return longterm_change

def calc_instability(swb):
    '''
    Instability is calculated using the correlation coefficient (CV)
    '''
    std = swb.std(axis=1)
    mean = swb.mean(axis=1)
    cv = std / mean
    return cv

def calc_recovery_time(swb):
    '''
    The recovery time is defined as the number of time steps (after t=50) 
    it takes to have a SWB within 5% of your level of SBW that you had at t=49.
    Note that this could also be immediately or never.
    '''
    perc = 0.05
    rel_to = 49
    relatives = (swb[:,50:]-swb[:,rel_to,np.newaxis]) / swb[:,rel_to,np.newaxis]
    abs_relatives = np.abs(relatives) < perc
    never_return = abs_relatives.sum(axis=1)==0
    recovery = np.argmax(abs_relatives, axis=1)
    recovery[never_return] = 50
    return recovery


def calc_gini(x):
    """Compute Gini coefficient for a 1D array."""
    x = np.asarray(x, dtype=float)
    x = x - np.min(x)  # shift so min is zero
    if np.all(x == 0):
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * sorted_x) / (n * np.sum(sorted_x))


def calc_welfare(samples):
    # Welfare function according to Sen (1973)
    if len(samples.shape)>1:
        gini = np.array([calc_gini(samples_t) for samples_t in samples])
        avg = samples.mean(axis=1)
        welfare = avg * (1 - gini)
        return welfare.mean()
    else:
        gini = calc_gini(samples)
        avg = samples.mean()
        welfare = avg * (1 - gini)
        return welfare

def calculate_stats(model):
    '''
    Function that calculates all the outputs and stores them in a 
    DataFrame with proper variable names.
    '''
    
    plot_vars = pd.DataFrame()
    swb = model.wellbeing
    cap = model.capital
    init_cap = cap[:,0]
    init_swb = swb[:,0]
    params = model.params

    plot_vars['Set-point ($swb_{i(t=0)}$)'] = init_swb
    plot_vars['Final SWB ($swb_{i(t=99)}$)'] = swb[:,99]
    plot_vars['Initial Income ($y_{i(t=0)}$)'] = init_cap
    plot_vars['Final Income ($y_{i(t=99)}$)'] = cap[:,99]
    plot_vars['$\\Delta y_i = y_{i(t=99)} - y_{i(t=0)}$'] = cap[:,99] - init_cap
    plot_vars['$\\Delta swb_i = swb_{i(t=99)} - swb_{i(t=0)}$'] = swb[:,99] - init_swb
    plot_vars['Comparison ($w_i$)'] = params['soc_comp_w']
    plot_vars['Adaptation ($\\alpha_i$)'] = params['alpha']
    plot_vars['Feedback ($\\delta_i$)'] = params['wb_to_cap']
    
    shocked = model.random_idx==1
    plot_vars['Shocked'] = shocked
    lowest_quintile = np.where(init_cap <= np.percentile(init_cap, 20), 1, np.nan)
    
    plot_vars['Abs. Differences'] = np.abs(swb[:,50:] - swb[:,49,np.newaxis]).sum(axis=1)
    plot_vars['Recovery Time'] = calc_recovery_time(swb)
    plot_vars['Instability (CV)'] = calc_instability(swb)
    plot_vars['Long-term Change'] = calc_longterm_change(swb)
    plot_vars['LTC Lower Quintile'] = lowest_quintile * plot_vars['Long-term Change'].values
    net_change_cap = (model.capital[:,51] - model.soc_comp_values[:,51]) - \
                        (model.capital[:,49] - model.soc_comp_values[:,49])
    plot_vars['Net change in income'] = net_change_cap

    # Calculate welfare
    plot_vars['Welfare (inc, $t=0$)'] = calc_welfare(cap[:,0])
    plot_vars['Welfare (swb, $t=0$)'] = calc_welfare(swb[:,0])
    plot_vars['Welfare (inc)'] = calc_welfare(cap[:,99])
    plot_vars['Welfare (swb)'] = calc_welfare(swb[:,99])
    plot_vars['$\\Delta$Welfare (inc)'] = plot_vars['Welfare (inc)'] - plot_vars['Welfare (inc, $t=0$)']
    plot_vars['$\\Delta$Welfare (swb)'] = plot_vars['Welfare (swb)'] - plot_vars['Welfare (swb, $t=0$)']
    
    return plot_vars, cap, swb, shocked

def scatterplot(data, x, y, hue, vmin, vmax, vcenter, cmap, ax, fig):
    """
    Helper function to create a scatterplot with a colorbar."""
    normalise = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    if cmap=='viridis': colormap=cm.viridis
    else: colormap = cm.RdBu

    # Plot only shocked
    if y=='Recovery Time':
        data = data[data['Shocked']==1]

    ax=sns.scatterplot(
        data=data,
        y=y,
        x=x,
        c=data[hue],
        norm=normalise,
        cmap=colormap,
        ax=ax,
    )

    scalarmappaple = cm.ScalarMappable(norm=normalise, cmap=colormap)
    scalarmappaple.set_array(data[y].values)
    cbar = fig.colorbar(scalarmappaple, ax=ax)
    cbar.set_label(hue, size=8)
    return ax

def sqrt_concave(x, y0, r):
    """
    Increasing concave square-root function.
    x >= r
    y0: y-intercept at x=0
    r: x-intercept (x < 0)
    """
    a = y0 / np.sqrt(-r)
    concave = a * np.sqrt(x - r)
    concave = np.where(concave<0, 0, concave)
    concave = np.where(concave>1, 1, concave)
    return concave

def discretise(data, n_bins, strategy):
    est = KBinsDiscretizer(
        n_bins=n_bins, encode='ordinal', 
        strategy=strategy, quantile_method='linear',
    )
    est.fit(data)
    data_discr = est.transform(data).astype(int)
    return data_discr

@ray.remote
def comparison_parallel(params):
    model = WellbeingSim(params=params)
    model.run_simulation()

    comparison_type = (model.capital>model.soc_comp_values).sum(axis=1)
    comparison_frame = pd.DataFrame(model.wellbeing)    
    col = 'Comparison'
    comparison_frame[col] = comparison_type
    conditions  = [comparison_frame[col]>=90, 
                   (comparison_frame[col]<90) & (comparison_frame[col]>1), 
                   comparison_frame[col]<=1]

    choices     = [f'Downward', f'Mixed', f'Upward' ]
    comparison_frame[col] = np.select(conditions, choices, default='None')
    comparison_frame = comparison_frame.melt(
        id_vars='Comparison', var_name='Time', value_name='SWB'
        )
    
    # Calculate income groups every time step
    discrete_groups = np.array([
        discretise(
            model.capital[:,t].reshape(-1, 1), 
            n_bins=2, 
            strategy='quantile',
        ).flatten() for t in range(model.n_steps+1)
    ])
    comparison_frame['Income'] = discrete_groups.ravel()
    comparison_frame['Income'] = comparison_frame['Income'].replace(
        {0:'Low (<50%)',1:'High (>=50%)'}
        )
    return comparison_frame


# Helper function to run the model in parallel 
@ray.remote
def run_model(par_values):

    # Get the parameter values and run the model
    par_dict = par_values.to_dict()
    model = WellbeingSim(params=par_dict)
    model.run_simulation()

    # Get values from the simulation
    cap = model.capital
    init_cap = cap[:,0]
    swb = model.wellbeing
    shocked = model.random_idx==1

    # Calculate metrics
    instability = calc_instability(swb)
    recovery_time = calc_recovery_time(swb)
    longterm_change = calc_longterm_change(swb)
    sum_abs_diffs = np.abs(swb[:,50:] - swb[:,49,np.newaxis]).sum(axis=1)
    cap_diff = (cap[:,99] - init_cap) / 1000.
    swb_diff = swb[:,99] - swb[:,0]
    
    t = np.array(list(range(swb[:,50:].shape[1])))
    mean_swb = swb[:,50:].mean(axis=0)
    mean_cap = cap[:,50:].mean(axis=0)
    const_swb, slope_swb = sm.OLS(
        mean_swb, sm.add_constant(t), missing='drop'
        ).fit().params
    const_cap, slope_cap = sm.OLS(
        mean_cap / 1000., sm.add_constant(t), missing='drop'
        ).fit().params
    slope_ratio = slope_swb / slope_cap

    # welfare calculations
    welfare_cap_0 = calc_welfare(cap[:,0])
    welfare_swb_0 = calc_welfare(swb[:,0])
    welfare_cap_49 = calc_welfare(cap[:,30:50])
    welfare_swb_49 = calc_welfare(swb[:,30:50])
    welfare_cap_99 = calc_welfare(cap[:,80:100])
    welfare_swb_99 = calc_welfare(swb[:,80:100])
    delta_welfare_cap = welfare_cap_99 - welfare_cap_49
    delta_welfare_swb = welfare_swb_99 - welfare_swb_49
    delta_welfare_cap_init = welfare_cap_99 - welfare_cap_0
    delta_welfare_swb_init = welfare_swb_99 - welfare_swb_0

    outputs = np.array([longterm_change, instability, 
                        recovery_time, sum_abs_diffs])
    if par_dict['p']>0:
        idx_srtd_init_cap = init_cap.argsort()
        vulnerable = idx_srtd_init_cap[:int(len(idx_srtd_init_cap)/2)]
        outputs_for_shocked = outputs[:,shocked]
        outputs_for_vulnerable = outputs[:,vulnerable]
    else:
        outputs_for_shocked = np.array([[0,0]]*len(outputs))
        outputs_for_vulnerable = np.array([[0,0]]*len(outputs))

    # Combine all output metrics
    output_metrics = np.hstack((
        # Calculated output metrics averaged over system (or std)
        slope_ratio,
        delta_welfare_swb,
        delta_welfare_cap,
        delta_welfare_swb_init,
        delta_welfare_cap_init,
        outputs.mean(axis=1),
        outputs_for_vulnerable.mean(axis=1),
        outputs_for_shocked.mean(axis=1),
        outputs.std(axis=1),
        outputs_for_vulnerable.std(axis=1),
        outputs_for_shocked.std(axis=1)
    ))
    
    return output_metrics


@ray.remote
def run_wellbeing_traps(param_iterable):
    
    # Initiate the society
    (cap, swb, feedback), params = param_iterable
    model = WellbeingSim(params=params)

    # Change the parameters of the specific, randomly chosen individual
    random_idx = np.random.randint(low=0, high=model.N)
    model.capital[random_idx,0] = model.capital[random_idx,1] = cap
    if params['p']>0:
        model.capital[random_idx,50] = params['int_size']
    model.al_capital[random_idx,0] = model.al_capital[random_idx,1] = cap
    model.wellbeing[random_idx,0] = model.wellbeing[random_idx,1] = swb
    model.swb_setpoint[random_idx] = swb

    # Change values like this because of read-only error
    alpha = model.alpha.copy()
    alpha[random_idx] = 0.2
    model.alpha = alpha
    wb_to_cap = model.wb_to_cap.copy()
    wb_to_cap[random_idx] = feedback
    model.wb_to_cap = wb_to_cap
    soc_comp_w = model.soc_comp_w.copy()
    soc_comp_w[random_idx] = 0.5
    model.soc_comp_w = soc_comp_w

    # Run the model and extract final income of the specific agent (#random_idx)
    model.run_simulation()
    final_capital = model.capital[random_idx,99]
    final_wellbeing = model.wellbeing[random_idx,99]
    longterm_change = calc_longterm_change(model.wellbeing)[random_idx]
    return [final_capital, final_wellbeing, longterm_change] 