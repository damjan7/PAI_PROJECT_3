import random
import os
import typing
import logging
import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import *
from scipy.stats import norm
from scipy.stats import bernoulli
from sklearn.gaussian_process import GaussianProcessRegressor

EXTENDED_EVALUATION = False
# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.


""" Solution """


class BO_algo(object):
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.previous_points = []
        # IMPORTANT: DO NOT REMOVE THOSE ATTRIBUTES AND USE sklearn.gaussian_process.GaussianProcessRegressor instances!
        # Otherwise, the extended evaluation will break.

        # We define the kernel (prior) for the objective and constraint function
        # We introduce the WhiteKernel to encapsulate some noise in the measurements
        obj_kernel = ConstantKernel(1.5)*RBF(1.5, length_scale_bounds="fixed") + WhiteKernel(0.01, noise_level_bounds="fixed")
        con_kernel = ConstantKernel(3.5)*RBF(2, length_scale_bounds="fixed") + WhiteKernel(0.005, noise_level_bounds="fixed")

        # We initialize to GP to track the objective and constraint function
        self.constraint_model = GaussianProcessRegressor(kernel=con_kernel)  # TODO : GP model for the constraint function
        self.objective_model = GaussianProcessRegressor(kernel=obj_kernel) # TODO : GP model for your acquisition function

    def next_recommendation(self) -> np.ndarray:
        """
        Recommend the next input to sample.
        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        size_of_data = len(self.previous_points)

        if(size_of_data == 0):
            explor_vs_exploi = 1
        else:
            explor_vs_exploi = np.exp(-size_of_data)

        # We toss a coin to decide if we should choose the next point according to the acquisition function
        # or if we should choose a random points
        coin = bernoulli.rvs(p=explor_vs_exploi, size=1)

        if self.previous_points:
            # We already have data points
            if coin == 1:
                # The coin flip decided to explore
                return np.ndarray((1, 2), buffer=np.random.uniform(0, 6, 2))
            else:
                # The coin flip decided to exploit
                return self.optimize_acquisition_function()
        else:
            # We don't have any data points so we start with the point (3,3)
            return np.atleast_2d([3,3])



    def optimize_acquisition_function(self) -> np.ndarray:  # DON'T MODIFY THIS FUNCTION
        """
        Optimizes the acquisition function.
        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that approximately maximizes the acquisition function.
        """

        def objective(x: np.array):
            return - self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain_x[0, 0] + (domain_x[0, 1] - domain_x[0, 0]) * \
                 np.random.rand(1)
            x1 = domain_x[1, 0] + (domain_x[1, 1] - domain_x[1, 0]) * \
                 np.random.rand(1)
            result = fmin_l_bfgs_b(objective, x0=np.array([x0, x1]), bounds=domain_x,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain_x[0]))
            f_values.append(result[1])

        ind = np.argmin(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the acquisition function.
        Parameters
        ----------
        x: np.ndarray
            point in the domain of f
        Returns
        ------
        af_value: float
            value of the acquisition function at x
        """
        # Get the predictive mean and variance of the objective function
        obj_mean, obj_sigma = self.objective_model.predict( np.atleast_2d(x), return_std=True)

        # get the predictive mean and variance of the constraint function
        con_mean, con_sigma = self.constraint_model.predict( np.atleast_2d(x), return_std=True)

        # We calculate the Expected Improvement as explained in the paper of Snoek et al.

        # Step 1: Calculate f(x_best)
        min_acq_function = min([point[2] for point in self.previous_points])

        # Step 2: Calculate gamma(x)
        gamma = (min_acq_function-obj_mean)/obj_sigma

        # Step 3: Calculate PHI(x)
        phi = scipy.stats.norm.cdf(gamma)

        # Step 4: Calculate Expected Improvement
        EI = obj_sigma*(gamma*phi + scipy.stats.norm.pdf(gamma))

        # Step 5: Incorporate the constraint function into the EI according to Gelbart et al (7)
        EI_constraint = scipy.stats.norm(con_mean, con_sigma).cdf(0) # prob that the constraint is satisfied (c<0)
        #EI = EI*EI_constraint
        EI = EI * np.exp(self.constraint_model.log_marginal_likelihood())

        return EI

    def add_data_point(self, x: np.ndarray, z: float, c: float):
        """
        Add data points to the model.
        Parameters
        ----------
        x: np.ndarray
            point in the domain of f
        z: np.ndarray
            value of the acquisition function at x
        c: np.ndarray
            value of the condition function at x
        """

        assert x.shape == (1, 2)
        self.previous_points.append([float(x[:, 0]), float(x[:, 1]), float(z), float(c)])

        points = np.array([[x1, x2] for x1, x2, z, c in self.previous_points])
        acq_value = np.array([z for x1, x2, z, c in self.previous_points])
        cond_value = np.array([c for x1, x2, z, c in self.previous_points])

        self.constraint_model.fit(points, cond_value)
        self.objective_model.fit(points, acq_value)

    def get_solution(self) -> np.ndarray:
        """
        Return x_opt that is believed to be the minimizer of f.
        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        con_sat = []
        try:
            x_opt = min([point for point in self.previous_points if point[3] < 0], key = lambda x: x[2])[:2]
        except:
            x_opt = min([point for point in self.previous_points], key = lambda x: x[3])[:2] #if none of the observed datapoints has c<0 we take the one with smallest c.
        return np.atleast_2d(x_opt)
        """
        try:
            con_sat = []
            # We are interested in the cases where the condition value is < 0
            for point in self.previous_points:
                if point[3] < 0:
                    con_sat.append[point]
            # From all the points that satisfy the condition we find the point which has the min function value
            opt_point = min([point for point in con_sat], key = lambda x: x[2])[:2]
        except:
            # if none of the points satisfy the condition we simply take the point with the lowest constraint values
            opt_point = min([point for point in self.previous_points], key=lambda x: x[3])[:2]
        return  np.atleast_2d(opt_point)
        """


""" 
    Toy problem to check  you code works as expected
    IMPORTANT: This example is never used and has nothing in common with the task you
    are evaluated on, it's here only for development and illustration purposes.
"""
domain_x = np.array([[0, 6], [0, 6]])
EVALUATION_GRID_POINTS = 250
CONSTRAINT_OFFSET = - 0.8  # This is an offset you can change to make the constraint more or less difficult to fulfill
LAMBDA = 0.0  # You shouldn't change this value


def check_in_domain(x) -> bool:
    """Validate input"""
    x = np.atleast_2d(x)
    v_dim_0 = np.all(x[:, 0] >= domain_x[0, 0]) and np.all(x[:, 0] <= domain_x[0, 1])
    v_dim_1 = np.all(x[:, 1] >= domain_x[1, 0]) and np.all(x[:, 0] <= domain_x[1, 1])

    return v_dim_0 and v_dim_1


def f(x) -> np.ndarray:
    """Dummy objective"""
    l1 = lambda x0, x1: np.sin(x0) + x1 - 1

    return l1(x[:, 0], x[:, 1])


def c(x) -> np.ndarray:
    """Dummy constraint"""
    c1 = lambda x, y: np.cos(x) * np.cos(y) - 0.1

    return c1(x[:, 0], x[:, 1]) - CONSTRAINT_OFFSET


def get_valid_opt(f, c, domain) -> typing.Tuple[float, float, np.ndarray, np.ndarray]:
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(domain[0, 0], domain[0, 1], nx)
    y = np.linspace(domain[1, 0], domain[1, 1], ny)
    xv, yv = np.meshgrid(x, y)
    samples = np.array([xv.reshape(-1), yv.reshape(-1)]).T

    true_values = f(samples)
    true_cond = c(samples)
    valid_data_idx = np.where(true_cond < LAMBDA)[0]
    f_opt = np.min(true_values[np.where(true_cond < LAMBDA)])
    x_opt = samples[valid_data_idx][np.argmin(true_values[np.where(true_cond < LAMBDA)])]
    f_max = np.max(np.abs(true_values))
    x_max = np.argmax(np.abs(true_values))
    return f_opt, f_max, x_opt, x_max


def perform_extended_evaluation(agent, output_dir='./'):
    fig = plt.figure(figsize=(25, 5), dpi=50)
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(0.0, 6.0, nx)
    y = np.linspace(0.0, 6.0, ny)
    xv, yv = np.meshgrid(x, y)
    x_b, y_b = agent.get_solution()
    samples = np.array([xv.reshape(-1), yv.reshape(-1)]).T
    predictions, stds = agent.objective_model.predict(samples, return_std=True)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    conds = agent.constraint_model.predict(samples)
    conds = np.reshape(conds, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    true_values = f(samples)
    true_cond = c(samples)
    conditions_verif = (true_cond < LAMBDA).astype(float)
    conditions_with_nans = 1 - np.copy(conditions_verif)
    conditions_with_nans[np.where(conditions_with_nans == 0)] = np.nan
    conditions_with_nans = np.reshape(conditions_with_nans, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    valid_data_idx = np.where(true_cond < LAMBDA)[0]

    f_opt = np.min(true_values[np.where(true_cond < LAMBDA)])
    x_opt = samples[valid_data_idx][np.argmin(true_values[np.where(true_cond < LAMBDA)])]

    sampled_point = np.array(agent.previous_points)

    ax_condition = fig.add_subplot(1, 4, 4)
    im_cond = ax_condition.pcolormesh(xv, yv, conds.reshape((EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)), shading='auto',
                                      linewidth=0)
    im_cond.set_edgecolor('face')
    fig.colorbar(im_cond, ax=ax_condition)
    ax_condition.scatter(sampled_point[:, 0], sampled_point[:, 1], cmap='Blues', marker='x',
                         label='Sampled Point by BO', antialiased=True, linewidth=0)
    ax_condition.pcolormesh(xv, yv, conditions_with_nans, shading='auto', cmap='Reds', alpha=0.7, vmin=0, vmax=1.0,
                            linewidth=0, antialiased=True)
    ax_condition.set_title('Constraint GP Posterior +  True Constraint (Red is Infeasible)')
    ax_condition.legend(fontsize='x-small')

    ax_gp_f = fig.add_subplot(1, 4, 2, projection='3d')
    ax_gp_f.plot_surface(
        X=xv,
        Y=yv,
        Z=predictions,
        rcount=100,
        ccount=100,
        linewidth=0,
        antialiased=False
    )
    ax_gp_f.set_title('Posterior 3D for Objective')

    ax_gp_c = fig.add_subplot(1, 4, 3, projection='3d')
    ax_gp_c.plot_surface(
        X=xv,
        Y=yv,
        Z=conds,
        rcount=100,
        ccount=100,
        linewidth=0,
        antialiased=False
    )
    ax_gp_c.set_title('Posterior 3D for Constraint')

    ax_predictions = fig.add_subplot(1, 4, 1)
    im_predictions = ax_predictions.pcolormesh(xv, yv, predictions, shading='auto', label='Posterior',linewidth=0, antialiased=True)
    im_predictions.set_edgecolor('face')
    fig.colorbar(im_predictions, ax=ax_predictions)
    ax_predictions.pcolormesh(xv, yv, conditions_with_nans, shading='auto', cmap='Reds', alpha=0.7, vmin=0, vmax=1.0,
                              label=' True Infeasible',linewidth=0, antialiased=True)
    ax_predictions.scatter(x_b, y_b, s=20, marker='x', label='Predicted Value by BO')
    ax_predictions.scatter(x_opt[0], x_opt[1], s=20, marker='o', label='True Optimimum Under Constraint')
    ax_predictions.set_title('Objective GP Posterior + True Constraint (Red is Infeasible)')
    ax_predictions.legend(fontsize='x-small')
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    plt.show()


def train_on_toy(agent, iteration):
    logging.info('Running model on toy example.')
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    for j in range(iteration):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain_x.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain_x.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal(size=(x.shape[0],), scale=0.01)
        cost_val = c(x) + np.random.normal(size=(x.shape[0],), scale=0.005)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain_x.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain_x.shape[0]})"

    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    f_opt, f_max, x_opt, x_max = get_valid_opt(f, c, domain_x)
    if c(solution) > 0.0:
        regret = 1
    else:
        regret = (f(solution) - f_opt) / f_max

    print(f'Optimal value: {f_opt}\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')
    return agent


def main():
    logging.warning(
        'This main method is for illustrative purposes only and will NEVER be called by the checker!\n'
        'The checker always calls run_solution directly.\n'
        'Please implement your solution exclusively in the methods and classes mentioned in the task description.'
    )

    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = BO_algo()

    agent = train_on_toy(agent, 20)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(agent)


if __name__ == "__main__":
    main()
