from time import perf_counter
import matplotlib 
from matplotlib.patches import Rectangle 
import matplotlib.pyplot as plt
from jax import jacobian, vmap, jit
import jax.numpy as np 
import jax.random as npr 
from jax.flatten_util import ravel_pytree

ndarray: type = np.ndarray 

def human_seconds_str(seconds: int) -> str:
    units: Tuple[str] = ("seconds", "milliseconds", "microseconds")
    power: int = 1

    for unit in units:
        if seconds > power:
            return f"{seconds:.1f} {unit}"

        seconds *= 1000

    return f"{int(seconds)} nanoseconds"

class PythonProfiler: 
    """Python profiling context manager. 
    """
    def __init__(self, identifier: str, **kwargs): 
        self.identifier: str = identifier

    def __enter__(self): 
        self.start_time: float = perf_counter()

    def __exit__(self, type, value, traceback): 
        run_time: float = perf_counter() - self.start_time
        report_str: str = f"region [{self.identifier}]: {human_seconds_str(run_time)}"
        print(report_str)

def levenberg_marquardt(f: callable, x: ndarray, penalty: float=1.) -> ndarray: 
    max_iterations: int = 100 
    convergence_tolerance: float = 1e-03
    accept_iterate: callable = jit(lambda f_x, tentative: np.linalg.norm(f_x)**2 > np.linalg.norm(f(tentative))**2)

    small_residual: callable = jit(lambda x: np.linalg.norm(f(x))**2 <= convergence_tolerance)
    small_optimality_residual: callable = jit(lambda J, x: np.linalg.norm(2 * J.T @ J)**2 <= convergence_tolerance)
    converged: callable = lambda J, x: small_residual(x) or small_optimality_residual(J, x)
    jac_f: callable = jit(jacobian(f))

    for i in range(max_iterations): 
        J: ndarray = jac_f(x) 
        f_x: ndarray = f(x) 
        A: ndarray = np.block([[J], [np.sqrt(penalty) * np.eye(x.size)]])
        b: ndarray = np.block([J @ x - f_x, np.sqrt(penalty) * x]).reshape(-1)
        tentative_iterate: ndarray = np.linalg.solve(A.T @ A, A.T @ b)


        if accept_iterate(f_x, tentative_iterate): 
            x = tentative_iterate
            penalty *= 0.8 
        else: 
            penalty *= 2

        current_objective: float = np.linalg.norm(f(x))**2 

        if current_objective <= convergence_tolerance or small_optimality_residual(J, x): 
            break 

    return x

def augmented_lagrangian(f: callable, g: callable, x: ndarray, num_constraints: int) -> ndarray: 
    penalty: float = 1. 
    lagrange_multiplier: ndarray = np.zeros(num_constraints)
    convergence_tolerance: float = 1e-04
    converged: callable = lambda x: np.linalg.norm(g(x)) < convergence_tolerance 

    max_iterations: int = 50

    for i in range(max_iterations): 
        surrogate: callable = lambda x: np.linalg.norm(f(x))**2 + penalty * np.linalg.norm(g(x) + lagrange_multiplier/(2 * penalty))**2
        new_x: ndarray = levenberg_marquardt(surrogate, x)
        lagrange_multiplier += 2 * penalty * g(new_x) 

        if np.linalg.norm(g(new_x)) >= 0.25 * np.linalg.norm(g(x)): 
            penalty *= 2. 

        x = new_x 

        if converged(x): 
            break 

        print(f"Iteration: {i}\t||f(x)||: {np.linalg.norm(f(x))}\t||g(x)||: {np.linalg.norm(g(x))}")

    return x

def dynamics(x: ndarray, u: ndarray, dt=0.1, L=0.1) -> ndarray: 
    return x + dt * u[0] * np.array([np.cos(x)[2], np.sin(x)[2], np.tan(u)[1] / L])

def car_control(start: ndarray, end: ndarray, T: int=50): 
    key = npr.PRNGKey(0)
    
    speed_key, steering_key = npr.split(key)
    speed: ndarray = npr.uniform(speed_key, (T,), minval=0., maxval=0.1)
    steering: ndarray = npr.uniform(steering_key, (T,), minval=0., maxval=np.pi / 2)
    u: ndarray = np.vstack((speed, steering)).T
    x: ndarray = np.zeros((T-1, 3))
    s, unravel = ravel_pytree((u, x))

    @jit 
    def constraints(s: ndarray): 
        u, x = unravel(s) 
        actual: ndarray = vmap(dynamics, in_axes=(0, 0))(np.vstack((start, x)), u) 
        desired: ndarray = np.vstack((x, end)) 
        error: ndarray = actual - desired
        return vmap(np.dot, in_axes=(0, 0))(error, error).sum()

    @jit
    def f(s: ndarray, roughness_penalty: float=10.) -> ndarray: 
        u, x = unravel(s) 
        roughness = np.diff(u, 1, 0) 
        return vmap(np.dot, in_axes=(0, 0))(u, u).sum() + roughness_penalty * vmap(np.dot, in_axes=(0, 0))(roughness, roughness).sum()

    u, x = unravel(augmented_lagrangian(f, constraints, s, T))

    return u, x

T: int = 50
start: ndarray = np.zeros(3)
end: ndarray = np.array([0, 1., np.pi / 2]) 
u, x = car_control(start, end, T=T) 


matplotlib.use("Agg") 
def plot_trajectory(states: ndarray): 
    fig, ax = plt.subplots() 

    for state in states: 
        rect = Rectangle((state[0], state[1]), 0.07, 0.04, angle=state[2] * (180 / np.pi), facecolor='k', alpha=0.3) 
        ax.add_patch(rect)

    ax.set_xlim(-1., 1.) 
    ax.set_ylim(-1., 1.) 
    plt.savefig("driving")
    plt.close()

plot_trajectory(np.vstack((start, x, dynamics(x[-1], u[-1])))[::2, ...])
