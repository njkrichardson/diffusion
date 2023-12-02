import matplotlib 
import matplotlib.pyplot as plt
from jax import jacobian, vmap 
import jax.numpy as np 
import jax.random as npr 
from jax.flatten_util import ravel_pytree

ndarray: type = np.ndarray 

def levenberg_marquardt(f: callable, x: ndarray, penalty: float=1.) -> ndarray: 
    max_iterations: int = 100 
    convergence_tolerance: float = 1e-03
    accept_iterate: callable = lambda f_x, tentative: np.linalg.norm(f_x)**2 > np.linalg.norm(f(tentative))**2

    small_residual: callable = lambda x: np.linalg.norm(f(x))**2 <= convergence_tolerance
    small_optimality_residual: callable = lambda J, x: np.linalg.norm(2 * J.T @ J)**2 <= convergence_tolerance
    converged: callable = lambda J, x: small_residual(x) or small_optimality_residual(J, x)

    for i in range(max_iterations): 
        J: ndarray = jacobian(f)(x) 
        f_x: ndarray = f(x) 
        A: ndarray = np.block([[J], [np.sqrt(penalty) * np.eye(x.size)]])
        b: ndarray = np.block([J @ x - f_x, np.sqrt(penalty) * x]).reshape(-1)
        breakpoint()
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
    convergence_tolerance: float = 1e-02
    converged: callable = lambda x: np.linalg.norm(g(x)) < convergence_tolerance 

    max_iterations: int = 100

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
    
    u: ndarray = npr.uniform(key, (T, 2), minval=0., maxval=np.pi/2)
    state = start 
    states = [state]

    for control in u: 
        state = dynamics(state, control)
        states.append(state) 

    x: ndarray = np.array(states)[1:-1, :]
    
    s, unravel = ravel_pytree((u, x))

    def constraints(s: ndarray): 
        u, x = unravel(s) 
        actual: ndarray = vmap(dynamics, in_axes=(0, 0))(np.vstack((start, x)), u) 
        desired: ndarray = np.vstack((x, end)) 
        error: ndarray = actual - desired
        return vmap(np.dot, in_axes=(0, 0))(error, error).sum()

    def f(s: ndarray, roughness_penalty: float=10.) -> ndarray: 
        u, x = unravel(s) 
        roughness = np.diff(u, 1, 0) 

        return vmap(np.dot, in_axes=(0, 0))(u, u).sum() + roughness_penalty * vmap(np.dot, in_axes=(0, 0))(roughness, roughness).sum()

    u, x = unravel(augmented_lagrangian(f, constraints, s, T))

    return u, x

T: int = 50
start: ndarray = np.zeros(3)
end: ndarray = np.array([1, 0., 0.]) 
u, x = car_control(start, end, T=T) 

matplotlib.use("Agg") 
def plot_trajectory(states: ndarray): 
    plt.figure() 
    plt.scatter(states[:, 0], states[:, 1])
    plt.savefig("driving")
    plt.close()

plot_trajectory(np.vstack((start, x, dynamics(x[-1], u[-1]))))
