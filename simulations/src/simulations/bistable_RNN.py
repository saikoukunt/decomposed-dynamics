from .rnn import RNN, pos_tanh
from jax import Array
import jax.numpy as jnp

class bistable_RNN(RNN):
    def __init__(self, W_exc:float, W_inhib:float, dt: float = 0.05, activation_fn: Callable = pos_tanh):
        W_rec = jnp.zeros((2,2))
        W_rec[0,0] = W_exc
        W_rec[1,1] = W_exc
        W_rec[0,1] = W_inhib
        W_rec[1, 0] = W_inhib

        W_in[0, 0] = 1
        W_in[1, 0] = -1
        W_in[:, 1] = 1

        super.__init__(state_dim=2, input_dim=2, dt, W_rec, W_in, activation_fn)


    def sample_trajectory(self, x_0, coherence: float, E: float, sigma: float,T: int, dt:float) -> Array:
        num_iter = floor(T/dt)
        inputs = jnp.zeros((num_iter, 2))
        inputs[:, 0] = E
        inputs[:, 1] = coherence

        return self.euler_sequence(x_0, inputs, sigma, num_iter) 
        
        
