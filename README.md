# euler_equations_FV_scheme

For use with Python >= 3.6.
To run main.py or advection_test.py in terminal, enter, for example, "ipython main.py" with no inputs in the directory where the codes are stored.
Each routine returns an animation of the PDE solution, given the parameters inside the relevant code.
solvers.py is a class containing the routines for the FV scheme that are called in the other driver codes.
In this case the FV scheme is the HLLE approximate Riemann solver.
