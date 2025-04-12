import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def generar_grafica_serie_taylor(ruta_salida="static/taylor.png", n_terminos=10):
    x = sp.Symbol('x')
    f = sp.exp(-x**2)

    taylor_expr = sum([
        sp.simplify(sp.diff(f, x, i).subs(x, 0) / sp.factorial(i)) * x**i
        for i in range(n_terminos)
    ])

    f_lamb = sp.lambdify(x, f, 'numpy')
    taylor_lamb = sp.lambdify(x, taylor_expr, 'numpy')

    X = np.linspace(-1.5, 1.5, 400)
    Y_f = f_lamb(X)
    Y_taylor = taylor_lamb(X)

    plt.figure(figsize=(10, 6))
    plt.plot(X, Y_f, label=r"$e^{-x^2}$", color='blue', linewidth=2)
    plt.plot(X, Y_taylor, label=f"Serie de Taylor (orden {n_terminos})", color='red', linestyle='--')
    plt.title("Comparación entre $e^{-x^2}$ y su serie de Taylor centrada en $x=0$")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0, color='gray', linestyle=':')
    plt.axvline(0, color='gray', linestyle=':')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(ruta_salida)
    plt.close()
