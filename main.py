from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Union
import numpy as np
from sympy import (
    symbols, sympify, integrate, solveset, Interval, oo,
    diff, factorial, Sum, latex, simplify, N, sstr
)
from scipy.integrate import simpson, quad
import matplotlib.pyplot as plt
import sympy as sp
import os
import random

app = FastAPI()

# Preparar carpeta estática para guardar imágenes
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Variables simbólicas
x, n = symbols('x n')

# Entrada del usuario
class InputDatos(BaseModel):
    funcion: str
    a: Union[str, float]
    b: Union[str, float]
    n_terminos: int = Field(default=10, ge=1, le=20)
    tolerancia: float = Field(default=1e-6, ge=1e-10)

# Función para graficar f(x) y su Taylor
def generar_grafica_serie_taylor(expr, n_terminos=10, ruta_salida="static/taylor.png"):
    taylor_expr = sum([
        sp.simplify(sp.diff(expr, x, i).subs(x, 0) / sp.factorial(i)) * x**i
        for i in range(n_terminos)
    ])
    f_lamb = sp.lambdify(x, expr, 'numpy')
    taylor_lamb = sp.lambdify(x, taylor_expr, 'numpy')
    X = np.linspace(-2, 2, 500)
    Y_f = f_lamb(X)
    Y_taylor = taylor_lamb(X)
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y_f, label=r"$f(x)$", color='blue', linewidth=2)
    plt.plot(X, Y_taylor, label=f"Taylor orden {n_terminos}", color='red', linestyle='--')
    plt.title("f(x) vs Serie de Taylor centrada en x = 0")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle=':')
    plt.axvline(0, color='gray', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()

# Función para exportar texto compatible con GeoGebra
def exportar_para_geogebra(expr):
    expr_str = sstr(expr)
    expr_str = expr_str.replace('**', '^')      # Potencias
    expr_str = expr_str.replace('*', '')        # Evita 'x*y'
    expr_str = expr_str.replace(')/', ')/')     # Paréntesis ya cerrados
    return expr_str

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        a_eval = float(sympify(datos.a))
        b_eval = float(sympify(datos.b))

        if a_eval >= b_eval:
            raise HTTPException(status_code=400, detail="El límite inferior debe ser menor que el superior.")

        f = sympify(datos.funcion)
        if str(f) == 'sin(x)/x':
            def f_lambda(x_val):
                return np.where(x_val == 0, 1.0, np.sin(x_val)/x_val)
        else:
            f_lambda = sp.lambdify(x, f, modules=['numpy'])

        posibles_sing = solveset(1/f, x, domain=Interval(a_eval, b_eval))
        advertencias = []
        for p in posibles_sing:
            try:
                val = float(p.evalf())
                if a_eval < val < b_eval:
                    advertencias.append("⚠️ Posible singularidad en el intervalo.")
            except:
                continue

        try:
            F_exacta = integrate(f, x)
            F_exacta_tex = f"$$ {latex(F_exacta)} $$"
        except:
            F_exacta = "No tiene primitiva elemental"
            F_exacta_tex = "No tiene primitiva elemental"

        funciones_especiales = []
        if "erf" in str(F_exacta):
            funciones_especiales.append({
                "funcion": "erf(x)",
                "latex": r"\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt",
                "descripcion": "Función error asociada a integrales gaussianas."
            })
        if "Si" in str(F_exacta):
            funciones_especiales.append({
                "funcion": "Si(x)",
                "latex": r"\mathrm{Si}(x) = \int_0^x \frac{\sin(t)}{t} \, dt",
                "descripcion": "Función seno integral, primitiva de sin(x)/x."
            })

        resultado_exacto = integrate(f, (x, a_eval, b_eval))
        resultado_exacto_val = float(N(resultado_exacto))
        resultado_exacto_tex = f"$$ {latex(resultado_exacto)} $$"

        a_taylor = 0
        serie_general = Sum(diff(f, x, n).subs(x, a_taylor) / factorial(n) * (x - a_taylor)**n, (n, 0, oo))
        sumatoria_general_tex = f"$$ {latex(serie_general)} $$"

        explicacion_taylor = (
            f"**Para la función** \\( {latex(f)} \\), "
            f"**el desarrollo en serie de Taylor alrededor de** \\( x = 0 \\) **es:**"
        )

        terminos = []
        for i in range(datos.n_terminos):
            deriv_i = diff(f, x, i).subs(x, a_taylor)
            term = simplify(deriv_i / factorial(i)) * (x - a_taylor)**i
            terminos.append(term)

        f_series_sumada = " + ".join([latex(term) for term in terminos]) + r" + \cdots"
        f_series_tex = f"$$ {latex(f)} = {f_series_sumada} $$"

        f_series_expr = sum(terminos)
        F_aproximada = integrate(f_series_expr, x)
        F_aproximada_tex = f"$$ {latex(F_aproximada)} $$"

        integral_definida_tex = f"$$ \\int_{{{a_eval}}}^{{{b_eval}}} {latex(f)} \\, dx $$"

        puntos = np.linspace(a_eval, b_eval, 1000)
        y_vals = f_lambda(puntos)
        dx = (b_eval - a_eval) / 1000

        integral_simpson = simpson(y_vals, dx=dx)
        integral_romberg, _ = quad(f_lambda, a_eval, b_eval)
        integral_gauss, _ = quad(f_lambda, a_eval, b_eval)

        def monte_carlo_integration(f, a, b, n_samples=10000):
            total = 0
            for _ in range(n_samples):
                x_rand = random.uniform(a, b)
                total += f(x_rand)
            return (b - a) * total / n_samples

        integral_montecarlo = monte_carlo_integration(
            lambda x_val: f_lambda(np.array([x_val]))[0], a_eval, b_eval
        )

        generar_grafica_serie_taylor(f, datos.n_terminos)

        geogebra_expresiones = {
            "funcion": f"f(x) = {exportar_para_geogebra(f)}",
            "taylor": f"T(x) = {exportar_para_geogebra(f_series_expr)}"
        }

        return {
            "primitiva_real": F_exacta_tex,
            "funciones_especiales": funciones_especiales,
            "serie_taylor_general": sumatoria_general_tex,
            "explicacion_taylor_general": explicacion_taylor,
            "serie_taylor_finita": f_series_tex,
            "integral_definida_exacta": integral_definida_tex,
            "integral_definida_valor": resultado_exacto_tex,
            "valor_numerico_exacto": resultado_exacto_val,
            "metodos_numericos": {
                "simpson": integral_simpson,
                "romberg": integral_romberg,
                "cuadratura_gaussiana": integral_gauss,
                "montecarlo": integral_montecarlo,
            },
            "advertencias": advertencias,
            "grafica_url": "https://gpt-integrador.onrender.com/static/taylor.png",
            "geogebra_expresiones": geogebra_expresiones
        }

    except Exception as e:
        return {"error": str(e)}
