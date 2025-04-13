from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union
import numpy as np
from sympy import (
    symbols, sympify, integrate, solveset, Interval, oo,
    diff, factorial, Sum, latex, simplify, N, sstr, Function, erf, sqrt, pi
)
from scipy.integrate import simpson, quad
import sympy as sp
import random

app = FastAPI()

x, n = symbols('x n')

class InputDatos(BaseModel):
    funcion: str
    a: Union[str, float]
    b: Union[str, float]
    n_terminos: int = Field(default=10, ge=1, le=20)
    tolerancia: float = Field(default=1e-6, ge=1e-10)

def exportar_para_geogebra(expr):
    expr_str = sstr(expr)
    expr_str = expr_str.replace('**', '^').replace('*', '')
    return expr_str

def obtener_funciones_especiales(expr):
    definiciones = []

    if "Si" in str(expr):
        definiciones.append({
            "funcion": "Si(x)",
            "latex": r"\mathrm{Si}(x) = \int_0^x \frac{\sin(t)}{t} \, dt",
            "descripcion": "La función seno integral aparece como primitiva de \\( \\frac{\\sin(x)}{x} \\)."
        })
    if "erf" in str(expr):
        definiciones.append({
            "funcion": "erf(x)",
            "latex": r"\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt",
            "descripcion": "La función error aparece como primitiva de \\( e^{-x^2} \\)."
        })
    if "Li" in str(expr):
        definiciones.append({
            "funcion": "Li(x)",
            "latex": r"\mathrm{Li}(x) = \int_0^x \frac{dt}{\log(t)}",
            "descripcion": "La función logaritmo integral aparece al integrar \\( \\frac{1}{\\log(x)} \\)."
        })
    if "Ci" in str(expr):
        definiciones.append({
            "funcion": "Ci(x)",
            "latex": r"\mathrm{Ci}(x) = -\int_x^\infty \frac{\cos(t)}{t} \, dt",
            "descripcion": "La función coseno integral aparece en análisis armónico y transformadas."
        })
    if "gamma" in str(expr):
        definiciones.append({
            "funcion": "Gamma(x)",
            "latex": r"\Gamma(x) = \int_0^\infty t^{x-1} e^{-t} \, dt",
            "descripcion": "Extiende el factorial a los números reales y complejos."
        })
    if "beta" in str(expr):
        definiciones.append({
            "funcion": "Beta(x, y)",
            "latex": r"B(x, y) = \int_0^1 t^{x-1} (1 - t)^{y-1} \, dt",
            "descripcion": "Relacionada con la función Gamma. Aparece en teoría de probabilidad."
        })

    return definiciones

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        a_sym = sympify(datos.a)
        b_sym = sympify(datos.b)
        a_eval = float(N(a_sym))
        b_eval = float(N(b_sym))

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

        funciones_especiales = []
        F_exacta_tex = ""
        valor_simbolico = "Valor simbólico no disponible"

        if str(f) == 'sin(x)/x':
            F_exacta_tex = r"\mathrm{Si}(x)"
            valor_simbolico = rf"\mathrm{{Si}}({latex(b_sym)}) - \mathrm{{Si}}({latex(a_sym)})"
        elif str(f) == 'exp(-x**2)':
            valor_simbolico = rf"2 \sqrt{{\pi}} \cdot \mathrm{{erf}}({latex(b_sym)})" if a_sym == -b_sym else \
                              rf"\sqrt{{\pi}} \cdot (\mathrm{{erf}}({latex(b_sym)}) - \mathrm{{erf}}({latex(a_sym)}))"
            F_exacta_tex = r"\frac{\sqrt{\pi}}{2} \cdot \mathrm{erf}(x)"
        else:
            try:
                F_expr = integrate(f, x)
                F_exacta_tex = f"{latex(F_expr)}"
                valor_simbolico = rf"{latex(F_expr.subs(x, b_sym) - F_expr.subs(x, a_sym))}"
            except:
                F_exacta_tex = "No tiene primitiva elemental"

        funciones_especiales += obtener_funciones_especiales(f)

        resultado_exacto = integrate(f, (x, a_eval, b_eval))
        resultado_exacto_val = float(N(resultado_exacto))
        resultado_exacto_tex = f"{latex(resultado_exacto)}"

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

        integral_definida_tex = f"$$ \\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\, dx $$"

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

        geogebra_expresiones = {
            "funcion": f"f(x) = {exportar_para_geogebra(f)}",
            "taylor": f"T(x) = {exportar_para_geogebra(f_series_expr)}",
            "area_comando": f"Integral(f, {a_eval}, {b_eval})"
        }

        return {
            "primitiva_real": f"$$ {F_exacta_tex} $$",
            "funciones_especiales": funciones_especiales,
            "serie_taylor_general": sumatoria_general_tex,
            "explicacion_taylor_general": explicacion_taylor,
            "serie_taylor_finita": f_series_tex,
            "integral_definida_exacta": integral_definida_tex,
            "integral_definida_valor": f"$$ {valor_simbolico} $$",
            "valor_numerico_exacto": resultado_exacto_val,
            "metodos_numericos": {
                "simpson": integral_simpson,
                "romberg": integral_romberg,
                "cuadratura_gaussiana": integral_gauss,
                "montecarlo": integral_montecarlo,
            },
            "advertencias": advertencias,
            "geogebra_expresiones": geogebra_expresiones
        }

    except Exception as e:
        return {"error": str(e)}
