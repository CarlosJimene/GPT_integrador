from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union
import numpy as np
from sympy import (
    symbols, sympify, integrate, solveset, Interval, oo, diff,
    factorial, Sum, latex, simplify, N, sstr, Function, erf, sqrt, pi,
    fresnels, fresnelc, sin, cos, Si, Ci, gamma, lowergamma, uppergamma,
    Rational, binomial
)
from scipy.integrate import simpson, quad
import sympy as sp
import random

app = FastAPI()

x, n = symbols('x n', real=True)

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

def crear_subintervalos(a_eval, b_eval, singularidades, epsilon=1e-8):
    sing_sorted = sorted(singularidades)
    a_adj = a_eval
    b_adj = b_eval
    if sing_sorted and abs(sing_sorted[0] - a_eval) < epsilon:
        a_adj = a_eval + epsilon
    if sing_sorted and abs(b_eval - sing_sorted[-1]) < epsilon:
        b_adj = b_eval - epsilon

    puntos = [a_adj]
    for s in sing_sorted:
        s_left = s - epsilon
        s_right = s + epsilon
        if s_left > puntos[-1]:
            puntos.append(s_left)
        if s_right < b_adj:
            puntos.append(s_right)
    puntos.append(b_adj)
    subintervalos = []
    for i in range(len(puntos) - 1):
        izq, der = puntos[i], puntos[i+1]
        if izq < der:
            subintervalos.append((izq, der))
    return subintervalos

def simpson_subintervalos(f_lambda, subintervalos, n_points_simpson=1001):
    total = 0.0
    total_points = 0
    for (a_i, b_i) in subintervalos:
        pts = np.linspace(a_i, b_i, n_points_simpson)
        vals = f_lambda(pts)
        total += simpson(vals, x=pts)
        total_points += len(pts)
    return total, total_points

def monte_carlo_subintervalos(f_lambda, subintervalos, n_samples=10000):
    total = 0.0
    for (a_i, b_i) in subintervalos:
        acum = 0.0
        for _ in range(n_samples):
            x_rand = random.uniform(a_i, b_i)
            val = f_lambda(x_rand)
            if isinstance(val, (list, np.ndarray)):
                val = val[0]
            acum += val
        total += (b_i - a_i) * (acum / n_samples)
    return total

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        # 0) Reemplazar ^ por ** para compatibilidad con sympy
        f_str = datos.funcion.strip()
        f_str_mod = f_str.replace("^", "**")

        # 1) LÍMITES
        a_sym = sympify(datos.a)
        b_sym = sympify(datos.b)
        a_eval = float(N(a_sym))
        b_eval = float(N(b_sym))
        if a_eval >= b_eval:
            raise HTTPException(status_code=400, detail="Límite inferior debe ser menor que el superior.")

        # 2) FUNCIÓN
        f = sympify(f_str_mod)
        f_lambda = sp.lambdify(x, f, modules=['numpy'])

        # 3) Singularidades
        advertencias = []
        singular_points = set()
        def es_finito(val):
            try:
                return np.isfinite(val)
            except:
                return False
        try:
            interior_sings = solveset(1/f, x, domain=Interval(a_eval, b_eval))
            for p in interior_sings:
                val_sing = float(p.evalf())
                if a_eval < val_sing < b_eval:
                    singular_points.add(val_sing)
                    advertencias.append(f"⚠️ Posible singularidad en x={val_sing}")
        except:
            pass
        for val, name in [(a_eval, a_sym), (b_eval, b_sym)]:
            try:
                v = f_lambda(val)
                if isinstance(v, (list, np.ndarray)):
                    v = v[0]
                if not es_finito(v):
                    singular_points.add(val)
                    advertencias.append(f"⚠️ Singularidad en x={val}")
            except:
                singular_points.add(val)
                advertencias.append(f"⚠️ Singularidad en x={val}")

        # 4) PRIMITIVA
        F_expr = None
        if f_str_mod == "exp(-x**2)":
            F_expr = (sqrt(pi)/2)*erf(x)
        elif f_str_mod == "sin(x**2)":
            F_expr = (sqrt(pi)/2)*fresnels(x/sqrt(pi))
        elif f_str_mod == "cos(x**2)":
            F_expr = (sqrt(pi)/2)*fresnelc(x/sqrt(pi))
        elif f_str_mod == "sin(x)/x":
            F_expr = Si(x)
        elif f_str_mod == "sqrt(1 - x**4)":
            try:
                F_expr = sp.integrate(f, x, meijerg=True)
            except Exception as e:
                advertencias.append(f"Error al integrar sqrt(1 - x^4): {e}")
        else:
            try:
                F_expr = sp.simplify(integrate(f, x))
            except Exception as e:
                advertencias.append(f"Error al integrar simbólicamente: {e}")
        
        if F_expr is None:
            F_exacta_tex = "No tiene primitiva elemental conocida"
            valor_simbolico = "Valor simbólico no disponible"
        else:
            F_exacta_tex = latex(F_expr)
            try:
                valor_sym = F_expr.subs(x, b_sym) - F_expr.subs(x, a_sym)
                valor_simbolico = latex(simplify(valor_sym))
            except:
                valor_simbolico = "No se pudo evaluar la primitiva"

        # 5) SERIE DE TAYLOR
        explicacion_taylor = ""
        sumatoria_general_tex = ""
        f_series_tex = ""
        terminos = []

        if f_str_mod == "exp(-x**2)":
            explicacion_taylor = "La serie de Taylor de \\(e^{-x^2}\\) alrededor de x=0 es:"
            sumatoria_general_tex = r"$$ e^{-x^2} = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{n!} $$"
            for n_ in range(datos.n_terminos):
                term_expr = (-1)**n_ * x**(2*n_) / factorial(n_)
                terminos.append(term_expr)
        else:
            a_taylor = 0
            serie_general_expr = Sum(diff(f, (x, n)).subs(x, a_taylor)/factorial(n)*(x-a_taylor)**n, (n, 0, oo))
            sumatoria_general_tex = f"$$ {latex(serie_general_expr)} $$"
            explicacion_taylor = f"La serie de Taylor de \\({latex(f)}\\) alrededor de x=0 es:"
            for i in range(datos.n_terminos):
                deriv_i = diff(f, (x, i)).subs(x, a_taylor)
                term_expr = simplify(deriv_i/factorial(i))*(x-a_taylor)**i
                terminos.append(term_expr)

        serie_latex_terms = [latex(t) for t in terminos]
        f_series_sumada = " + ".join(serie_latex_terms) + r" + \cdots"
        f_series_expr = sum(terminos)
        f_series_tex = f"$$ {latex(f)} = {f_series_sumada} $$"
        try:
            F_aproximada = integrate(f_series_expr, x)
            F_aproximada_tex = f"$$ {latex(F_aproximada)} $$"
        except:
            F_aproximada_tex = "No se pudo calcular la integral de la serie truncada"

        # 6) Integral definida
        try:
            resultado_sympy_def = integrate(f, (x, a_sym, b_sym))
            resultado_exacto_tex = latex(resultado_sympy_def)
            resultado_exacto_val = float(N(resultado_sympy_def))
        except:
            resultado_exacto_val = None
            resultado_exacto_tex = "No se pudo calcular simbólicamente"

        # 7) Métodos numéricos
        subintervalos = crear_subintervalos(a_eval, b_eval, singular_points)
        integral_simpson, n_points_simpson = simpson_subintervalos(f_lambda, subintervalos)
        pts_sing = sorted(list(singular_points))
        try:
            val_romberg, _ = quad(f_lambda, a_eval, b_eval, points=pts_sing)
            integral_romberg = val_romberg
        except:
            integral_romberg = None
        try:
            val_gauss, _ = quad(f_lambda, a_eval, b_eval, points=pts_sing)
            integral_gauss = val_gauss
        except:
            integral_gauss = None
        integral_montecarlo = monte_carlo_subintervalos(f_lambda, subintervalos)

        # 8) GeoGebra
        texto_geogebra = (
            f"Función: f(x) = {exportar_para_geogebra(f)}\n"
            f"Taylor truncada: T(x) = {exportar_para_geogebra(f_series_expr)}\n"
            f"Área: Integral(f, {str(a_sym)}, {str(b_sym)})"
        )

        return {
            "funcion_introducida": str(f),
            "primitiva_real": f"$$ {F_exacta_tex} $$",
            "valor_simbolico_integral": f"$$ {valor_simbolico} $$",
            "valor_numerico_exacto": resultado_exacto_val,
            "valor_numerico_exacto_latex": f"$$ {resultado_exacto_tex} $$",
            "integral_definida": f"$$ \\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\, dx $$",
            "serie_taylor_general": sumatoria_general_tex,
            "explicacion_taylor_general": explicacion_taylor,
            "serie_taylor_finita": f_series_tex,
            "integral_serie_taylor": F_aproximada_tex,
            "metodos_numericos": {
                "simpson": {"value": integral_simpson, "n_points": n_points_simpson},
                "romberg": integral_romberg,
                "cuadratura_gaussiana": integral_gauss,
                "montecarlo": integral_montecarlo
            },
            "advertencias": advertencias,
            "texto_geogebra": texto_geogebra
        }
    except Exception as e:
        return {"error": str(e)}
