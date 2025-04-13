from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union
import numpy as np
from sympy import (
    symbols, sympify, integrate, solveset, Interval, oo,
    diff, factorial, Sum, latex, simplify, N, sstr, Function,
    erf, sqrt, pi, erfi, sin, cos
)
from scipy.integrate import simpson, quad
import sympy as sp
import random

app = FastAPI()

# Declaramos las variables simbólicas
x, n = symbols('x n', real=True)

# Clase para la entrada de datos
class InputDatos(BaseModel):
    funcion: str
    a: Union[str, float]
    b: Union[str, float]
    n_terminos: int = Field(default=10, ge=1, le=20)
    tolerancia: float = Field(default=1e-6, ge=1e-10)

def exportar_para_geogebra(expr):
    """
    Convierte una expresión Sympy a un string apto para GeoGebra:
      - Reemplaza '**' por '^'
      - Elimina '*' (para multiplicación implícita)
    """
    expr_str = sstr(expr)
    expr_str = expr_str.replace('**', '^').replace('*', '')
    return expr_str

def crear_subintervalos(a_eval, b_eval, singularidades, epsilon=1e-8):
    """
    Genera subintervalos en [a_eval, b_eval] evitando una vecindad epsilon
    alrededor de cada punto singular.
    """
    sing_sorted = sorted(singularidades)
    a_adj = a_eval
    b_adj = b_eval
    if sing_sorted and abs(sing_sorted[0] - a_eval) < epsilon:
        a_adj = a_eval + epsilon
    if sing_sorted and abs(b_adj - sing_sorted[-1]) < epsilon:
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

def simpson_subintervalos(f_lambda, subintervalos, n_points=1001):
    """
    Aplica la regla de Simpson a cada subintervalo y suma el resultado.
    """
    total = 0.0
    for (a_i, b_i) in subintervalos:
        pts = np.linspace(a_i, b_i, n_points)
        vals = f_lambda(pts)
        total += simpson(vals, x=pts)
    return total

def monte_carlo_subintervalos(f_lambda, subintervalos, n_samples=10000):
    """
    Integra usando el método de Monte Carlo en cada subintervalo y acumula.
    """
    total = 0.0
    for (a_i, b_i) in subintervalos:
        acum = 0.0
        for _ in range(n_samples):
            x_rand = random.uniform(a_i, b_i)
            val = f_lambda(x_rand)
            if isinstance(val, (list, np.ndarray)):
                val = val[0]
            acum += val
        largo = (b_i - a_i)
        total += largo * (acum / n_samples)
    return total

def obtener_funciones_especiales(expr):
    """
    Detecta funciones especiales en la expresión Sympy 'expr' y
    retorna una lista de diccionarios con:
      - "funcion": notación (ej. Si(x)).
      - "latex": definición matemática en LaTeX.
      - "descripcion": breve explicación.
    Si no se encuentra ninguna, devuelve una lista vacía.
    """
    definiciones = []
    if expr is None:
        return definiciones
    expr_str = str(expr).lower()
    if "si" in expr_str:
        definiciones.append({
            "funcion": "Si(x)",
            "latex": r"\mathrm{Si}(x) = \int_{0}^{x} \frac{\sin(t)}{t}\,dt",
            "descripcion": "Función seno integral (primitiva de sin(x)/x)."
        })
    if "erf" in expr_str and "erfi" not in expr_str:
        definiciones.append({
            "funcion": "erf(x)",
            "latex": r"\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2}\,dt",
            "descripcion": "Función error, típica al integrar e^{-x^2}."
        })
    if "erfi" in expr_str:
        definiciones.append({
            "funcion": "erfi(x)",
            "latex": r"\mathrm{erfi}(x) = -\,i\,\mathrm{erf}(i\,x)",
            "descripcion": "Función error imaginaria, surge al integrar e^{x^2}."
        })
    if "li" in expr_str:
        definiciones.append({
            "funcion": "Li(x)",
            "latex": r"\mathrm{Li}(x) = \int_{0}^{x} \frac{dt}{\log(t)}",
            "descripcion": "Función logaritmo integral, primitiva de 1/\log(x)."
        })
    if "ci" in expr_str:
        definiciones.append({
            "funcion": "Ci(x)",
            "latex": r"\mathrm{Ci}(x) = -\int_{x}^{\infty} \frac{\cos(t)}{t}\,dt",
            "descripcion": "Función coseno integral, aparece en análisis armónico."
        })
    if "gamma" in expr_str:
        definiciones.append({
            "funcion": "Gamma(x)",
            "latex": r"\Gamma(x) = \int_{0}^{\infty} t^{x-1} e^{-t}\,dt",
            "descripcion": "Extiende el factorial a números reales y complejos."
        })
    if "beta" in expr_str:
        definiciones.append({
            "funcion": "Beta(x, y)",
            "latex": r"B(x, y) = \int_{0}^{1} t^{x-1}(1-t)^{y-1}\,dt",
            "descripcion": "Relación con Gamma; surge en probabilidad."
        })
    if "ei" in expr_str:
        definiciones.append({
            "funcion": "Ei(x)",
            "latex": r"\mathrm{Ei}(x) = \int_{-\infty}^{x} \frac{e^t}{t}\,dt",
            "descripcion": "Función exponencial integral, integra e^x/x."
        })
    if "besselj" in expr_str or "j(" in expr_str:
        definiciones.append({
            "funcion": "J_n(x)",
            "latex": r"J_n(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m!\,\Gamma(m+n+1)}\left(\frac{x}{2}\right)^{2m+n}",
            "descripcion": "Función de Bessel de la primera especie."
        })
    if "bessely" in expr_str or "y(" in expr_str:
        definiciones.append({
            "funcion": "Y_n(x)",
            "latex": r"Y_n(x) = \frac{J_n(x)\cos(n\pi)-J_{-n}(x)}{\sin(n\pi)}",
            "descripcion": "Función de Bessel de la segunda especie."
        })
    if "dawson" in expr_str:
        definiciones.append({
            "funcion": "F(x)",
            "latex": r"F(x) = e^{-x^2}\int_{0}^{x} e^{t^2}\,dt",
            "descripcion": "Función de Dawson, ligada a la función error."
        })
    if "fresnels" in expr_str:
        definiciones.append({
            "funcion": "S(z)",
            "latex": r"S(z) = \int_{0}^{z} \sin(t^2)\,dt",
            "descripcion": "Función Fresnel S, primitiva de sin(x^2)."
        })
    if "fresnelc" in expr_str:
        definiciones.append({
            "funcion": "C(z)",
            "latex": r"C(z) = \int_{0}^{z} \cos(t^2)\,dt",
            "descripcion": "Función Fresnel C, primitiva de cos(x^2)."
        })
    if "airy" in expr_str or "ai(" in expr_str:
        definiciones.append({
            "funcion": "Ai(x)",
            "latex": r"\mathrm{Ai}(x) = \frac{1}{\pi}\int_{0}^{\infty}\cos\!\Bigl(\frac{t^3}{3}+xt\Bigr)\,dt",
            "descripcion": "Función Airy Ai, solución de y'' - x y = 0."
        })
    if "bi(" in expr_str:
        definiciones.append({
            "funcion": "Bi(x)",
            "latex": r"\mathrm{Bi}(x) = \dots",
            "descripcion": "Función Airy Bi, solución secundaria de la ecuación de Airy."
        })
    if "zeta" in expr_str:
        definiciones.append({
            "funcion": "ζ(s)",
            "latex": r"\zeta(s) = \sum_{n=1}^\infty \frac{1}{n^s}",
            "descripcion": "Función zeta de Riemann, clave en teoría de números."
        })
    return definiciones

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        # 1) Convertir los límites (símbolos y valores numéricos)
        a_sym = sympify(datos.a)
        b_sym = sympify(datos.b)
        a_eval = float(N(a_sym))
        b_eval = float(N(b_sym))
        if a_eval >= b_eval:
            raise HTTPException(status_code=400, detail="El límite inferior debe ser menor que el superior.")
        
        # 2) Parsear la función
        f_str = datos.funcion.strip()
        f = sympify(f_str)
        if str(f) == "sin(x)/x":
            def f_lambda(z):
                z_arr = np.array(z, ndmin=1)
                vals = np.where(np.isclose(z_arr, 0.0, atol=1e-14), 1.0, np.sin(z_arr)/z_arr)
                return vals if len(vals) > 1 else vals[0]
        else:
            f_lambda = sp.lambdify(x, f, modules=['numpy'])
        
        # 3) Detectar singularidades
        advertencias = []
        singular_points = set()
        try:
            sing_interior = solveset(1/f, x, domain=Interval(a_eval, b_eval))
            for p in sing_interior:
                val_sing = float(p.evalf())
                if a_eval <= val_sing <= b_eval:
                    singular_points.add(val_sing)
                    advertencias.append(f"⚠️ Posible singularidad en x={val_sing}")
        except:
            pass
        def es_finito(valor):
            try:
                return np.isfinite(valor)
            except:
                return False
        try:
            val_a = f_lambda(a_eval)
            if isinstance(val_a, (list, np.ndarray)):
                val_a = val_a[0]
            if not es_finito(val_a):
                singular_points.add(a_eval)
                advertencias.append(f"⚠️ Singularidad en x={a_eval} (límite inferior)")
        except:
            singular_points.add(a_eval)
            advertencias.append(f"⚠️ Singularidad en x={a_eval} (límite inferior)")
        try:
            val_b = f_lambda(b_eval)
            if isinstance(val_b, (list, np.ndarray)):
                val_b = val_b[0]
            if not es_finito(val_b):
                singular_points.add(b_eval)
                advertencias.append(f"⚠️ Singularidad en x={b_eval} (límite superior)")
        except:
            singular_points.add(b_eval)
            advertencias.append(f"⚠️ Singularidad en x={b_eval} (límite superior)")
        
        # 4) Calcular la primitiva simbólica
        try:
            F_expr = integrate(f, x)
            F_exacta_tex = latex(F_expr)
            valor_simbolico = latex(F_expr.subs(x, b_sym) - F_expr.subs(x, a_sym))
        except:
            F_expr = None
            F_exacta_tex = "No tiene primitiva elemental"
            valor_simbolico = "Valor simbólico no disponible"
        
        # 5) Detectar funciones especiales en f y en la primitiva
        funcs_en_f = obtener_funciones_especiales(f)
        funcs_en_F = obtener_funciones_especiales(F_expr)
        all_funcs = {}
        for d in (funcs_en_f + funcs_en_F):
            all_funcs[d["funcion"]] = d
        funciones_especiales_detectadas = list(all_funcs.values())
        
        # 6) Calcular la integral definida simbólica y su valor numérico
        try:
            resultado_exacto = integrate(f, (x, a_sym, b_sym))
            resultado_exacto_val = float(N(resultado_exacto))
            resultado_exacto_tex = latex(resultado_exacto)
        except:
            resultado_exacto_val = None
            resultado_exacto_tex = "No calculable simbólicamente"
        
        # 7) Calcular la serie de Taylor
        if f_str == "sin(x^2)":
            serie_infinita = r"\sin(x^2) = \sum_{k=0}^{\infty} \frac{(-1)^k x^{4k+2}}{(2k+1)!}"
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = "La serie de Taylor de \\(\\sin(x^2)\\) alrededor de \\(x=0\\) es:"
            terminos = []
            for k in range(datos.n_terminos):
                coef = (-1)**k
                potencia = 4*k + 2
                term_expr = coef * x**potencia / factorial(2*k + 1)
                terminos.append(term_expr)
            f_series_expr = sum(terminos)
            serie_str = [latex(t_expr) for t_expr in terminos]
            f_series_sumada = " + ".join(serie_str) + r" + \cdots"
            f_series_tex = f"$$ \\sin(x^2) = {f_series_sumada} $$"
        elif f_str == "cos(x^2)":
            serie_infinita = r"\cos(x^2) = \sum_{k=0}^{\infty} \frac{(-1)^k x^{4k}}{(2k)!}"
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = "La serie de Taylor de \\(\\cos(x^2)\\) alrededor de \\(x=0\\) es:"
            terminos = []
            for k in range(datos.n_terminos):
                coef = (-1)**k
                potencia = 4*k
                term_expr = coef * x**potencia / factorial(2*k)
                terminos.append(term_expr)
            f_series_expr = sum(terminos)
            serie_str = [latex(t_expr) for t_expr in terminos]
            f_series_sumada = " + ".join(serie_str) + r" + \cdots"
            f_series_tex = f"$$ \\cos(x^2) = {f_series_sumada} $$"
        else:
            a_taylor = 0
            serie_general_expr = Sum(diff(f, (x, n)).subs(x, a_taylor)/factorial(n)*(x - a_taylor)**n, (n, 0, oo))
            sumatoria_general_tex = f"$$ {latex(serie_general_expr)} $$"
            explicacion_taylor = f"La serie de Taylor de \\( {latex(f)} \\) alrededor de \\( x=0 \\) es:"
            terminos = []
            for i in range(datos.n_terminos):
                deriv_i = diff(f, (x, i)).subs(x, a_taylor)
                termino_i = simplify(deriv_i/factorial(i))*(x - a_taylor)**i
                terminos.append(termino_i)
            f_series_expr = sum(terminos)
            serie_str = [latex(t_expr) for t_expr in terminos]
            f_series_sumada = " + ".join(serie_str) + r" + \cdots"
            f_series_tex = f"$$ {latex(f)} = {f_series_sumada} $$"
        try:
            F_aproximada = integrate(f_series_expr, x)
            F_aproximada_tex = f"$$ {latex(F_aproximada)} $$"
        except:
            F_aproximada_tex = "No se pudo calcular la integral de la serie truncada."
        
        integral_definida_tex = f"$$ \\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\; dx $$"
        
        # 8) Métodos numéricos
        epsilon = 1e-8
        subintervalos = crear_subintervalos(a_eval, b_eval, singular_points, epsilon)
        integral_simpson = simpson_subintervalos(f_lambda, subintervalos)
        points_sing = sorted(list(singular_points))
        try:
            val_romberg, _ = quad(f_lambda, a_eval, b_eval, points=points_sing)
            integral_romberg = val_romberg
        except:
            integral_romberg = None
            advertencias.append("No se pudo calcular con Romberg (singularidades).")
        try:
            val_gauss, _ = quad(f_lambda, a_eval, b_eval, points=points_sing)
            integral_gauss = val_gauss
        except:
            integral_gauss = None
            advertencias.append("No se pudo calcular con Cuadratura Gaussiana (singularidades).")
        integral_montecarlo = monte_carlo_subintervalos(f_lambda, subintervalos)
        
        # 9) Texto para GeoGebra
        geogebra_expresiones = {
            "funcion": f"f(x) = {exportar_para_geogebra(f)}",
            "taylor": f"T(x) = {exportar_para_geogebra(f_series_expr)}",
            "area_comando": f"Integral(f, {str(a_sym)}, {str(b_sym)})"
        }
        
        # 10) Devolver el JSON (notar las comas entre claves)
        return {
            "funcion_introducida": str(f),
            "integral_definida": integral_definida_tex,
            "primitiva_real": f"$$ {F_exacta_tex} $$",
            "funciones_especiales_detectadas": funciones_especiales_detectadas,
            "valor_simbolico_integral": f"$$ {valor_simbolico} $$",
            "valor_numerico_exacto": resultado_exacto_val,
            "valor_numerico_exacto_latex": f"$$ {resultado_exacto_tex} $$",
            "serie_taylor_general": sumatoria_general_tex,
            "explicacion_taylor_general": explicacion_taylor,
            "serie_taylor_finita": f_series_tex,
            "integral_serie_taylor": F_aproximada_tex,
            "metodos_numericos": {
                "simpson": integral_simpson,
                "romberg": integral_romberg,
                "cuadratura_gaussiana": integral_gauss,
                "montecarlo": integral_montecarlo
            },
            "advertencias": advertencias,
            "geogebra_expresiones": geogebra_expresiones
        }
    except Exception as e:
        return {"error": str(e)}
