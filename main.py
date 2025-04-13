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

# Instancia de la app FastAPI
app = FastAPI()

# Declaramos las variables simbólicas
x, n = symbols('x n', real=True)

# Estructura de entrada
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
      - Elimina '*'
    """
    expr_str = sstr(expr)
    expr_str = expr_str.replace('**', '^').replace('*', '')
    return expr_str

def obtener_funciones_especiales(expr):
    """
    Detecta funciones especiales en la expresión Sympy 'expr' y
    devuelve definiciones con su LaTeX y descripción.
    """
    definiciones = []
    if expr is None:
        return definiciones

    expr_str = str(expr).lower()

    # -- Funciones "clásicas" --
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
            "latex": r"\mathrm{erfi}(x) = -i\,\mathrm{erf}(i\,x)",
            "descripcion": "Función error imaginaria, surge al integrar e^{x^2}."
        })
    if "li" in expr_str:
        definiciones.append({
            "funcion": "Li(x)",
            "latex": r"\mathrm{Li}(x) = \int_{0}^{x} \frac{dt}{\log(t)}",
            "descripcion": "Función logaritmo integral, primitiva de 1/log(x)."
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

    # -- Funciones especiales adicionales --
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
            "latex": r"Y_n(x) = \frac{J_n(x)\cos(n\pi) - J_{-n}(x)}{\sin(n\pi)}",
            "descripcion": "Función de Bessel de la segunda especie."
        })
    if "dawson" in expr_str:
        definiciones.append({
            "funcion": "F(x)",
            "latex": r"F(x) = e^{-x^2}\int_{0}^{x} e^{t^2}\,dt",
            "descripcion": "Función de Dawson, ligada a la función error."
        })
    # Fresnel S
    if "fresnels" in expr_str:
        definiciones.append({
            "funcion": "S(z)",
            "latex": r"S(z) = \int_{0}^{z} \sin(t^2)\,dt",
            "descripcion": "Función Fresnel S, primitiva de sin(x^2)."
        })
    # Fresnel C
    if "fresnelc" in expr_str:
        definiciones.append({
            "funcion": "C(z)",
            "latex": r"C(z) = \int_{0}^{z} \cos(t^2)\,dt",
            "descripcion": "Función Fresnel C, primitiva de cos(x^2)."
        })
    if "airy" in expr_str or "ai(" in expr_str:
        definiciones.append({
            "funcion": "Ai(x)",
            "latex": r"\mathrm{Ai}(x) = \frac{1}{\pi}\int_{0}^{\infty} \cos\!\Bigl(\tfrac{t^3}{3} + xt\Bigr)\,dt",
            "descripcion": "Función Airy Ai, solución de y'' - xy=0."
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

####################################################
#  --- MANEJO DE SUBINTERVALOS PARA SINGULARIDADES ---
####################################################

def crear_subintervalos(a_eval, b_eval, singularidades, epsilon=1e-8):
    """
    Genera subintervalos en [a_eval, b_eval] evitando epsilon alrededor
    de cada punto singular.
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
    Integra con Simpson cada subintervalo y suma.
    """
    total = 0.0
    for (a_i, b_i) in subintervalos:
        pts = np.linspace(a_i, b_i, n_points)
        vals = f_lambda(pts)
        total += simpson(vals, x=pts)
    return total

def monte_carlo_subintervalos(f_lambda, subintervalos, n_samples=10000):
    """
    Integra con Monte Carlo cada subintervalo y acumula el resultado.
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

#############################################
# --- FIN MANEJO DE SUBINTERVALOS ---
#############################################

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    """
    Endpoint que:
      - Genera la primitiva simbólica (si Sympy la reconoce).
      - Calcula la integral definida simbólica y su valor numérico.
      - Muestra las funciones especiales detectadas (inmediatamente después de la primitiva).
      - Desarrolla la serie de Taylor (infinita y truncada) con factoriales.
      - Ofrece la integración por Simpson, Romberg, Gauss y MonteCarlo.
      - Devuelve el texto para GeoGebra (función, polinomio de Taylor y comando de área).
    """
    try:
        # 1) Leer y convertir límites (símbolo + numérico)
        a_sym = sympify(datos.a)  
        b_sym = sympify(datos.b)
        a_eval = float(N(a_sym))
        b_eval = float(N(b_sym))

        if a_eval >= b_eval:
            raise HTTPException(status_code=400, detail="El límite inferior debe ser menor que el superior.")

        # 2) Parsear la función
        f_str = datos.funcion.strip()
        f = sympify(f_str)

        # Caso especial: sin(x)/x
        if str(f) == "sin(x)/x":
            def f_lambda(z):
                z_arr = np.array(z, ndmin=1)
                vals = np.where(np.isclose(z_arr, 0.0, atol=1e-14),
                                1.0,
                                np.sin(z_arr)/z_arr)
                return vals if len(vals) > 1 else vals[0]
        else:
            f_lambda = sp.lambdify(x, f, modules=['numpy'])

        # 3) Detectar singularidades
        advertencias = []
        singular_points = set()

        # 3a) Interior con solveset
        try:
            sing_interior = solveset(1/f, x, domain=Interval(a_eval, b_eval))
            for p in sing_interior:
                val_sing = float(p.evalf())
                if a_eval <= val_sing <= b_eval:
                    singular_points.add(val_sing)
                    advertencias.append(f"⚠️ Posible singularidad en x={val_sing}")
        except:
            pass

        # 3b) Extremos
        def es_finito(valor):
            try:
                return np.isfinite(valor)
            except:
                return False

        # Límite a
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

        # Límite b
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

        # 4) Hallar la primitiva simbólica
        #    Sympy a menudo expresa sin(x^2), cos(x^2) en términos de Fresnel.
        try:
            F_expr = integrate(f, x)  # Indefinida
            F_exacta_tex = latex(F_expr)
            valor_simbolico = latex(F_expr.subs(x, b_sym) - F_expr.subs(x, a_sym))
        except:
            F_expr = None
            F_exacta_tex = "No tiene primitiva elemental"
            valor_simbolico = "Valor simbólico no disponible"

        # 5) Funciones especiales detectadas: f y F_expr
        funcs_en_f = obtener_funciones_especiales(f)
        funcs_en_F = obtener_funciones_especiales(F_expr)
        all_funcs = {}
        for d in (funcs_en_f + funcs_en_F):
            all_funcs[d["funcion"]] = d
        funciones_especiales_detectadas = list(all_funcs.values())

        # 6) Integral definida simbólica y valor numérico
        try:
            resultado_exacto = integrate(f, (x, a_sym, b_sym))  # Definida
            resultado_exacto_val = float(N(resultado_exacto))
            resultado_exacto_tex = latex(resultado_exacto)
        except:
            resultado_exacto_val = None
            resultado_exacto_tex = "No calculable simbólicamente"

        # 7) Serie de Taylor alrededor de 0
        #    Distinguimos si la función es sin(x^2) o cos(x^2) => series manual
        #    Para otras funciones => derivada genérica (podría salir muchos ceros).
        if f_str == "sin(x^2)":
            # Serie infinita: sin(x^2) = \sum_{k=0}^\infty (-1)^k x^{4k+2} / (2k+1)!
            serie_infinita = r"\sin(x^2) = \sum_{k=0}^{\infty} \frac{(-1)^k x^{4k+2}}{(2k+1)!}"
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = (
                "La serie de Taylor de \\( \\sin(x^2) \\) alrededor de \\( x=0 \\) es:"
            )

            # Serie truncada
            terminos = []
            for k in range(datos.n_terminos):
                coef = (-1)**k
                potencia = 4*k + 2
                # Construimos la expresión en Sympy para integrarlo luego
                term_expr = coef * x**potencia / factorial(2*k + 1)
                terminos.append(term_expr)

            f_series_expr = sum(terminos)
            # Lo convertimos a un string LaTeX
            serie_str = []
            for k, t_expr in enumerate(terminos):
                # t_expr es algo como (-1)^k * x^(4k+2)/(2k+1)!
                serie_str.append(latex(t_expr))

            f_series_sumada = " + ".join(serie_str) + r" + \cdots"
            f_series_tex = f"$$ \\sin(x^2) = {f_series_sumada} $$"

        elif f_str == "cos(x^2)":
            # Serie infinita: cos(x^2) = \sum_{k=0}^\infty (-1)^k x^{4k} / (2k)!
            serie_infinita = r"\cos(x^2) = \sum_{k=0}^{\infty} \frac{(-1)^k x^{4k}}{(2k)!}"
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = (
                "La serie de Taylor de \\( \\cos(x^2) \\) alrededor de \\( x=0 \\) es:"
            )

            # Serie truncada
            terminos = []
            for k in range(datos.n_terminos):
                coef = (-1)**k
                potencia = 4*k
                term_expr = coef * x**potencia / factorial(2*k)
                terminos.append(term_expr)

            f_series_expr = sum(terminos)
            serie_str = []
            for k, t_expr in enumerate(terminos):
                serie_str.append(latex(t_expr))

            f_series_sumada = " + ".join(serie_str) + r" + \cdots"
            f_series_tex = f"$$ \\cos(x^2) = {f_series_sumada} $$"

        else:
            # Caso general: derivadas sucesivas en x=0
            a_taylor = 0
            serie_general = Sum(diff(f, (x, n)).subs(x, a_taylor)/factorial(n)*(x - a_taylor)**n, (n, 0, oo))
            sumatoria_general_tex = f"$$ {latex(serie_general)} $$"
            explicacion_taylor = (
                f"La serie de Taylor de \\( {latex(f)} \\) alrededor de \\( x=0 \\) es:"
            )

            # Serie truncada
            terminos = []
            for i in range(datos.n_terminos):
                deriv_i = diff(f, (x, i)).subs(x, a_taylor)
                termino_i = simplify(deriv_i/factorial(i)) * (x - a_taylor)**i
                terminos.append(termino_i)

            f_series_expr = sum(terminos)
            # Armamos el string LaTeX
            serie_str = []
            for t_expr in terminos:
                serie_str.append(latex(t_expr))
            f_series_sumada = " + ".join(serie_str) + r" + \cdots"
            f_series_tex = f"$$ {latex(f)} = {f_series_sumada} $$"

        # Integramos la serie truncada (simbólicamente) para mostrar su primitiva
        try:
            F_aproximada = integrate(f_series_expr, x)
            F_aproximada_tex = f"$$ {latex(F_aproximada)} $$"
        except:
            F_aproximada_tex = "No se pudo calcular la integral de la serie truncada."

        # 8) Notación LaTeX de la integral definida con símbolos originales
        integral_definida_tex = f"$$ \\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\; dx $$"

        # 9) Métodos numéricos
        epsilon = 1e-8
        subintervalos = crear_subintervalos(a_eval, b_eval, singular_points, epsilon)

        # Simpson
        integral_simpson = simpson_subintervalos(f_lambda, subintervalos)

        # Romberg / Gauss con 'quad' y 'points'
        points_sing = sorted(list(singular_points))
        try:
            val_romberg, _ = quad(f_lambda, a_eval, b_eval, points=points_sing)
            integral_romberg = val_romberg
        except:
            integral_romberg = None
            advertencias.append("No se pudo calcular con Romberg (posibles singularidades).")

        try:
            val_gauss, _ = quad(f_lambda, a_eval, b_eval, points=points_sing)
            integral_gauss = val_gauss
        except:
            integral_gauss = None
            advertencias.append("No se pudo calcular con Cuadratura Gaussiana (posibles singularidades).")

        # Monte Carlo
        integral_montecarlo = monte_carlo_subintervalos(f_lambda, subintervalos)

        # 10) Texto para GeoGebra (función, polinomio de Taylor, área)
        geogebra_expresiones = {
            "funcion": f"f(x) = {exportar_para_geogebra(f)}",
            "taylor": f"T(x) = {exportar_para_geogebra(f_series_expr)}",
            "area_comando": f"Integral(f, {str(a_sym)}, {str(b_sym)})"
        }

        # 11) Construcción final de la respuesta
        return {
            "funcion_introducida": str(f),
            "integral_definida": integral_definida_tex,

            # PRIMITIVA
            "primitiva_real": f"$$ {F_exacta_tex} $$",

            # ESPECIALES (justo después de la primitiva)
            "funciones_especiales_detectadas": funciones_especiales_detectadas,

            # Valor simbólico exacto F(b) - F(a)
            "valor_simbolico_integral": f"$$ {valor_simbolico} $$",

            # Valor numérico exacto
            "valor_numerico_exacto": resultado_exacto_val,
            "valor_numerico_exacto_latex": f"$$ {resultado_exacto_tex} $$",

            # Series
            "serie_taylor_general": sumatoria_general_tex,
            "explicacion_taylor_general": explicacion_taylor,
            "serie_taylor_finita": f_series_tex,
            "integral_serie_taylor": F_aproximada_tex,

            # GeoGebra
            "geogebra_expresiones": geogebra_expresiones

            # Métodos numéricos
            "metodos_numericos": {
                "simpson": integral_simpson,
                "romberg": integral_romberg,
                "cuadratura_gaussiana": integral_gauss,
                "montecarlo": integral_montecarlo
            },

            "advertencias": advertencias,


        }

    except Exception as e:
        return {"error": str(e)}
