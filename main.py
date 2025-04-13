from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union
import numpy as np
from sympy import (
    symbols, sympify, integrate, solveset, Interval, oo,
    diff, factorial, Sum, latex, simplify, N, sstr, Function,
    erf, sqrt, pi, erfi
)
from scipy.integrate import simpson, quad
import sympy as sp
import random

app = FastAPI()

# Variables simbólicas
x, n = symbols('x n', real=True)

class InputDatos(BaseModel):
    funcion: str
    a: Union[str, float]
    b: Union[str, float]
    n_terminos: int = Field(default=10, ge=1, le=20)
    tolerancia: float = Field(default=1e-6, ge=1e-10)

def exportar_para_geogebra(expr):
    """
    Convierte una expresión simbólica a un string apto para GeoGebra:
      - Reemplaza '**' por '^'
      - Elimina '*'
    De este modo, 'x**2' se vuelve 'x^2' y '2*x' se vuelve '2x'.
    """
    expr_str = sstr(expr)
    expr_str = expr_str.replace('**', '^').replace('*', '')
    return expr_str

def obtener_funciones_especiales(expr):
    """
    Dada una expresión Sympy 'expr', detecta funciones especiales
    (erf, fresnels, etc.) y devuelve definiciones con su LaTeX y descripción.
    """
    definiciones = []
    expr_str = str(expr).lower()  # Para búsquedas insensibles a mayúsc/minúsc

    # ---- Funciones "clásicas" ----
    if "si" in expr_str:
        definiciones.append({
            "funcion": "Si(x)",
            "latex": r"\mathrm{Si}(x) = \int_{0}^{x} \frac{\sin(t)}{t}\,dt",
            "descripcion": "Función seno integral (primitiva de sin(x)/x)."
        })
    # Notar que 'erf' también aparece en 'erfi', así que distinguimos
    if "erf" in expr_str and "erfi" not in expr_str:
        definiciones.append({
            "funcion": "erf(x)",
            "latex": r"\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2}\,dt",
            "descripcion": "La función error, típica al integrar e^{-x^2}."
        })
    if "erfi" in expr_str:
        definiciones.append({
            "funcion": "erfi(x)",
            "latex": r"\mathrm{erfi}(x) = -\,i\,\mathrm{erf}(i\,x)",
            "descripcion": "Función error imaginaria, relevante para e^{x^2}."
        })
    if "li" in expr_str:
        definiciones.append({
            "funcion": "Li(x)",
            "latex": r"\mathrm{Li}(x) = \int_{0}^{x} \frac{dt}{\log(t)}",
            "descripcion": "Logaritmo integral, primitiva de 1/log(x)."
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
            "latex": r"B(x, y) = \int_{0}^{1} t^{x-1} (1 - t)^{y-1}\,dt",
            "descripcion": "Relación con la Gamma; surge en probabilidades."
        })

    # ---- Funciones especiales adicionales ----
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
    # Fresnel S => sin(x^2)
    if "fresnels" in expr_str:
        definiciones.append({
            "funcion": "S(z)",
            "latex": r"S(z) = \int_{0}^{z} \sin(t^2)\,dt",
            "descripcion": "Función Fresnel S, primitiva de sin(x^2)."
        })
    # Fresnel C => cos(x^2)
    if "fresnelc" in expr_str:
        definiciones.append({
            "funcion": "C(z)",
            "latex": r"C(z) = \int_{0}^{z} \cos(t^2)\,dt",
            "descripcion": "Función Fresnel C, primitiva de cos(x^2)."
        })
    if "ai(" in expr_str or "airy" in expr_str:
        definiciones.append({
            "funcion": "Ai(x)",
            "latex": r"\mathrm{Ai}(x) = \frac{1}{\pi} \int_{0}^{\infty} \cos\!\Bigl(\tfrac{t^3}{3} + xt\Bigr)\,dt",
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

####################################################
#  --- MANEJO DE SUBINTERVALOS PARA SINGULARIDADES ---
####################################################

def crear_subintervalos(a_eval, b_eval, singularidades, epsilon=1e-8):
    """
    Toma [a_eval, b_eval] y un conjunto de puntos singulares,
    y genera subintervalos que evitan una vecindad epsilon
    alrededor de cada singularidad.
    """
    sing_sorted = sorted(singularidades)
    a_adj = a_eval
    b_adj = b_eval

    # Ajuste en caso de singularidades pegadas a los límites
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
    Integra con la regla de Simpson sobre cada subintervalo y suma el total.
    """
    total = 0.0
    for (a_i, b_i) in subintervalos:
        pts = np.linspace(a_i, b_i, n_points)
        vals = f_lambda(pts)
        total += simpson(vals, x=pts)
    return total

def monte_carlo_subintervalos(f_lambda, subintervalos, n_samples=10000):
    """
    Integra usando Monte Carlo en cada subintervalo y acumula los resultados.
    """
    total = 0.0
    for (a_i, b_i) in subintervalos:
        acumulado = 0.0
        for _ in range(n_samples):
            x_rand = random.uniform(a_i, b_i)
            val = f_lambda(x_rand)
            if isinstance(val, (list, np.ndarray)):
                val = val[0]
            acumulado += val
        largo = (b_i - a_i)
        total += largo * (acumulado / n_samples)
    return total

#############################################
# --- FIN MANEJO DE SUBINTERVALOS ---
#############################################

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    """
    Procesa la función dada en 'datos.funcion' en el rango [a, b].
    Devuelve:
      - Primitiva simbólica en LaTeX (cuando Sympy la reconoce).
      - Integral definida exacta (simbólica).
      - Valor numérico exacto (Sympy).
      - Métodos numéricos (Simpson, Romberg, Gauss, Monte Carlo).
      - Serie de Taylor (infinita en LaTeX y truncada).
      - Definiciones de funciones especiales detectadas.
      - Expresiones de GeoGebra para copiar y pegar.
    """
    try:
        # 1) Parseo de límites simbólicos
        a_sym = sympify(datos.a)  # p.ej. -sqrt(pi)
        b_sym = sympify(datos.b)  # p.ej. sqrt(pi)
        a_eval = float(N(a_sym))  # Para cálculos numéricos
        b_eval = float(N(b_sym))

        if a_eval >= b_eval:
            raise HTTPException(
                status_code=400,
                detail="El límite inferior debe ser menor que el superior."
            )

        # 2) Parseo de la función integranda
        f = sympify(datos.funcion)

        # Caso particular sin(x)/x
        if str(f) == "sin(x)/x":
            def f_lambda(z):
                z_arr = np.array(z, ndmin=1)
                vals = np.where(np.isclose(z_arr, 0.0, atol=1e-14),
                                1.0,
                                np.sin(z_arr)/z_arr)
                return vals if len(vals) > 1 else vals[0]
        else:
            f_lambda = sp.lambdify(x, f, modules=['numpy'])

        # 3) Detectar singularidades (interiores y extremos)
        advertencias = []
        singular_points = set()

        # 3a) Sing interior
        try:
            sing_interior = solveset(1/f, x, domain=Interval(a_eval, b_eval))
            for p in sing_interior:
                val = float(p.evalf())
                if a_eval <= val <= b_eval:
                    singular_points.add(val)
                    advertencias.append(f"⚠️ Posible singularidad en x={val}")
        except:
            pass

        # 3b) Sing en extremos
        def es_finito(valor):
            try:
                return np.isfinite(valor)
            except:
                return False

        # Extremo a
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

        # Extremo b
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

        # 4) Primitiva simbólica (intenta con Sympy)
        try:
            F_expr = integrate(f, x)  # Indefinida
            F_exacta_tex = latex(F_expr)
            valor_simbolico = latex(F_expr.subs(x, b_sym) - F_expr.subs(x, a_sym))
        except:
            F_expr = None
            F_exacta_tex = "No tiene primitiva elemental"
            valor_simbolico = "Valor simbólico no disponible"

        # 5) Funciones especiales detectadas: en f y en la primitiva (si existe)
        funciones_en_f = obtener_funciones_especiales(f)
        funciones_en_F = obtener_funciones_especiales(F_expr) if F_expr else []
        all_funcs = {}
        for d in (funciones_en_f + funciones_en_F):
            all_funcs[d["funcion"]] = d
        funciones_especiales_detectadas = list(all_funcs.values())

        # 6) Integral definida (Sympy)
        try:
            resultado_exacto = integrate(f, (x, a_sym, b_sym))
            resultado_exacto_val = float(N(resultado_exacto))
            resultado_exacto_tex = latex(resultado_exacto)
        except:
            resultado_exacto_val = None
            resultado_exacto_tex = "No calculable simbólicamente"

        # 7) Serie de Taylor alrededor de 0
        a_taylor = 0
        # Serie simbólica "infinita"
        serie_general = Sum(
            diff(f, (x, n)).subs(x, a_taylor)/factorial(n)*(x - a_taylor)**n,
            (n, 0, oo)
        )
        sumatoria_general_tex = f"$$ {latex(serie_general)} $$"
        explicacion_taylor = (
            f"La serie de Taylor de \\( {latex(f)} \\) alrededor de \\( x=0 \\) es:"
        )

        # Serie truncada
        terminos = []
        for i in range(datos.n_terminos):
            deriv_i = diff(f, (x, i)).subs(x, a_taylor)
            termino_i = simplify(deriv_i/factorial(i))*(x - a_taylor)**i
            terminos.append(termino_i)

        f_series_expr = sum(terminos)
        # Texto con factoriales sin evaluar
        f_series_sumada = " + ".join(latex(t) for t in terminos) + " + \\cdots"
        f_series_tex = f"$$ {latex(f)} = {f_series_sumada} $$"

        # Integral simbólica de la serie truncada
        try:
            F_aproximada = integrate(f_series_expr, x)
            F_aproximada_tex = f"$$ {latex(F_aproximada)} $$"
        except:
            F_aproximada_tex = "No se pudo calcular la integral de la serie truncada."

        # La integral definida en notación LaTeX con los límites originales
        integral_definida_tex = (
            f"$$ \\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\; dx $$"
        )

        # 8) Métodos numéricos (Simpson, Romberg, Gauss, Monte Carlo)
        epsilon = 1e-8
        subintervalos = crear_subintervalos(a_eval, b_eval, singular_points, epsilon)

        # Simpson
        integral_simpson = simpson_subintervalos(f_lambda, subintervalos)

        # Romberg/Gauss (quad)
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

        # Monte Carlo
        integral_montecarlo = monte_carlo_subintervalos(f_lambda, subintervalos)

        # 9) Texto para GeoGebra
        geogebra_expresiones = {
            "funcion": f"f(x) = {exportar_para_geogebra(f)}",
            "taylor": f"T(x) = {exportar_para_geogebra(f_series_expr)}",
            "area_comando": f"Integral(f, {str(a_sym)}, {str(b_sym)})"
        }

        # 10) Respuesta final en JSON
        return {
            "funcion_introducida": str(f),

            "integral_definida": integral_definida_tex,
            "primitiva_real": f"$$ {F_exacta_tex} $$",
            "valor_simbolico_integral": f"$$ {valor_simbolico} $$",

            "valor_numerico_exacto": resultado_exacto_val,
            "valor_numerico_exacto_latex": f"$$ {resultado_exacto_tex} $$",

            "funciones_especiales_detectadas": funciones_especiales_detectadas,

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
        # Si ocurre un error global, lo devolvemos
        return {"error": str(e)}
