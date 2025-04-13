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

x, n = symbols('x n', real=True)  # Aseguramos que x y n sean reales

class InputDatos(BaseModel):
    funcion: str
    a: Union[str, float]
    b: Union[str, float]
    n_terminos: int = Field(default=10, ge=1, le=20)
    tolerancia: float = Field(default=1e-6, ge=1e-10)

def exportar_para_geogebra(expr):
    """
    Convierte la expresión simbólica a un string compatible con GeoGebra:
    - Reemplaza '**' por '^'
    - Reemplaza '*' por ''
    """
    expr_str = sstr(expr)
    expr_str = expr_str.replace('**', '^').replace('*', '')
    return expr_str

def obtener_funciones_especiales(expr):
    """
    Dada una expresión de sympy, busca en su representación en cadena
    ciertas subcadenas (Si, erf, Li, Ci, gamma, beta, Ei, besselj, etc.)
    para generar una lista de definiciones matemáticas (en LaTeX)
    con breve descripción. Devuelve una lista de diccionarios.
    """
    definiciones = []

    expr_str = str(expr)

    # Funciones ya existentes
    if "Si" in expr_str:
        definiciones.append({
            "funcion": "Si(x)",
            "latex": r"\mathrm{Si}(x) = \int_0^x \frac{\sin(t)}{t}\,dt",
            "descripcion": "La función seno integral, primitiva de \\( \\frac{\\sin(x)}{x} \\)."
        })
    if "erf" in expr_str:
        definiciones.append({
            "funcion": "erf(x)",
            "latex": r"\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}\,dt",
            "descripcion": "La función error, aparece al integrar \\( e^{-x^2} \\)."
        })
    if "Li" in expr_str:
        definiciones.append({
            "funcion": "Li(x)",
            "latex": r"\mathrm{Li}(x) = \int_0^x \frac{dt}{\log(t)}",
            "descripcion": "La función logaritmo integral, primitiva de \\( \\frac{1}{\\log(x)} \\)."
        })
    if "Ci" in expr_str:
        definiciones.append({
            "funcion": "Ci(x)",
            "latex": r"\mathrm{Ci}(x) = -\int_x^\infty \frac{\cos(t)}{t}\,dt",
            "descripcion": "La función coseno integral, aparece en análisis armónico y transformadas."
        })
    if "gamma" in expr_str:
        definiciones.append({
            "funcion": "Gamma(x)",
            "latex": r"\Gamma(x) = \int_0^\infty t^{x-1} e^{-t}\,dt",
            "descripcion": "Extiende el factorial a los números reales y complejos."
        })
    if "beta" in expr_str:
        definiciones.append({
            "funcion": "Beta(x, y)",
            "latex": r"B(x, y) = \int_0^1 t^{x-1}(1-t)^{y-1}\,dt",
            "descripcion": "Relacionada con Gamma y aparece en teoría de probabilidad."
        })

    # Funciones especiales adicionales
    if "Ei" in expr_str:
        definiciones.append({
            "funcion": "Ei(x)",
            "latex": r"\mathrm{Ei}(x) = \int_{-\infty}^x \frac{e^t}{t}\,dt",
            "descripcion": "La función exponencial integral, primitiva de \\( \\frac{e^x}{x} \\)."
        })
    if "besselj" in expr_str or "BesselJ" in expr_str or "J(" in expr_str:
        definiciones.append({
            "funcion": "J_n(x)",
            "latex": r"J_n(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m!\,\Gamma(m+n+1)}\left(\frac{x}{2}\right)^{2m+n}",
            "descripcion": "La función de Bessel de la primera especie."
        })
    if "bessely" in expr_str or "BesselY" in expr_str or "Y(" in expr_str:
        definiciones.append({
            "funcion": "Y_n(x)",
            "latex": r"Y_n(x) = \frac{J_n(x)\cos(n\pi)-J_{-n}(x)}{\sin(n\pi)}",
            "descripcion": "La función de Bessel de la segunda especie."
        })
    if "dawson" in expr_str:
        definiciones.append({
            "funcion": "F(x)",
            "latex": r"F(x) = e^{-x^2}\int_0^x e^{t^2}\,dt",
            "descripcion": "La función de Dawson, relacionada con 'erf'."
        })
    if "fresnels" in expr_str or "S(" in expr_str:
        definiciones.append({
            "funcion": "S(x)",
            "latex": r"S(x) = \int_0^x \sin(t^2)\,dt",
            "descripcion": "La función Fresnel S, aparece en óptica y difracción."
        })
    if "fresnelc" in expr_str or "C(" in expr_str:
        definiciones.append({
            "funcion": "C(x)",
            "latex": r"C(x) = \int_0^x \cos(t^2)\,dt",
            "descripcion": "La función Fresnel C, aparece en óptica y difracción."
        })
    if "Ai(" in expr_str or "airy" in expr_str:
        definiciones.append({
            "funcion": "Ai(x)",
            "latex": r"\mathrm{Ai}(x) = \frac{1}{\pi}\int_0^\infty \cos\left(\tfrac{t^3}{3}+xt\right)\,dt",
            "descripcion": "La función Airy Ai, solución de la ecuación diferencial de Airy."
        })
    if "Bi(" in expr_str:
        definiciones.append({
            "funcion": "Bi(x)",
            "latex": r"\mathrm{Bi}(x) = \dots",
            "descripcion": "La función Airy Bi complementa a Ai(x) en la ecuación de Airy."
        })
    if "zeta" in expr_str:
        definiciones.append({
            "funcion": "ζ(s)",
            "latex": r"\zeta(s) = \sum_{n=1}^\infty \frac{1}{n^s}",
            "descripcion": "La función zeta de Riemann, fundamental en teoría de números."
        })

    return definiciones

#############################################
# --- FUNCIONES PARA INTEGRACIÓN SOBRE SUBINTERVALOS ---
#############################################

def crear_subintervalos(a_eval, b_eval, singularidades, epsilon=1e-8):
    """
    A partir del intervalo [a_eval, b_eval] y una lista de singularidades,
    crea una lista ordenada de subintervalos (tuplas) evitando evaluar
    en una vecindad epsilon alrededor de cada singularidad.
    También maneja si a_eval o b_eval son singulares.
    """
    # Ordenar singularidades y filtrar las que estén dentro de [a_eval, b_eval]
    singular_interior = [s for s in sorted(singularidades) if a_eval <= s <= b_eval]

    # Ajuste de extremos
    a_adj = a_eval
    b_adj = b_eval
    if singular_interior and abs(singular_interior[0] - a_eval) < epsilon:
        a_adj = a_eval + epsilon
    if singular_interior and abs(b_eval - singular_interior[-1]) < epsilon:
        b_adj = b_eval - epsilon

    # Puntos de corte
    puntos = [a_adj]
    for s in singular_interior:
        s_min = s - epsilon
        s_max = s + epsilon
        if s_min > puntos[-1]:
            puntos.append(s_min)
        if s_max < b_adj:
            puntos.append(s_max)
    puntos.append(b_adj)

    subintervalos = []
    for i in range(len(puntos) - 1):
        left = puntos[i]
        right = puntos[i+1]
        # Se añade solo si left < right
        if left < right:
            subintervalos.append((left, right))

    return subintervalos

def simpson_subintervalos(f_lambda, subintervalos, n_points=1001):
    """
    Integra usando Simpson sobre cada subintervalo y retorna la suma de todas.
    """
    integral_total = 0.0
    for (sub_a, sub_b) in subintervalos:
        puntos = np.linspace(sub_a, sub_b, n_points)
        y_vals = f_lambda(puntos)
        integral_sub = simpson(y_vals, x=puntos)
        integral_total += integral_sub
    return integral_total

def monte_carlo_subintervalos(f_lambda, subintervalos, n_samples=10000):
    """
    Integra usando método Monte Carlo en cada subintervalo y suma los resultados.
    """
    total = 0.0
    for (sub_a, sub_b) in subintervalos:
        acum = 0.0
        for _ in range(n_samples):
            x_rand = random.uniform(sub_a, sub_b)
            acum += f_lambda([x_rand])[0]
        # (sub_b - sub_a) por el valor promedio
        total += (sub_b - sub_a) * (acum / n_samples)
    return total

#############################################
# --- END FUNCIONES PARA INTEGRACIÓN ---
#############################################

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    """
    Procesa una función (en formato string), con límites a y b, y devuelve:
      - Primitiva (cuando exista)
      - Detección de funciones especiales en integrando y primitiva
      - Integral definida exacta (Sympy)
      - Varios métodos numéricos (Simpson, Romberg, Gauss, Monte Carlo)
      - Serie de Taylor
      - Expresiones para GeoGebra
      - Advertencias de singularidades
    """
    try:
        # 1. Parseo de los datos de entrada
        a_sym = sympify(datos.a)
        b_sym = sympify(datos.b)
        a_eval = float(N(a_sym))
        b_eval = float(N(b_sym))

        if a_eval >= b_eval:
            raise HTTPException(
                status_code=400,
                detail="El límite inferior debe ser menor que el superior."
            )

        # 2. Construcción de la función simbólica y lambda
        f = sympify(datos.funcion)
        if str(f) == 'sin(x)/x':
            def f_lambda(x_val):
                return np.where(x_val == 0, 1.0, np.sin(x_val)/x_val)
        else:
            f_lambda = sp.lambdify(x, f, modules=['numpy'])

        # 3. Detección de singularidades (interior y extremos)
        advertencias = []
        singular_points = set()

        # 3a. Sing interior mediante solveset
        try:
            posibles_sing = solveset(1/f, x, domain=Interval(a_eval, b_eval))
            for p in posibles_sing:
                val = float(p.evalf())
                if a_eval <= val <= b_eval:
                    singular_points.add(val)
                    advertencias.append(f"⚠️ Posible singularidad en x={val}")
        except:
            pass

        # 3b. Sing en extremos
        # Ver si f(a_eval) es finito
        try:
            if not np.isfinite(f_lambda(a_eval)):
                singular_points.add(a_eval)
                advertencias.append(f"⚠️ Singularidad en x={a_eval} (límite inferior)")
        except:
            singular_points.add(a_eval)
            advertencias.append(f"⚠️ Singularidad en x={a_eval} (límite inferior)")

        # Ver si f(b_eval) es finito
        try:
            if not np.isfinite(f_lambda(b_eval)):
                singular_points.add(b_eval)
                advertencias.append(f"⚠️ Singularidad en x={b_eval} (límite superior)")
        except:
            singular_points.add(b_eval)
            advertencias.append(f"⚠️ Singularidad en x={b_eval} (límite superior)")

        # 4. Hallar la primitiva exacta (si existe)
        F_exacta_tex = ""
        valor_simbolico = "Valor simbólico no disponible"

        # Casos especiales (sin x / x, e^(-x^2))
        if str(f) == 'sin(x)/x':
            F_exacta_tex = r"\mathrm{Si}(x)"
            valor_simbolico = rf"\mathrm{{Si}}({latex(b_sym)}) - \mathrm{{Si}}({latex(a_sym)})"
        elif str(f) == 'exp(-x**2)':
            valor_simbolico = (rf"2 \sqrt{{\pi}} \cdot \mathrm{{erf}}({latex(b_sym)})"
                               if a_sym == -b_sym else
                               rf"\sqrt{{\pi}} \cdot (\mathrm{{erf}}({latex(b_sym)}) - \mathrm{{erf}}({latex(a_sym)}))")
            F_exacta_tex = r"\frac{\sqrt{\pi}}{2}\mathrm{erf}(x)"
        else:
            try:
                F_expr = integrate(f, x)
                F_exacta_tex = latex(F_expr)
                valor_simbolico = latex(F_expr.subs(x, b_sym) - F_expr.subs(x, a_sym))
            except:
                F_exacta_tex = "No tiene primitiva elemental"

        # 5. Detección de funciones especiales en la función y en la primitiva
        #    Esto asegura que si la primitiva es, por ejemplo, Li(x),
        #    aparezca la definición en el resultado.
        funcs_en_f = obtener_funciones_especiales(f)
        if 'F_expr' in locals():
            funcs_en_primitiva = obtener_funciones_especiales(F_expr)
        else:
            funcs_en_primitiva = []
        # Combinamos en un solo dict para evitar duplicados.
        # Usaremos el "nombre" de la función como clave.
        all_funcs = {}
        for dic in (funcs_en_f + funcs_en_primitiva):
            all_funcs[dic["funcion"]] = dic
        # listamos de nuevo
        funciones_especiales_detectadas = list(all_funcs.values())

        # 6. Integral definida exacta con sympy
        try:
            resultado_exacto = integrate(f, (x, a_eval, b_eval))
            resultado_exacto_val = float(N(resultado_exacto))
            resultado_exacto_tex = latex(resultado_exacto)
        except:
            resultado_exacto_val = None
            resultado_exacto_tex = "No calculable simbólicamente"

        # 7. Serie de Taylor
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

        # 8. Integramos usando métodos numéricos con subintervalos
        epsilon = 1e-8
        subintervalos = crear_subintervalos(a_eval, b_eval, singular_points, epsilon)

        # 8.1 Simpson
        integral_simpson = simpson_subintervalos(f_lambda, subintervalos)

        # 8.2 Romberg y Gauss (quad)
        # pasamos 'points' con los puntos singulares
        puntos_criticos = sorted(singular_points)
        try:
            integral_romberg, _ = quad(f_lambda, a_eval, b_eval, points=puntos_criticos)
        except:
            integral_romberg = None
            advertencias.append("No se pudo calcular con Romberg (singularidades).")

        try:
            integral_gauss, _ = quad(f_lambda, a_eval, b_eval, points=puntos_criticos)
        except:
            integral_gauss = None
            advertencias.append("No se pudo calcular con Cuadratura Gaussiana (singularidades).")

        # 8.3 Monte Carlo
        integral_montecarlo = monte_carlo_subintervalos(f_lambda, subintervalos)

        # 9. Texto de GeoGebra
        #    - f(x) = ...
        #    - T(x) = ...
        #    - Integral(f, a, b)
        geogebra_expresiones = {
            "funcion": f"f(x) = {exportar_para_geogebra(f)}",
            "taylor": f"T(x) = {exportar_para_geogebra(f_series_expr)}",
            "area_comando": f"Integral(f, {a_eval}, {b_eval})"
        }

        # 10. Construcción de la respuesta JSON
        return {
            "funcion_introducida": str(f),
            "primitiva_real": f"$$ {F_exacta_tex} $$",
            "valor_simbolico_integral": f"$$ {valor_simbolico} $$",
            "valor_numerico_exacto": resultado_exacto_val,
            "valor_numerico_exacto_latex": f"$$ {resultado_exacto_tex} $$",
            "funciones_especiales_detectadas": funciones_especiales_detectadas,

            "serie_taylor_general": sumatoria_general_tex,
            "explicacion_taylor_general": explicacion_taylor,
            "serie_taylor_finita": f_series_tex,
            "integral_definida": integral_definida_tex,
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
