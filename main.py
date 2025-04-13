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

    # Funciones ya existentes
    if "Si" in str(expr):
        definiciones.append({
            "funcion": "Si(x)",
            "latex": r"\mathrm{Si}(x) = \int_0^x \frac{\sin(t)}{t}\,dt",
            "descripcion": "La función seno integral aparece como primitiva de \\( \\frac{\\sin(x)}{x} \\)."
        })
    if "erf" in str(expr):
        definiciones.append({
            "funcion": "erf(x)",
            "latex": r"\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}\,dt",
            "descripcion": "La función error aparece al integrar \\( e^{-x^2} \\)."
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
            "latex": r"\mathrm{Ci}(x) = -\int_x^\infty \frac{\cos(t)}{t}\,dt",
            "descripcion": "La función coseno integral aparece en análisis armónico y transformadas."
        })
    if "gamma" in str(expr):
        definiciones.append({
            "funcion": "Gamma(x)",
            "latex": r"\Gamma(x) = \int_0^\infty t^{x-1} e^{-t}\,dt",
            "descripcion": "Extiende el factorial a los números reales y complejos."
        })
    if "beta" in str(expr):
        definiciones.append({
            "funcion": "Beta(x, y)",
            "latex": r"B(x, y) = \int_0^1 t^{x-1}(1-t)^{y-1}\,dt",
            "descripcion": "Relacionada con la función Gamma. Aparece en teoría de probabilidad."
        })

    # Funciones especiales adicionales
    if "Ei" in str(expr):
        definiciones.append({
            "funcion": "Ei(x)",
            "latex": r"\mathrm{Ei}(x) = \int_{-\infty}^x \frac{e^t}{t}\,dt",
            "descripcion": "La función exponencial integral surge al integrar \\( \\frac{e^x}{x} \\)."
        })
    if "besselj" in str(expr) or "BesselJ" in str(expr) or "J(" in str(expr):
        definiciones.append({
            "funcion": "J_n(x)",
            "latex": r"J_n(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m!\,\Gamma(m+n+1)}\left(\frac{x}{2}\right)^{2m+n}",
            "descripcion": "La función de Bessel de la primera especie aparece en problemas con simetría cilíndrica."
        })
    if "bessely" in str(expr) or "BesselY" in str(expr) or "Y(" in str(expr):
        definiciones.append({
            "funcion": "Y_n(x)",
            "latex": r"Y_n(x) = \frac{J_n(x)\cos(n\pi)-J_{-n}(x)}{\sin(n\pi)}",
            "descripcion": "La función de Bessel de la segunda especie complementa a J_n(x)."
        })
    if "dawson" in str(expr):
        definiciones.append({
            "funcion": "F(x)",
            "latex": r"F(x) = e^{-x^2}\int_0^x e^{t^2}\,dt",
            "descripcion": "La función de Dawson se relaciona con la función error (erf)."
        })
    if "fresnels" in str(expr) or "S(" in str(expr):
        definiciones.append({
            "funcion": "S(x)",
            "latex": r"S(x) = \int_0^x \sin(t^2)\,dt",
            "descripcion": "La función Fresnel S aparece en óptica y difracción."
        })
    if "fresnelc" in str(expr) or "C(" in str(expr):
        definiciones.append({
            "funcion": "C(x)",
            "latex": r"C(x) = \int_0^x \cos(t^2)\,dt",
            "descripcion": "La función Fresnel C aparece en óptica y difracción."
        })
    if "Ai(" in str(expr) or "airy" in str(expr):
        definiciones.append({
            "funcion": "Ai(x)",
            "latex": r"\mathrm{Ai}(x) = \frac{1}{\pi}\int_0^\infty \cos\left(\tfrac{t^3}{3}+xt\right)\,dt",
            "descripcion": "La función Airy Ai es solución de la ecuación diferencial de Airy."
        })
    if "Bi(" in str(expr) or "airy" in str(expr):
        definiciones.append({
            "funcion": "Bi(x)",
            "latex": r"\mathrm{Bi}(x) = \dots",
            "descripcion": "La función Airy Bi complementa a Ai(x) en la ecuación de Airy."
        })
    if "zeta" in str(expr):
        definiciones.append({
            "funcion": "ζ(s)",
            "latex": r"\zeta(s) = \sum_{n=1}^\infty \frac{1}{n^s}",
            "descripcion": "La famosa función zeta de Riemann, clave en la teoría de números."
        })

    return definiciones

#############################################
# --- FUNCIONES PARA INTEGRACIÓN SOBRE SUBINTERVALOS ---
#############################################

def crear_subintervalos(a_eval, b_eval, singularidades, epsilon=1e-8):
    """
    A partir del intervalo [a_eval, b_eval] y una lista de singularidades,
    crea una lista ordenada de subintervalos (en forma de tuplas) evitando evaluaciones
    en una vecindad epsilon alrededor de cada singularidad.
    Se consideran también singularidades en los extremos.
    """
    # Ordenar las singularidades y filtrar las que se encuentren dentro del intervalo
    singular_interior = [s for s in sorted(singularidades) if a_eval <= s <= b_eval]

    # Ajustar los extremos en caso de singularidad
    a_adj = a_eval
    b_adj = b_eval
    if singular_interior and abs(singular_interior[0] - a_eval) < epsilon:
        a_adj = a_eval + epsilon
    if singular_interior and abs(b_eval - singular_interior[-1]) < epsilon:
        b_adj = b_eval - epsilon

    # Preparar la lista de "puntos de corte"
    puntos = [a_adj]
    for s in singular_interior:
        if s - epsilon > puntos[-1]:
            puntos.append(s - epsilon)
        if s + epsilon < b_adj:
            puntos.append(s + epsilon)
    puntos.append(b_adj)

    # Crear subintervalos a partir de puntos consecutivos
    subintervalos = []
    for i in range(len(puntos) - 1):
        # Solo agregar subintervalos de longitud positiva
        if puntos[i] < puntos[i+1]:
            subintervalos.append((puntos[i], puntos[i+1]))
    return subintervalos

def simpson_subintervalos(f_lambda, subintervalos, n_points=1001):
    """
    Integra usando Simpson sobre cada subintervalo y retorna la suma.
    Se genera una malla específica para cada subintervalo.
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
    Integra usando método Monte Carlo sobre cada subintervalo y suma los resultados.
    """
    total = 0.0
    for (sub_a, sub_b) in subintervalos:
        suma = 0.0
        for _ in range(n_samples):
            x_rand = random.uniform(sub_a, sub_b)
            suma += f_lambda(np.array([x_rand]))[0]
        total += (sub_b - sub_a) * suma / n_samples
    return total

#############################################
# --- END FUNCIONES PARA INTEGRACIÓN ---
#############################################

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        # Convertir a objetos sympy y numéricos
        a_sym = sympify(datos.a)
        b_sym = sympify(datos.b)
        a_eval = float(N(a_sym))
        b_eval = float(N(b_sym))

        if a_eval >= b_eval:
            raise HTTPException(status_code=400, detail="El límite inferior debe ser menor que el superior.")

        f = sympify(datos.funcion)

        # Conversión a función numérica
        if str(f) == 'sin(x)/x':
            def f_lambda(x_val):
                # Evitar división por 0 en x=0
                return np.where(x_val == 0, 1.0, np.sin(x_val)/x_val)
        else:
            f_lambda = sp.lambdify(x, f, modules=['numpy'])

        advertencias = []
        singular_points = set()

        # --- 1. DETECCIÓN DE SINGULARIDADES ---
        # a) Usando solveset para detectar singularidades en (a,b)
        try:
            posibles_sing = solveset(1/f, x, domain=Interval(a_eval, b_eval))
            for p in posibles_sing:
                try:
                    val = float(p.evalf())
                    # Verificar si la singularidad está en el intervalo (o cerca de los extremos)
                    if a_eval <= val <= b_eval:
                        singular_points.add(val)
                        advertencias.append(f"⚠️ Posible singularidad en x = {val}")
                except Exception:
                    continue
        except Exception:
            pass

        # b) Comprobamos singularidades en los extremos usando f_lambda:
        try:
            fa = f_lambda(a_eval)
            if not np.isfinite(fa):
                singular_points.add(a_eval)
                advertencias.append(f"⚠️ Singularidad en el límite inferior x = {a_eval}")
        except Exception:
            singular_points.add(a_eval)
            advertencias.append(f"⚠️ Singularidad en el límite inferior x = {a_eval}")

        try:
            fb = f_lambda(b_eval)
            if not np.isfinite(fb):
                singular_points.add(b_eval)
                advertencias.append(f"⚠️ Singularidad en el límite superior x = {b_eval}")
        except Exception:
            singular_points.add(b_eval)
            advertencias.append(f"⚠️ Singularidad en el límite superior x = {b_eval}")

        # --- 2. FUNCIONES ESPECIALES Y PRIMITIVA SIMBÓLICA ---
        funciones_especiales = []
        F_exacta_tex = ""
        valor_simbolico = "Valor simbólico no disponible"

        if str(f) == 'sin(x)/x':
            F_exacta_tex = r"\mathrm{Si}(x)"
            valor_simbolico = rf"\mathrm{{Si}}({latex(b_sym)}) - \mathrm{{Si}}({latex(a_sym)})"
        elif str(f) == 'exp(-x**2)':
            valor_simbolico = (rf"2 \sqrt{{\pi}} \cdot \mathrm{{erf}}({latex(b_sym)})"
                               if a_sym == -b_sym else
                               rf"\sqrt{{\pi}} \cdot (\mathrm{{erf}}({latex(b_sym)}) - \mathrm{{erf}}({latex(a_sym)}))")
            F_exacta_tex = r"\frac{\sqrt{\pi}}{2} \cdot \mathrm{erf}(x)"
        else:
            try:
                F_expr = integrate(f, x)
                F_exacta_tex = f"{latex(F_expr)}"
                valor_simbolico = rf"{latex(F_expr.subs(x, b_sym) - F_expr.subs(x, a_sym))}"
            except Exception:
                F_exacta_tex = "No tiene primitiva elemental"

        funciones_especiales += obtener_funciones_especiales(f)

        # --- 3. INTEGRACIÓN SIMBÓLICA EXACTA ---
        try:
            resultado_exacto = integrate(f, (x, a_eval, b_eval))
            resultado_exacto_val = float(N(resultado_exacto))
            resultado_exacto_tex = f"{latex(resultado_exacto)}"
        except Exception:
            resultado_exacto_val = None
            resultado_exacto_tex = "No calculable simbólicamente"

        # --- 4. SERIE DE TAYLOR ---
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

        # --- 5. MÉTODOS NUMÉRICOS CON SUBINTERVALOS ---
        # Definir un epsilon para evitar evaluar directamente en los puntos singulares
        epsilon = 1e-8
        subintervalos = crear_subintervalos(a_eval, b_eval, singular_points, epsilon)

        # 5.1 Simpson: integrar en cada subintervalo
        integral_simpson = simpson_subintervalos(f_lambda, subintervalos)

        # 5.2 Romberg y Gauss: pasamos la lista de puntos singulares a quad
        puntos_singular = sorted(list(singular_points))
        try:
            integral_romberg, _ = quad(f_lambda, a_eval, b_eval, points=puntos_singular)
        except Exception:
            integral_romberg = None
            advertencias.append("No se pudo calcular con Romberg (singularidad).")

        try:
            integral_gauss, _ = quad(f_lambda, a_eval, b_eval, points=puntos_singular)
        except Exception:
            integral_gauss = None
            advertencias.append("No se pudo calcular con Cuadratura Gaussiana (singularidad).")

        # 5.3 Monte Carlo: integrar en cada subintervalo
        integral_montecarlo = monte_carlo_subintervalos(f_lambda, subintervalos)

        # --- 6. SALIDA PARA GEOGEBRA ---
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
