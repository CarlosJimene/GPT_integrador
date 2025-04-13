from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union
import numpy as np
from sympy import (
    symbols, sympify, integrate, solveset, Interval, oo,
    diff, factorial, Sum, latex, simplify, N, sstr, Function,
    erf, sqrt, pi, erfi, sin, cos, fresnels, fresnelc, Si, Ci,
    gamma, lowergamma, uppergamma
)
from scipy.integrate import simpson, quad
import sympy as sp
import random

#############################################################
# Instancia de la aplicación FastAPI
#############################################################
app = FastAPI()

# Variables simbólicas
x, n = symbols('x n', real=True)

class InputDatos(BaseModel):
    funcion: str         # e.g. "sin(x^2)"
    a: Union[str, float] # e.g. "-sqrt(pi)"
    b: Union[str, float] # e.g. "sqrt(pi)"
    n_terminos: int = Field(default=10, ge=1, le=20)
    tolerancia: float = Field(default=1e-6, ge=1e-10)

#############################################################
# Herramientas auxiliares
#############################################################
def exportar_para_geogebra(expr):
    """
    Convierte una expresión Sympy a string apto para GeoGebra:
      - '**' -> '^'
      - '*' -> ''
    """
    expr_str = sstr(expr)
    expr_str = expr_str.replace('**', '^').replace('*', '')
    return expr_str

def crear_subintervalos(a_eval, b_eval, singularidades, epsilon=1e-8):
    """
    Crea subintervalos evitando epsilon alrededor de cada singularidad
    en [a_eval, b_eval].
    """
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
    puntos.append(b_eval)

    subintervalos = []
    for i in range(len(puntos) - 1):
        izq, der = puntos[i], puntos[i+1]
        if izq < der:
            subintervalos.append((izq, der))
    return subintervalos

def simpson_subintervalos(f_lambda, subintervalos, n_points=1001):
    """
    Integra con Simpson cada subintervalo y suma.
    Devuelve el resultado y el número de puntos utilizados.
    """
    total = 0.0
    total_points = 0
    for (a_i, b_i) in subintervalos:
        pts = np.linspace(a_i, b_i, n_points)
        vals = f_lambda(pts)
        total += simpson(vals, x=pts)
        total_points += len(pts)
    return total, total_points

def monte_carlo_subintervalos(f_lambda, subintervalos, n_samples=10000):
    """
    Integra con Monte Carlo cada subintervalo y acumula.
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
    Detecta funciones especiales en expr y devuelve info LaTeX + descripción.
    """
    definiciones = []
    if expr is None:
        return definiciones

    expr_str = str(expr).lower()

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
    if "erf" in expr_str and "erfi" not in expr_str:
        definiciones.append({
            "funcion": r"\mathrm{erf}(x)",
            "latex": r"\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2}\,dt",
            "descripcion": "Función error, surge al integrar e^{-x^2}."
        })
    if "erfi" in expr_str:
        definiciones.append({
            "funcion": r"\mathrm{erfi}(x)",
            "latex": r"\mathrm{erfi}(x) = -i\,\mathrm{erf}(i\,x)",
            "descripcion": "Función error imaginaria, aparece con e^{x^2}."
        })
    if "si(" in expr_str: # Usamos "si(" para evitar falsos positivos con "sin"
        definiciones.append({
            "funcion": r"\mathrm{Si}(x)",
            "latex": r"\mathrm{Si}(x) = \int_{0}^{x} \frac{\sin(t)}{t}\,dt",
            "descripcion": "Seno integral, primitiva de sin(x)/x."
        })
    if "ci(" in expr_str: # Usamos "ci(" para evitar falsos positivos con "cos"
        definiciones.append({
            "funcion": r"\mathrm{Ci}(x)",
            "latex": r"\mathrm{Ci}(x) = \gamma + \ln|x| + \int_{0}^{x} \frac{\cos(t) - 1}{t}\,dt",
            "descripcion": "Coseno integral, primitiva de cos(x)/x."
        })
    if "gamma(" in expr_str and "lowergamma" not in expr_str and "uppergamma" not in expr_str:
        definiciones.append({
            "funcion": r"\Gamma(z)",
            "latex": r"\Gamma(z) = \int_{0}^{\infty} t^{z-1} e^{-t}\,dt",
            "descripcion": "Función Gamma, generalización del factorial."
        })
    if "lowergamma(" in expr_str:
        definiciones.append({
            "funcion": r"\gamma(s, x)",
            "latex": r"\gamma(s, x) = \int_{0}^{x} t^{s-1} e^{-t}\,dt",
            "descripcion": "Función Gamma incompleta inferior."
        })
    if "uppergamma(" in expr_str:
        definiciones.append({
            "funcion": r"\Gamma(s, x)",
            "latex": r"\Gamma(s, x) = \int_{x}^{\infty} t^{s-1} e^{-t}\,dt",
            "descripcion": "Función Gamma incompleta superior."
        })
    return definiciones

#############################################################
# PRIMITIVAS ESPECIALES: sin(x^2) y cos(x^2)
#############################################################
def primitiva_sin_x2():
    """
    Retorna la primitiva simbólica de sin(x^2) en forma Sympy,
    que es (sqrt(pi)/2)*fresnels(x/sqrt(pi)).
    """
    from sympy import sqrt, pi, fresnels
    return (sqrt(pi)/2) * fresnels(x/sqrt(pi))

def primitiva_cos_x2():
    """
    Primitiva simbólica de cos(x^2) = (sqrt(pi)/2)*fresnelc(x/sqrt(pi))
    """
    from sympy import sqrt, pi, fresnelc
    return (sqrt(pi)/2) * fresnelc(x/sqrt(pi))

#############################################################
# ENDPOINT
#############################################################
@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    """
    Calcula la primitiva (en funciones Fresnel si sin(x^2) o cos(x^2)),
    evalúa la integral definida, serie de Taylor, funciones especiales,
    y genera texto para GeoGebra.
    """
    try:
        # 1) LÍMITES
        a_sym = sympify(datos.a)   # e.g. -sqrt(pi)
        b_sym = sympify(datos.b)   # e.g. sqrt(pi)
        a_eval = float(N(a_sym))
        b_eval = float(N(b_sym))
        if a_eval >= b_eval:
            raise HTTPException(status_code=400, detail="Límite inferior >= límite superior.")

        # 2) FUNCIÓN
        f_str = datos.funcion.strip() # e.g. "sin(x^2)"
        f = sympify(f_str)
        # Lambdify para numérico
        if str(f) == "sin(x)/x":
            def f_lambda(z):
                z_arr = np.array(z, ndmin=1)
                vals = np.where(np.isclose(z_arr, 0.0, atol=1e-14),
                                1.0,
                                np.sin(z_arr)/z_arr)
                return vals if len(vals) > 1 else vals[0]
        else:
            f_lambda = sp.lambdify(x, f, modules=['numpy'])

        # 3) SINGULARIDADES
        advertencias = []
        singular_points = set()

        def es_finito(val):
            try:
                return np.isfinite(val)
            except:
                return False

        # Interior
        try:
            interior_sings = solveset(1/f, x, domain=Interval(a_eval, b_eval))
            for p in interior_sings:
                val_sing = float(p.evalf())
                if a_eval < val_sing < b_eval:
                    singular_points.add(val_sing)
                    advertencias.append(f"⚠️ Posible singularidad en x={val_sing}")
        except:
            pass

        # Extremos
        try:
            va = f_lambda(a_eval)
            if isinstance(va, (list, np.ndarray)):
                va = va[0]
            if not es_finito(va):
                singular_points.add(a_eval)
                advertencias.append(f"⚠️ Singularidad en x={a_eval} (límite inferior)")
        except:
            singular_points.add(a_eval)
            advertencias.append(f"⚠️ Singularidad en x={a_eval} (límite inferior)")

        try:
            vb = f_lambda(b_eval)
            if isinstance(vb, (list, np.ndarray)):
                vb = vb[0]
            if not es_finito(vb):
                singular_points.add(b_eval)
                advertencias.append(f"⚠️ Singularidad en x={b_eval} (límite superior)")
        except:
            singular_points.add(b_eval)
            advertencias.append(f"⚠️ Singularidad en x={b_eval} (límite superior)")

        # 4) PRIMITIVA
        if f_str == "sin(x^2)":
            F_expr = primitiva_sin_x2()  # En forma Sympy
        elif f_str == "cos(x^2)":
            F_expr = primitiva_cos_x2()
        else:
            try:
                F_expr = integrate(f, x)
            except Exception as e:
                F_expr = None
                advertencias.append(f"Error al intentar integrar simbólicamente: {e}")

        if F_expr is None:
            F_exacta_tex = "No tiene primitiva elemental conocida por SymPy"
            valor_simbolico = "Valor simbólico no disponible"
        else:
            F_exacta_tex = latex(F_expr)
            # Valor simbólico => F(b) - F(a)
            try:
                valor_sym = F_expr.subs(x, b_sym) - F_expr.subs(x, a_sym)
                valor_simbolico = latex(valor_sym)
            except Exception as e:
                valor_simbolico = f"Error al evaluar la primitiva simbólica: {e}"

        # 5) FUNCIONES ESPECIALES DETECTADAS
        funcs_en_f = obtener_funciones_especiales(f)
        funcs_en_F = obtener_funciones_especiales(F_expr) if F_expr is not None else []
        all_funcs = {}
        for dic in (funcs_en_f + funcs_en_F):
            all_funcs[dic["funcion"]] = dic
        funciones_especiales_detectadas = list(all_funcs.values())

        # 6) INTEGRAL DEFINIDA EXACTA y VALOR NUMÉRICO
        #    Forzamos un camino especial si es sin(x^2) o cos(x^2) y ya tenemos primitiva
        if f_str in ["sin(x^2)", "cos(x^2)"] and F_expr is not None:
            try:
                resultado_exact_expr = F_expr.subs(x, b_sym) - F_expr.subs(x, a_sym)
                resultado_exacto_val = float(N(resultado_exact_expr))
                resultado_exacto_tex = latex(resultado_exact_expr)
            except Exception as e:
                resultado_exacto_val = None
                resultado_exacto_tex = f"No calculable simbólicamente (evaluación): {e}"
        else:
            # Caso general: Sympy integrate definido
            try:
                resultado_def = integrate(f, (x, a_sym, b_sym))
                resultado_exacto_val = float(N(resultado_def))
                resultado_exacto_tex = latex(resultado_def)
            except Exception as e:
                resultado_exacto_val = None
                resultado_exacto_tex = f"No calculable simbólicamente (integración definida): {e}"

        # 7) SERIE DE TAYLOR
        if f_str == "sin(x^2)":
            # Serie infinita de sin(x^2)
            serie_infinita = (
                r"\sin(x^2) = \sum_{k=0}^{\infty} \frac{(-1)^k\, x^{4k+2}}{(2k+1)!}"
            )
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = "La serie de Taylor de \\(\\sin(x^2)\\) alrededor de \\(x=0\\) es:"
            terminos = []
            for k in range(datos.n_terminos):
                coef = (-1)**k
                potencia = 4*k + 2
                term_expr = coef * (x**potencia) / factorial(2*k + 1)
                terminos.append(term_expr)
            f_series_expr = sum(terminos)
            serie_latex_terms = [latex(t) for t in terminos]
            f_series_sumada = " + ".join(serie_latex_terms) + r" + \cdots"
            f_series_tex = f"$$ \\sin(x^2) = {f_series_sumada} $$"

        elif f_str == "cos(x^2)":
            # Serie infinita de cos(x^2)
            serie_infinita = (
                r"\cos(x^2) = \sum_{k=0}^{\infty} \frac{(-1)^k\, x^{4k}}{(2k)!}"
            )
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = "La serie de Taylor de \\(\\cos(x^2)\\) alrededor de \\(x=0\\) es:"
            terminos = []
            for k in range(datos.n_terminos):
                coef = (-1)**k
                potencia = 4*k
                term_expr = coef*(x**potencia)/factorial(2*k)
                terminos.append(term_expr)
            f_series_expr = sum(terminos)
            serie_latex_terms = [latex(t) for t in terminos]
            f_series_sumada = " + ".join(serie_latex_terms) + r" + \cdots"
            f_series_tex = f"$$ \\cos(x^2) = {f_series_sumada} $$"

        else:
            # Caso general
            a_taylor = 0
            serie_general_expr = Sum(
                diff(f, (x, n)).subs(x, a_taylor)/factorial(n)*(x-a_taylor)**n, (n, 0, oo)
            )
            sumatoria_general_tex = f"$$ {latex(serie_general_expr)} $$"
            explicacion_taylor = f"La serie de Taylor de \\({latex(f)}\\) alrededor de x=0 es:"
            terminos = []
            for i in range(datos.n_terminos):
                deriv_i = diff(f, (x, i)).subs(x, a_taylor)
                term_expr = simplify(deriv_i/factorial(i))*(x-a_taylor)**i
                terminos.append(term_expr)
            f_series_expr = sum(terminos)
            serie_latex_terms = [latex(t) for t in terminos]
            f_series_sumada = " + ".join(serie_latex_terms) + r" + \cdots"
            f_series_tex = f"$$ {latex(f)} = {f_series_sumada} $$"

        # Integración simbólica de la serie truncada
        try:
            F_aproximada = integrate(f_series_expr, x)
            F_aproximada_tex = f"$$ {latex(F_aproximada)} $$"
        except:
            F_aproximada_tex = "No se pudo calcular la integral de la serie truncada."

        # 8) Integral definida en notación LaTeX (con límites simbólicos)
        integral_definida_tex = f"$$ \\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\, dx $$"

        # 9) MÉTODOS NUMÉRICOS
        subintervalos = crear_subintervalos(a_eval, b_eval, singular_points)
        integral_simpson, n_puntos_simpson = simpson_subintervalos(f_lambda, subintervalos)
        points_sing = sorted(list(singular_points))

        try:
            val_romberg, _ = quad(f_lambda, a_eval, b_eval, points=points_sing)
            integral_romberg = val_romberg
        except Exception as e:
            integral_romberg = None
            advertencias.append(f"No se pudo calcular con Romberg (singularidades): {e}")

        try:
            val_gauss, _ = quad(f_lambda, a_eval, b_eval, points=points_sing)
            integral_gauss = val_gauss
        except Exception as e:
            integral_gauss = None
            advertencias.append(f"No se pudo calcular con Cuadratura Gaussiana (singularidades): {e}")

        integral_montecarlo = monte_carlo_subintervalos(f_lambda, subintervalos)

        # 10) TEXTO PARA GEOGEBRA
        geogebra_expresiones = {
            "funcion": f"f(x) = {exportar_para_geogebra(f)}",
            "taylor": f"T(x) = {exportar_para_geogebra(f_series_expr)}",
            "area_comando": f"Integral(f, {str(a_sym)}, {str(b_sym)})"
        }

        # 11) DEVOLVER JSON
        return {
            "funcion_introducida": str(f),
            "primitiva_real": f"$$ {F_exacta_tex} $$",
            "funciones_especiales_detectadas": funciones_especiales_detectadas,
            "valor_simbolico_integral": f"$$ {valor_simbolico} $$",
            "valor_numerico_exacto": resultado_exacto_val,
            "valor_numerico_exacto_latex": f"$$ {resultado_exacto_tex} $$",
            "integral_definida": integral_definida_tex,
            "serie_taylor_general": sumatoria_general_tex,
            "explicacion_taylor_general": explicacion_taylor,
            "serie_taylor_finita": f_series_tex,
            "integral_serie_taylor": F_aproximada_tex,
            "metodos_numericos": {
                "simpson": {"value": integral_simpson, "n_points": n_puntos_simpson},
                "romberg": integral_romberg,
                "cuadratura_gaussiana": integral_gauss,
                "montecarlo": integral_montecarlo
            },
            "advertencias": advertencias,
            "geogebra_expresiones": geogebra_expresiones
        }

    except Exception as e:
        return {"error": str(e)}
