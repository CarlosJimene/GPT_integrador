from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union
import numpy as np
from sympy import (
    symbols, sympify, integrate, solveset, Interval, oo,
    diff, factorial, Sum, latex, simplify, N, sstr, Function,
    erf, sqrt, pi, erfi, sin, cos, fresnels, fresnelc, Si, Ci,
    gamma, lowergamma, uppergamma, binomial, Rational,
    # Se eliminan importaciones directas de funciones elípticas para evitar problemas.
)
from scipy.integrate import simpson, quad
import sympy as sp
import random

#############################################################
# Instancia de la aplicación FastAPI
#############################################################
app = FastAPI()

# Declaramos variables simbólicas
x, n = symbols('x n', real=True)

class InputDatos(BaseModel):
    funcion: str         # Ejemplo: "exp(-x^2)" o "sqrt(1 - x^4)" o "sin(x)/x" o "sin(x^2)"
    a: Union[str, float] # Ejemplo: "-sqrt(pi)" o "-0.5"
    b: Union[str, float] # Ejemplo: "sqrt(pi)" o "0.5"
    n_terminos: int = Field(default=10, ge=1, le=20)
    tolerancia: float = Field(default=1e-6, ge=1e-10)

#############################################################
# Herramientas auxiliares
#############################################################
def exportar_para_geogebra(expr):
    """
    Convierte una expresión Sympy a string apto para GeoGebra:
      - '**' se reemplaza por '^'
      - Se eliminan los asteriscos para multiplicación implícita.
    """
    expr_str = sstr(expr)
    expr_str = expr_str.replace('**', '^').replace('*', '')
    return expr_str

def crear_subintervalos(a_eval, b_eval, singularidades, epsilon=1e-8):
    """
    Genera subintervalos en [a_eval, b_eval] evitando evaluar en una
    vecindad epsilon alrededor de cada singularidad.
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
    puntos.append(b_adj)
    subintervalos = []
    for i in range(len(puntos) - 1):
        izq, der = puntos[i], puntos[i+1]
        if izq < der:
            subintervalos.append((izq, der))
    return subintervalos

def simpson_subintervalos(f_lambda, subintervalos, n_points_simpson=1001):
    """
    Aplica la regla de Simpson en cada subintervalo y retorna:
      - El valor total de la integral.
      - El número total de puntos utilizados.
    """
    total = 0.0
    total_points = 0
    for (a_i, b_i) in subintervalos:
        pts = np.linspace(a_i, b_i, n_points_simpson)
        vals = f_lambda(pts)
        total += simpson(vals, x=pts)
        total_points += len(pts)
    return total, total_points

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
        total += (b_i - a_i) * (acum / n_samples)
    return total

def obtener_funciones_especiales(expr):
    """
    Detecta funciones especiales en 'expr' y devuelve una lista de diccionarios con:
      - "funcion": notación textual (por ejemplo, "Si(x)", "E(x|m)", etc.)
      - "latex": definición en LaTeX.
      - "descripcion": breve explicación.
    Se amplía para detectar funciones elípticas.
    """
    definiciones = []
    if expr is None:
        return definiciones
    expr_str = str(expr).lower()
    # Fresnel
    if "fresnels" in expr_str:
        definiciones.append({
            "funcion": "S(z)",
            "latex": r"S(z) = \int_{0}^{z}\sin(t^2)\,dt",
            "descripcion": "Función Fresnel S, antiderivada de sin(x^2)."
        })
    if "fresnelc" in expr_str:
        definiciones.append({
            "funcion": "C(z)",
            "latex": r"C(z) = \int_{0}^{z}\cos(t^2)\,dt",
            "descripcion": "Función Fresnel C, antiderivada de cos(x^2)."
        })
    # erf y erfi
    if "erf(" in expr_str and "erfi(" not in expr_str:
        definiciones.append({
            "funcion": r"\mathrm{erf}(x)",
            "latex": r"\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}}\int_{0}^{x}e^{-t^2}\,dt",
            "descripcion": "Función error."
        })
    if "erfi(" in expr_str:
        definiciones.append({
            "funcion": r"\mathrm{erfi}(x)",
            "latex": r"\mathrm{erfi}(x) = -i\,\mathrm{erf}(i\,x)",
            "descripcion": "Función error imaginaria."
        })
    # Si(x) y Ci(x)
    if "si(" in expr_str:
        definiciones.append({
            "funcion": r"\mathrm{Si}(x)",
            "latex": r"\mathrm{Si}(x) = \int_{0}^{x}\frac{\sin(t)}{t}\,dt",
            "descripcion": "Seno integral."
        })
    if "ci(" in expr_str:
        definiciones.append({
            "funcion": r"\mathrm{Ci}(x)",
            "latex": r"\mathrm{Ci}(x) = \gamma+\ln|x|+\int_{0}^{x}\frac{\cos(t)-1}{t}\,dt",
            "descripcion": "Coseno integral."
        })
    # Gamma y versiones incompletas
    if "gamma(" in expr_str and "lowergamma" not in expr_str and "uppergamma" not in expr_str:
        definiciones.append({
            "funcion": r"\Gamma(z)",
            "latex": r"\Gamma(z) = \int_{0}^{\infty}t^{z-1}e^{-t}\,dt",
            "descripcion": "Función Gamma."
        })
    if "lowergamma(" in expr_str:
        definiciones.append({
            "funcion": r"\gamma(s,x)",
            "latex": r"\gamma(s,x) = \int_{0}^{x}t^{s-1}e^{-t}\,dt",
            "descripcion": "Gamma incompleta inferior."
        })
    if "uppergamma(" in expr_str:
        definiciones.append({
            "funcion": r"\Gamma(s,x)",
            "latex": r"\Gamma(s,x) = \int_{x}^{\infty}t^{s-1}e^{-t}\,dt",
            "descripcion": "Gamma incompleta superior."
        })
    # Funciones elípticas
    if "elliptic_e(" in expr_str and "elliptic_ec(" not in expr_str:
        definiciones.append({
            "funcion": r"E(x|m)",
            "latex": r"E(x|m) = \int_{0}^{x}\sqrt{1-m\sin^2(t)}\,dt",
            "descripcion": "Integral elíptica incompleta de segunda especie."
        })
    if "elliptic_f(" in expr_str:
        definiciones.append({
            "funcion": r"F(x|m)",
            "latex": r"F(x|m) = \int_{0}^{x}\frac{dt}{\sqrt{1-m\sin^2(t)}}",
            "descripcion": "Integral elíptica incompleta de primera especie."
        })
    if "elliptic_k(" in expr_str:
        definiciones.append({
            "funcion": r"K(m)",
            "latex": r"K(m) = \int_{0}^{\pi/2}\frac{dt}{\sqrt{1-m\sin^2(t)}}",
            "descripcion": "Integral elíptica completa de primera especie."
        })
    if "elliptic_pi(" in expr_str:
        definiciones.append({
            "funcion": r"\Pi(n;x|m)",
            "latex": r"\Pi(n;x|m) = \int_{0}^{x}\frac{dt}{(1-n\sin^2(t))\sqrt{1-m\sin^2(t)}}",
            "descripcion": "Integral elíptica incompleta de tercera especie."
        })
    return definiciones

#############################################################
# PRIMITIVAS ESPECIALES
#############################################################
def primitiva_sin_x2():
    from sympy import sqrt, pi, fresnels
    return (sqrt(pi)/2)*fresnels(x/sqrt(pi))

def primitiva_cos_x2():
    from sympy import sqrt, pi, fresnelc
    return (sqrt(pi)/2)*fresnelc(x/sqrt(pi))

#############################################################
# ENDPOINT /resolver-integral
#############################################################
@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        # 1) LÍMITES: Se conservan tanto la forma simbólica como la numérica.
        a_sym = sympify(datos.a)
        b_sym = sympify(datos.b)
        a_eval = float(N(a_sym))
        b_eval = float(N(b_sym))
        if a_eval >= b_eval:
            raise HTTPException(status_code=400, detail="Límite inferior debe ser menor que el superior.")

        # 2) FUNCIÓN: Parseo del string.
        f_str = datos.funcion.strip()
        f = sympify(f_str)
        f_lambda = sp.lambdify(x, f, modules=['numpy'])

        # 3) Detección de singularidades.
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
        try:
            va = f_lambda(a_eval)
            if isinstance(va, (list, np.ndarray)):
                va = va[0]
            if not es_finito(va):
                singular_points.add(a_eval)
                advertencias.append(f"⚠️ Singularidad en x={a_eval}")
        except:
            singular_points.add(a_eval)
            advertencias.append(f"⚠️ Singularidad en x={a_eval}")
        try:
            vb = f_lambda(b_eval)
            if isinstance(vb, (list, np.ndarray)):
                vb = vb[0]
            if not es_finito(vb):
                singular_points.add(b_eval)
                advertencias.append(f"⚠️ Singularidad en x={b_eval}")
        except:
            singular_points.add(b_eval)
            advertencias.append(f"⚠️ Singularidad en x={b_eval}")

        # 4) PRIMITIVA SIMBÓLICA
        if f_str == "sin(x^2)":
            F_expr = primitiva_sin_x2()
        elif f_str == "cos(x^2)":
            F_expr = primitiva_cos_x2()
        elif f_str == "sin(x)/x":
            F_expr = Si(x)
        elif f_str == "exp(-x^2)":
            F_expr = (sqrt(pi)/2)*erf(x)
        elif f_str == "sqrt(1 - x^4)":
            try:
                F_expr = sp.integrate(f, x, meijerg=True)
            except Exception as e:
                F_expr = None
                advertencias.append(f"Error al integrar sqrt(1 - x^4): {e}")
        else:
            try:
                F_expr = sp.simplify(integrate(f, x))
            except Exception as e:
                F_expr = None
                advertencias.append(f"Error al integrar simbólicamente: {e}")

        if F_expr is None:
            F_exacta_tex = "No tiene primitiva elemental conocida"
            valor_simbolico = "Valor simbólico no disponible"
        else:
            # Actualizamos el texto si en la primitiva aparece 'erf('
            F_exacta_tex = latex(F_expr)
            try:
                valor_sym = F_expr.subs(x, b_sym) - F_expr.subs(x, a_sym)
                valor_simbolico = latex(simplify(valor_sym))
                if "erf(" in str(F_expr).lower():
                    F_exacta_tex = "Primitiva en términos de erf: " + latex(F_expr)
            except Exception as e:
                valor_sym = None
                valor_simbolico = f"Error al evaluar la primitiva: {e}"

        # 5) FUNCIONES ESPECIALES DETECTADAS
        funcs_en_f = obtener_funciones_especiales(f)
        funcs_en_F = obtener_funciones_especiales(F_expr) if F_expr else []
        all_funcs = {}
        for d in (funcs_en_f + funcs_en_F):
            all_funcs[d["funcion"]] = d
        funciones_especiales_detectadas = list(all_funcs.values())

        # 6) INTEGRAL DEFINIDA EXACTA Y VALOR NUMÉRICO
        try:
            resultado_sympy_def = integrate(f, (x, a_sym, b_sym))
            resultado_exacto_tex = latex(resultado_sympy_def)
            try:
                resultado_exacto_val = float(N(resultado_sympy_def))
            except Exception as e:
                resultado_exacto_val = None
                advertencias.append(f"⚠️ Error numérico en la integral definida: {e}")
        except Exception as e:
            resultado_exacto_val = None
            resultado_exacto_tex = f"Error en la integral definida: {e}"

        # 7) SERIE DE TAYLOR
        from sympy import binomial  # Duplicado pero lo dejamos por claridad.
        if f_str == "exp(-x^2)":
            serie_infinita = r"e^{-x^2} = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{n!}"
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = "La serie de Taylor de \\(e^{-x^2}\\) alrededor de x=0 es:"
            terminos = []
            for n_ in range(datos.n_terminos):
                term_expr = (-1)**n_ * x**(2*n_) / factorial(n_)
                terminos.append(term_expr)
            f_series_expr = sum(terminos)
            serie_latex_terms = [latex(t) for t in terminos]
            f_series_sumada = " + ".join(serie_latex_terms) + r" + \cdots"
            f_series_tex = f"$$ e^{-x^2} = {f_series_sumada} $$"
        elif f_str == "sqrt(1 - x^4)":
            serie_infinita = (r"\sqrt{1 - x^4} = \sum_{k=0}^{\infty} \binom{1/2}{k} (-1)^k x^{4k}")
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = ("La serie de Taylor de \\(\\sqrt{1 - x^4}\\) se obtiene usando el binomio "
                                  "\\((1-u)^{1/2}\\) con u = x^4.")
            terminos = []
            for k in range(datos.n_terminos):
                coef = binomial(Rational(1,2), k)*(-1)**k
                term_expr = coef * x**(4*k)
                terminos.append(term_expr)
            f_series_expr = sum(terminos)
            serie_latex_terms = [latex(t) for t in terminos]
            f_series_sumada = " + ".join(serie_latex_terms) + r" + \cdots"
            f_series_tex = f"$$ \sqrt{{1 - x^4}} = {f_series_sumada} $$"
        elif f_str == "sin(x^2)":
            serie_infinita = r"\sin(x^2) = \sum_{k=0}^{\infty} \frac{(-1)^k x^{4k+2}}{(2k+1)!}"
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = "La serie de Taylor de \\(\\sin(x^2)\\) alrededor de x=0 es:"
            terminos = []
            for k in range(datos.n_terminos):
                coef = (-1)**k
                potencia = 4*k + 2
                term_expr = coef * x**potencia / factorial(2*k+1)
                terminos.append(term_expr)
            f_series_expr = sum(terminos)
            serie_latex_terms = [latex(t) for t in terminos]
            f_series_sumada = " + ".join(serie_latex_terms) + r" + \cdots"
            f_series_tex = f"$$ \\sin(x^2) = {f_series_sumada} $$"
        elif f_str == "cos(x^2)":
            serie_infinita = r"\cos(x^2) = \sum_{k=0}^{\infty} \frac{(-1)^k x^{4k}}{(2k)!}"
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = "La serie de Taylor de \\(\\cos(x^2)\\) alrededor de x=0 es:"
            terminos = []
            for k in range(datos.n_terminos):
                coef = (-1)**k
                potencia = 4*k
                term_expr = coef * x**potencia / factorial(2*k)
                terminos.append(term_expr)
            f_series_expr = sum(terminos)
            serie_latex_terms = [latex(t) for t in terminos]
            f_series_sumada = " + ".join(serie_latex_terms) + r" + \cdots"
            f_series_tex = f"$$ \\cos(x^2) = {f_series_sumada} $$"
        elif f_str == "sin(x)/x":
            serie_infinita = r"\frac{\sin(x)}{x} = \sum_{k=0}^{\infty} (-1)^k \frac{x^{2k}}{(2k+1)!}"
            sumatoria_general_tex = f"$$ {serie_infinita} $$"
            explicacion_taylor = ("La serie de Taylor de \\(\\frac{\sin(x)}{x}\\) alrededor de x=0 es:")
            terminos = []
            for k in range(datos.n_terminos):
                term_expr = (-1)**k * x**(2*k) / factorial(2*k+1)
                terminos.append(term_expr)
            f_series_expr = sum(terminos)
            serie_latex_terms = [latex(t) for t in terminos]
            f_series_sumada = " + ".join(serie_latex_terms) + r" + \cdots"
            f_series_tex = f"$$ \\frac{{\sin(x)}}{{x}} = {f_series_sumada} $$"
        else:
            a_taylor = 0
            serie_general_expr = Sum(diff(f, (x, n)).subs(x, a_taylor)/factorial(n)*(x-a_taylor)**n, (n, 0, oo))
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
        try:
            F_aproximada = integrate(f_series_expr, x)
            F_aproximada_tex = f"$$ {latex(F_aproximada)} $$"
        except Exception as e:
            F_aproximada_tex = f"No se pudo calcular la integral de la serie truncada: {e}"

        # 8) Integral definida en notación LaTeX
        integral_definida_tex = f"$$ \\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\, dx $$"

        # 9) Métodos numéricos
        subintervalos = crear_subintervalos(a_eval, b_eval, singular_points)
        integral_simpson, n_points_simpson = simpson_subintervalos(f_lambda, subintervalos)
        pts_sing = sorted(list(singular_points))
        try:
            val_romberg, _ = quad(f_lambda, a_eval, b_eval, points=pts_sing)
            integral_romberg = val_romberg
        except Exception as e:
            integral_romberg = None
            advertencias.append(f"Romberg: {e}")
        try:
            val_gauss, _ = quad(f_lambda, a_eval, b_eval, points=pts_sing)
            integral_gauss = val_gauss
        except Exception as e:
            integral_gauss = None
            advertencias.append(f"Gauss: {e}")
        integral_montecarlo = monte_carlo_subintervalos(f_lambda, subintervalos)

        # 10) Texto para GeoGebra: Se genera un string para copiar y pegar.
        texto_geogebra = (
            f"Función: f(x) = {exportar_para_geogebra(f)}\n"
            f"Taylor truncada: T(x) = {exportar_para_geogebra(f_series_expr)}\n"
            f"Área: Integral(f, {str(a_sym)}, {str(b_sym)})"
        )

        # 11) Devolver JSON
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
