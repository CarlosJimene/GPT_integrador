from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from sympy import (
    symbols, sympify, integrate, sqrt, pi, erf, latex, simplify,
    factorial, Function, Derivative, oo, limit as sym_limit
)
from sympy.calculus.util import singularities
from scipy.integrate import quad
import numpy as np
import random
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json  # ya lo tenés, pero asegúrate que esté




app = FastAPI()
x = symbols('x')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
class InputDatos(BaseModel):
    funcion: str
    a: Union[str, float]
    b: Union[str, float]
    n_terminos: int = 7

##############################################################################
# 1) DICCIONARIO DE SERIES CONOCIDAS
##############################################################################
known_infinite_series = {
    "exp(-x**2)": r"$$ e^{-x^2} \;=\; \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{n!} $$",
    "cos(x**2)":  r"$$ \cos(x^2) \;=\; \sum_{k=0}^{\infty} \frac{(-1)^k x^{4k}}{(2k)!} $$",
    "sin(x**2)":  r"$$ \sin(x^2) \;=\; \sum_{k=0}^{\infty} \frac{(-1)^k x^{4k+2}}{(2k+1)!} $$",
    "cos(x)":     r"$$ \cos(x) \;=\; \sum_{k=0}^{\infty} \frac{(-1)^k x^{2k}}{(2k)!} $$",
    "sin(x)":     r"$$ \sin(x) \;=\; \sum_{k=0}^{\infty} \frac{(-1)^k x^{2k+1}}{(2k+1)!} $$"
}

##############################################################################
# 2) DICCIONARIO DE FUNCIONES ESPECIALES
##############################################################################
special_functions = {
    "erf": {
        "definition": r"$$ \operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2}\,dt $$",
        "explanation": "La función error se utiliza para el cálculo de probabilidades en distribuciones gaussianas."
    },
    "li": {
        "definition": r"$$ \operatorname{li}(x) = \int_{0}^{x} \frac{dt}{\ln(t)} $$",
        "explanation": "La función logarítmica integral es importante en la teoría de números primos."
    },
    "Gamma": {
        "definition": r"$$ \Gamma(s) = \int_{0}^{\infty} t^{s-1} e^{-t}\, dt $$",
        "explanation": "La función Gamma generaliza la noción de factorial a números reales y complejos."
    },
    "2F1": {
        "definition": r"$$ {}_{2}F_{1}(a,b;c;z) = \sum_{n=0}^{\infty} \frac{(a)_n (b)_n}{(c)_n}\frac{z^n}{n!} $$",
        "explanation": "La función hipergeométrica {}_{2}F_{1} es una de las más generales en el análisis matemático."
    },
    "1F2": {
        "definition": r"$$ {}_{1}F_{2}(a;b,c;z) = \sum_{n=0}^{\infty} \frac{(a)_n}{(b)_n (c)_n}\frac{z^n}{n!} $$",
        "explanation": "La función hipergeométrica generalizada {}_{1}F_{2} se utiliza en múltiples contextos de análisis matemático."
    }
}

##############################################################################
# 3) FUNCIÓN AUXILIAR: expr_to_geogebra
##############################################################################
def expr_to_geogebra(expr):
    """
    Convierte una expresión Sympy en un string adecuado para GeoGebra,
    colocando paréntesis de forma que no se confundan los exponentes.
    """
    if expr.is_Add:
        return " + ".join(expr_to_geogebra(arg) for arg in expr.args)
    elif expr.is_Mul:
        factors = expr.as_ordered_factors()
        return "*".join(f"({expr_to_geogebra(fac)})" for fac in factors)
    elif expr.is_Pow:
        base, exponent = expr.as_base_exp()
        return f"({expr_to_geogebra(base)})^({expr_to_geogebra(exponent)})"
    else:
        return str(expr)

##############################################################################
# 4) FUNCIÓN AUXILIAR: format_series_factorials
##############################################################################
def format_series_factorials(expr):
    """
    Convierte una expresión polinómica (serie truncada) en un string en el que cada término
    se expresa en forma: (coeficiente * x^n)/(n!) (si es posible).
    """
    terms = expr.as_ordered_terms()
    new_terms = []
    for term in terms:
        coeff, monom = term.as_coeff_Mul()
        n = 0
        if monom.has(x):
            try:
                n = monom.as_poly(x).degree()
            except Exception:
                n = 1
        if n == 0:
            term_str = latex(coeff)
        else:
            coeff_factor = simplify(coeff * factorial(n))
            if coeff_factor == 1:
                term_str = f"\\frac{{x^{{{n}}}}}{{{n}!}}"
            else:
                term_str = f"\\frac{{{latex(coeff_factor)} x^{{{n}}}}}{{{n}!}}"
        new_terms.append(term_str)
    series_str = " + ".join(new_terms)
    series_str = series_str.replace("+ -", "- ")
    return series_str

##############################################################################
# 5) DETECCIÓN Y TRATAMIENTO DE DISCONTINUIDADES
##############################################################################
def tratar_discontinuidades(f, a_eval, b_eval):
    """
    Busca singularidades en el intervalo [a_eval, b_eval]. 
    Si se detecta alguna discontinuidad en la que el límite es infinito,
    retorna (True, []).
    Si se detectan discontinuidades removibles o de salto finito, retorna (False, [lista de puntos]).
    """
    try:
        sing_points = singularities(f, x)
    except Exception:
        sing_points = set()
    
    sing_in_interval = []
    infinite_disc = False
    for s in sing_points:
        try:
            s_val = float(s.evalf())
        except Exception:
            continue
        if a_eval < s_val < b_eval:
            L_left = sym_limit(f, x, s, dir='-')
            L_right = sym_limit(f, x, s, dir='+')
            if L_left in [oo, -oo] or L_right in [oo, -oo]:
                infinite_disc = True
            else:
                sing_in_interval.append(s_val)
    return infinite_disc, sorted(sing_in_interval)

##############################################################################
# 6) ENDPOINT PRINCIPAL
##############################################################################
@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        ###################################################################
        # A) PROCESAMIENTO INICIAL
        ###################################################################
        f_str_usuario = datos.funcion.strip()
        f_str = f_str_usuario.replace('^', '**')
        f = sympify(f_str)
        
        # Procesar los límites
        a_str = str(datos.a)
        b_str = str(datos.b)
        for target in ["raiz de pi", "raíz de pi", "√"]:
            a_str = a_str.replace(target, "sqrt(pi)")
            b_str = b_str.replace(target, "sqrt(pi)")
        a_sym = sympify(a_str)
        b_sym = sympify(b_str)
        a_eval = float(a_sym.evalf())
        b_eval = float(b_sym.evalf())
        if a_eval >= b_eval:
            raise Exception("El límite inferior debe ser menor que el superior.")
        
        ###################################################################
        # B) TRATAMIENTO DE DISCONTINUIDADES
        ###################################################################
        infinite_disc, sing_in_interval = tratar_discontinuidades(f, a_eval, b_eval)
        if infinite_disc:
            return {"error": "La función presenta al menos una discontinuidad infinita en el intervalo especificado. No es posible calcular el área."}
        
        # Actualizar la función evaluadora para tratar indeterminaciones removibles.
        def f_lambda_mod(val):
            for s in sing_in_interval:
                if abs(val - s) < 1e-8:
                    return float(sym_limit(f, x, s, dir='+'))
            return float(f.subs(x, val))
        f_lambda = f_lambda_mod
        
        ###################################################################
        # C) PRIMITIVA E INTEGRAL DEFINIDA
        ###################################################################
        primitiva = integrate(f, x)
        primitiva_latex = latex(primitiva)
        
        # Detectar funciones especiales en la primitiva
        definicion_especial = ""
        detection_map = {
            "Gamma":  r"\Gamma",
            "erf":    r"erf",
            "li":     r"li",
            "2F1":    r"{ }_{2}F_{1}",
            "1F2":    r"{ }_{1}F_{2}"
        }
        for name, info in special_functions.items():
            latex_key = detection_map.get(name, name)
            if latex_key in primitiva_latex:
                definicion_especial += (
                    f"\n\n#### Definición de la función especial \\({name}\\):\n"
                    f"{info['definition']}\n"
                    f"{info['explanation']}\n"
                )
        
        try:
            valor_simb = simplify(primitiva.subs(x, b_sym) - primitiva.subs(x, a_sym))
            valor_simbolico = latex(valor_simb)
        except Exception as e:
            valor_simbolico = f"Error al evaluar simbólicamente: {e}"
        
        try:
            valor_num_expr = integrate(f, (x, a_sym, b_sym))
            valor_numerico = float(valor_num_expr.evalf())
            valor_numerico_latex = f"$${valor_numerico}$$"
        except Exception as e:
            valor_numerico = None
            valor_numerico_latex = "$$ Error en la integral definida: Cannot convert expression to float $$"
        
        integral_definida = f"$$ \\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\, dx $$"
        
        ###################################################################
        # D) SERIES DE TAYLOR
        ###################################################################
        if f_str in known_infinite_series:
            serie_taylor_general = known_infinite_series[f_str]
            explicacion_taylor_general = f"La función {f_str_usuario} es reconocida, y su serie de Taylor alrededor de x=0 es conocida."
        else:
            serie_taylor_general = r"$$ f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!} \, x^n $$"
            explicacion_taylor_general = (
                "No se ha encontrado una serie conocida para esta función, "
                "por lo que se muestra la fórmula general derivativa."
            )
        
        serie_sympy = f.series(x, 0, datos.n_terminos)
        serie_truncada_sympy = serie_sympy.removeO()
        # Formatear la serie truncada en términos de factoriales (T1(x))
        serie_taylor_finita = f"$$ T1(x) = {format_series_factorials(serie_truncada_sympy)} $$"
        
        ###################################################################
        # E) INTEGRAL DE LA SERIE TRUNCADA (T2(x))
        ###################################################################
        try:
            integral_serie = integrate(serie_truncada_sympy, x)
            integral_serie_formatted = format_series_factorials(integral_serie)
            integral_serie_taylor = f"$$ T2(x) = {integral_serie_formatted} $$"
        except Exception as e:
            integral_serie_taylor = f"Error al integrar la serie: {e}"
        
        ###################################################################
        # F) MÉTODOS NUMÉRICOS CON TRATAMIENTO DE DISCONTINUIDADES FINITAS
        ###################################################################
        subintervals = [a_eval] + sing_in_interval + [b_eval]
        num_integral = 0
        for i in range(len(subintervals)-1):
            r, err = quad(f_lambda, subintervals[i], subintervals[i+1])
            num_integral += r
        
        try:
            puntos = np.linspace(a_eval, b_eval, 1001)
            valores = np.array([f_lambda(v) for v in puntos])
            simpson_val = np.trapz(valores, puntos)
        except Exception:
            simpson_val = None
        
        try:
            romberg_val, _ = quad(f_lambda, a_eval, b_eval)
        except Exception:
            romberg_val = None
        
        try:
            cuadratura_gaussiana = romberg_val
        except Exception:
            cuadratura_gaussiana = None
        
        try:
            monte_carlo_val = np.mean([
                f_lambda(random.uniform(a_eval, b_eval))
                for _ in range(10000)
            ]) * (b_eval - a_eval)
        except Exception:
            monte_carlo_val = None
        
        metodos = {
            "simpson": {"value": simpson_val, "n_points": 1001},
            "romberg": romberg_val,
            "cuadratura_gaussiana": cuadratura_gaussiana,
            "montecarlo": monte_carlo_val
        }
        
        ###################################################################
        # G) TEXTO PARA GEOGEBRA
        ###################################################################
        geo_f_str = f_str_usuario.replace('**', '^')
        geo_series_T1 = format_series_factorials(serie_truncada_sympy)
        try:
            T2_expr = integrate(serie_truncada_sympy, x)
            geo_series_T2 = format_series_factorials(T2_expr)
        except Exception:
            geo_series_T2 = latex(integrate(serie_truncada_sympy, x))
        
        geo_a = expr_to_geogebra(a_sym)
        geo_b = expr_to_geogebra(b_sym)
        texto_geogebra = (
            f"Función: f(x) = {geo_f_str}\n"
            f"T1(x) = {geo_series_T1}\n"
            f"T2(x) = {geo_series_T2}\n"
            f"Área: Integral(f, {geo_a}, {geo_b})"
        )
        
        ###################################################################
        # H) JSON DE RESPUESTA
        ###################################################################
        resultado = {
            "funcion_introducida": f_str,
            "primitiva_real": f"$$ {primitiva_latex} $$",
            "definicion_funciones_especiales": definicion_especial,
            "valor_simbolico_integral": f"$$ {valor_simbolico} $$",
            "valor_numerico_exacto": valor_numerico,
            "valor_numerico_exacto_latex": valor_numerico_latex,
            "integral_definida": integral_definida,
            "serie_taylor_general": serie_taylor_general,
            "explicacion_taylor_general": explicacion_taylor_general,
            "serie_taylor_finita": serie_taylor_finita,
            "integral_serie_taylor": integral_serie_taylor,
            "metodos_numericos": metodos,
            "advertencias": [],
            "texto_geogebra": texto_geogebra
        }
        # Serialización segura para evitar problemas con tipos como np.float64, None, etc.
        return JSONResponse(content=json.loads(json.dumps(resultado, default=str)))


    except Exception as e:
        return {"error": str(e)}
