from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from sympy import (
    symbols, sympify, integrate, sqrt, pi, erf, latex, simplify,
    factorial, Function, Derivative, Add, Mul, Pow
)
from sympy.printing.pycode import pycode
from scipy.integrate import quad
import numpy as np
import random

app = FastAPI()
x = symbols('x')

class InputDatos(BaseModel):
    funcion: str
    a: Union[str, float]
    b: Union[str, float]
    n_terminos: int = 7


##############################################################################
# 1) DICCIONARIO DE SERIES CONOCIDAS
#    Si la función introducida coincide exactamente con una de las claves,
#    se utilizará la serie infinita específica aquí.
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
#    Se agregan definiciones para "erf", "li", "Gamma", "2F1", "1F2", etc.
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
        "definition": r"$$ {}_{2}F_{1}(a,b;c;z) = \sum_{n=0}^{\infty} \frac{(a)_n (b)_n}{(c)_n} \frac{z^n}{n!} $$",
        "explanation": "La función hipergeométrica {}_{2}F_{1} es una de las más generales en el análisis matemático."
    },
    "1F2": {
        "definition": r"$$ {}_{1}F_{2}(a;b,c;z) = \sum_{n=0}^{\infty} \frac{(a)_n}{(b)_n (c)_n} \frac{z^n}{n!} $$",
        "explanation": "La función hipergeométrica generalizada {}_{1}F_{2} se utiliza en múltiples contextos de análisis matemático."
    }
}


def expr_to_geogebra(expr):
    """
    Convierte una expresión Sympy en un string adecuado para GeoGebra,
    colocando paréntesis de forma que no se confundan los exponentes.
    """
    if expr.is_Add:
        # Suma de términos, unimos con " + " (respetar signos si hay negativos)
        # Observación: si un término es negativo, Sympy lo manejará como Add(..., -Term)
        # Podemos filtrar los sign con str() si se desea. Por simplicidad, unimos con " + " y el str() mostrará el signo.
        return " + ".join(expr_to_geogebra(arg) for arg in expr.args)
    elif expr.is_Mul:
        # Producto de factores
        # Ej: 2*x -> "2*(x)"
        # Ej: -1*x^2 -> "(-1)*(x^2)"
        # Observa que multiplicación puede incluir signos.
        factors = expr.as_ordered_factors()
        return "*".join(f"({expr_to_geogebra(fac)})" for fac in factors)
    elif expr.is_Pow:
        # Potencia base^exp
        base, exponent = expr.as_base_exp()
        return f"({expr_to_geogebra(base)})^({expr_to_geogebra(exponent)})"
    else:
        # Números, símbolos, etc.
        return str(expr)


@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        ###################################################################
        # A) PROCESAMIENTO INICIAL
        ###################################################################
        
        # 1) Procesar la función introducida
        f_str_usuario = datos.funcion.strip()
        f_str = f_str_usuario.replace('^', '**')
        f = sympify(f_str)
        f_lambda = lambda val: float(f.subs(x, val))
        
        # 2) Procesar los límites de integración
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
        # B) PRIMITIVA E INTEGRAL DEFINIDA
        ###################################################################
        
        # 3) Calcular la primitiva (antiderivada)
        primitiva = integrate(f, x)
        primitiva_latex = latex(primitiva)
        
        # 3.1) Detectar y explicar funciones especiales en la primitiva
        definicion_especial = ""
        detection_map = {
            "Gamma":  r"\Gamma",
            "erf":    r"erf",
            "li":     r"li",
            "2F1":    r"{ }_{2}F_{1}",
            "1F2":    r"{ }_{1}F_{2}"
        }
        
        for name, info in special_functions.items():
            if name in detection_map:
                latex_key = detection_map[name]
            else:
                latex_key = name
            if latex_key in primitiva_latex:
                definicion_especial += (
                    f"\n\n#### Definición de la función especial \\({name}\\):\n"
                    f"{info['definition']}\n"
                    f"{info['explanation']}\n"
                )
        
        # 4) Valor simbólico de la integral definida
        try:
            valor_simb = simplify(primitiva.subs(x, b_sym) - primitiva.subs(x, a_sym))
            valor_simbolico = latex(valor_simb)
        except Exception as e:
            valor_simbolico = f"Error al evaluar simbólicamente: {e}"
        
        # 5) Valor numérico exacto
        try:
            valor_num_expr = integrate(f, (x, a_sym, b_sym))
            valor_numerico = float(valor_num_expr.evalf())
            valor_numerico_latex = f"$${valor_numerico}$$"
        except Exception as e:
            valor_numerico = None
            valor_numerico_latex = "$$ Error en la integral definida: Cannot convert expression to float $$"
        
        # 6) Integral definida (LaTeX)
        integral_definida = f"$$ \\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\, dx $$"
        
        ###################################################################
        # C) SERIES DE TAYLOR
        ###################################################################
        
        # 7) Serie de Taylor "general"
        if f_str in known_infinite_series:
            serie_taylor_general = known_infinite_series[f_str]
            explicacion_taylor_general = (
                f"La función {f_str_usuario} es reconocida, y su serie de Taylor alrededor de x=0 es conocida."
            )
        else:
            serie_taylor_general = (
                r"$$ f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!} \, x^n $$"
            )
            explicacion_taylor_general = (
                "No se ha encontrado una serie conocida para esta función, "
                "por lo que se muestra la fórmula general derivativa."
            )
        
        # 8) Serie de Taylor finita (truncada) en términos de factoriales
        serie_sympy = f.series(x, 0, datos.n_terminos)
        serie_truncada_sympy = serie_sympy.removeO()
        serie_taylor_finita = f"$$ {latex(serie_truncada_sympy)} $$"
        
        ###################################################################
        # D) INTEGRAL DE LA SERIE TRUNCADA
        ###################################################################
        
        try:
            integral_serie = integrate(serie_truncada_sympy, x)
            integral_serie_taylor = f"$$ {latex(integral_serie)} $$"
        except Exception as e:
            integral_serie_taylor = f"Error al integrar la serie: {e}"
        
        ###################################################################
        # E) MÉTODOS NUMÉRICOS
        ###################################################################
        
        puntos = np.linspace(a_eval, b_eval, 1001)
        try:
            valores = np.array([f_lambda(v) for v in puntos])
            simpson_val = np.trapz(valores, puntos)
        except Exception as e:
            simpson_val = None
        
        try:
            romberg_val, _ = quad(f_lambda, a_eval, b_eval)
        except Exception as e:
            romberg_val = None
        
        try:
            cuadratura_gaussiana = romberg_val
        except Exception as e:
            cuadratura_gaussiana = None
        
        try:
            monte_carlo_val = np.mean([
                f_lambda(random.uniform(a_eval, b_eval))
                for _ in range(10000)
            ]) * (b_eval - a_eval)
        except Exception as e:
            monte_carlo_val = None
        
        metodos = {
            "simpson": {
                "value": simpson_val,
                "n_points": 1001
            },
            "romberg": romberg_val,
            "cuadratura_gaussiana": cuadratura_gaussiana,
            "montecarlo": monte_carlo_val
        }
        
        ###################################################################
        # F) TEXTO PARA GEOGEBRA
        ###################################################################
        # Usamos la función expr_to_geogebra para evitar confusión de exponentes.
        geo_f_str = f_str_usuario.replace('**', '^')
        geo_series = expr_to_geogebra(serie_truncada_sympy)
        geo_a = expr_to_geogebra(a_sym)
        geo_b = expr_to_geogebra(b_sym)
        
        texto_geogebra = (
            f"Función: f(x) = {geo_f_str}\n"
            f"Taylor truncada: T(x) = {geo_series}\n"
            f"Área: Integral(f, {geo_a}, {geo_b})"
        )
        
        ###################################################################
        # G) JSON DE RESPUESTA
        ###################################################################
        
        resultado = {
            "funcion_introducida": f_str,
            "primitiva_real": f"$$ {primitiva_latex} $$",
            "definicion_funciones_especiales": definicion_especial,
            "valor_simbolico_integral": f"$$ {valor_simbolico} $$",
            "valor_numerico_exacto": valor_numerico,
            "valor_numerico_exacto_latex": valor_numerico_latex,
            "integral_definida": integral_definida,
            
            # Serie de Taylor (infinita) y su explicación
            "serie_taylor_general": serie_taylor_general,
            "explicacion_taylor_general": explicacion_taylor_general,
            
            # Serie de Taylor finita (la truncada)
            "serie_taylor_finita": serie_taylor_finita,
            "integral_serie_taylor": integral_serie_taylor,
            
            "metodos_numericos": metodos,
            "advertencias": [],
            "texto_geogebra": texto_geogebra
        }
        
        return resultado

    except Exception as e:
        return {"error": str(e)}
