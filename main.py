from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from sympy import (
    symbols, sympify, integrate, series, lambdify, N,
    solveset, Interval, oo, log, sin, cos, exp, diff, factorial,
    limit, Sum, S
)
from scipy.integrate import simpson, quad
import random

app = FastAPI()

# Declaración simbólica de variables
x, n = symbols('x n')

# Modelo de entrada
class InputDatos(BaseModel):
    funcion: str
    a: float
    b: float
    n_terminos: int = Field(default=10, ge=1, le=20)
    tolerancia: float = Field(default=1e-6, ge=1e-10)

    @classmethod
    def validate(cls, value):
        if value.a >= value.b:
            raise ValueError("El límite inferior debe ser menor que el límite superior.")
        return value

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        try:
            f = sympify(datos.funcion)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Función matemática no válida: {e}")

        if datos.a >= datos.b:
            raise HTTPException(status_code=400, detail="El límite inferior debe ser menor que el límite superior.")

        if str(f) == 'sin(x)/x':
            def f_lambda(x_val):
                return np.where(x_val == 0, 1.0, np.sin(x_val)/x_val)
        else:
            f_lambda = lambdify(x, f, modules=['numpy'])

        posibles_sing = solveset(1/f, x, domain=Interval(datos.a, datos.b))
        advertencias = []
        for p in posibles_sing:
            try:
                val = float(p.evalf())
                if datos.a < val < datos.b:
                    advertencias.append("⚠️ La función tiene una posible singularidad dentro del intervalo.")
            except:
                continue

        try:
            F_exacta = integrate(f, x)
            F_exacta_tex = f"$$ {F_exacta} $$"
        except:
            F_exacta = "No tiene primitiva elemental"
            F_exacta_tex = "No tiene primitiva elemental"

        if str(datos.a) == "inf" or str(datos.b) == "inf":
            resultado_exacto = limit(integrate(f, (x, datos.a, datos.b)), x, oo)
            if resultado_exacto in [oo, -oo]:
                raise HTTPException(status_code=400, detail="La integral tiene un valor infinito.")
            resultado_exacto_val = float(N(resultado_exacto))
            resultado_exacto_tex = f"$$ {resultado_exacto} $$"
        else:
            resultado_exacto = integrate(f, (x, datos.a, datos.b))
            resultado_exacto_val = float(N(resultado_exacto))
            resultado_exacto_tex = f"$$ {resultado_exacto} $$"

        # Serie de Taylor general
        sumatoria_general = Sum(
            diff(f, x, n).subs(x, datos.a) / factorial(n) * (x - datos.a)**n,
            (n, 0, oo)
        ).doit()

        # Serie de Taylor truncada hasta n_terminos
        f_series = sum([
            diff(f, x, i).subs(x, datos.a) / factorial(i) * (x - datos.a)**i
            for i in range(datos.n_terminos)
        ])
        F_aproximada = integrate(f_series, x)
        F_aproximada_tex = f"$$ {F_aproximada} $$"

        integral_definida_tex = f"$$ \\int_{{{datos.a}}}^{{{datos.b}}} {f} \\, dx $$"

        puntos = np.linspace(datos.a, datos.b, 1000)
        y_vals = f_lambda(puntos)
        dx = (datos.b - datos.a) / 1000

        integral_simpson = simpson(y_vals, dx=dx)
        integral_romberg, _ = quad(f_lambda, datos.a, datos.b)
        integral_gauss, _ = quad(f_lambda, datos.a, datos.b)

        def monte_carlo_integration(f, a, b, n_samples=10000):
            total = 0
            for _ in range(n_samples):
                x_rand = random.uniform(a, b)
                total += f(x_rand)
            return (b - a) * total / n_samples

        integral_montecarlo = monte_carlo_integration(
            lambda x_val: f_lambda(np.array([x_val]))[0],
            datos.a, datos.b
        )

        return {
            "primitiva_real": F_exacta_tex,
            "serie_taylor_general": f"$$ {sumatoria_general} $$",
            "serie_taylor_finita": f"$$ {f_series} $$",
            "integral_definida_exacta": integral_definida_tex,
            "integral_definida_valor": resultado_exacto_tex,
            "valor_numerico_exacto": resultado_exacto_val,
            "metodos_numericos": {
                "simpson": integral_simpson,
                "romberg": integral_romberg,
                "cuadratura_gaussiana": integral_gauss,
                "montecarlo": integral_montecarlo,
            },
            "advertencias": advertencias
        }

    except Exception as e:
        return {"error": str(e)}

# Funciones especiales detectables
def obtener_definiciones_especiales(expr):
    definiciones = []

    if "Si" in str(expr):
        definiciones.append({
            "funcion": "Si(x)",
            "latex": r"\mathrm{Si}(x) = \int_0^x \frac{\sin(t)}{t} \, dt",
            "descripcion": "La función seno integral aparece como primitiva de sin(x)/x. No tiene forma elemental, pero está perfectamente definida mediante una integral."
        })

    if "Li" in str(expr):
        definiciones.append({
            "funcion": "Li(x)",
            "latex": r"\mathrm{Li}(x) = \int_0^x \frac{dt}{\log(t)}",
            "descripcion": "La función logaritmo integral aparece al calcular la primitiva de 1/log(x). Es una función especial importante en teoría de números."
        })

    if "erf" in str(expr):
        definiciones.append({
            "funcion": "erf(x)",
            "latex": r"\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt",
            "descripcion": "La función error aparece como primitiva de exp(-x²). Es clave en estadísticas y distribución normal."
        })

    if "gamma" in str(expr):
        definiciones.append({
            "funcion": "Gamma(x)",
            "latex": r"\Gamma(x) = \int_0^\infty t^{x-1} e^{-t} \, dt",
            "descripcion": "La función Gamma generaliza el factorial a números reales y complejos."
        })

    if "beta" in str(expr):
        definiciones.append({
            "funcion": "Beta(x, y)",
            "latex": r"B(x, y) = \int_0^1 t^{x-1} (1 - t)^{y-1} \, dt",
            "descripcion": "La función Beta está relacionada con la función Gamma y aparece en el cálculo de probabilidades."
        })

    if "zeta" in str(expr):
        definiciones.append({
            "funcion": "Zeta(s)",
            "latex": r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}",
            "descripcion": "La función Zeta de Riemann es clave en la teoría de números, especialmente en la distribución de los números primos."
        })

    if "besselj" in str(expr):
        definiciones.append({
            "funcion": "J_n(x)",
            "latex": r"J_n(x) = \frac{1}{\pi} \int_0^\pi \cos(n \theta - x \sin \theta) \, d\theta",
            "descripcion": "Las funciones de Bessel son soluciones de la ecuación diferencial de Bessel, usada en problemas de física."
        })

    return definiciones
