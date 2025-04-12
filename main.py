from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sympy import symbols, sympify, integrate, series, lambdify, N, solveset, Interval, S, oo, log, sin, cos, exp, diff, factorial, Sum
from scipy.integrate import simpson, quad

app = FastAPI()
x = symbols('x')

# Número máximo de términos para evitar errores con series demasiado largas
MAX_TERMINOS = 10

class InputDatos(BaseModel):
    funcion: str
    a: float
    b: float
    n_terminos: int

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        f = sympify(datos.funcion)

        # Detectar sin(x)/x manualmente
        if str(f) == 'sin(x)/x':
            def f_lambda(x_val):
                return np.where(x_val == 0, 1.0, np.sin(x_val)/x_val)
        else:
            f_lambda = lambdify(x, f, modules=['numpy'])

        # Detectar posibles singularidades (división por cero, log, etc.)
        posibles_sing = solveset(1/f, x, domain=Interval(datos.a, datos.b))
        advertencias = []
        if any([p.is_real and datos.a < float(p.evalf()) < datos.b for p in posibles_sing]):
            advertencias.append("⚠️ La función tiene una posible singularidad dentro del intervalo de integración. El resultado numérico podría ser poco preciso.")

        # Primitiva exacta
        try:
            F_exacta = integrate(f, x)
            F_exacta_tex = f"$$ {F_exacta} $$"
        except:
            F_exacta = "No tiene primitiva elemental"
            F_exacta_tex = "No tiene primitiva elemental"

        # Calcular la serie de Taylor hasta el número de términos permitido
        n_terminos = min(datos.n_terminos, MAX_TERMINOS)
        f_series = series(f, x, datos.a, n_terminos + 1).removeO()  # Serie de Taylor hasta n términos

        # Sumar los términos hasta el orden n-1
        suma_terminos = ' + '.join([
            f"{diff(f, x, n).subs(x, datos.a) / factorial(n)} * (x - {datos.a})^{n}"
            for n in range(n_terminos)
        ])

        # Formato de la serie de Taylor infinita
        sumatoria_general = r"$$ \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n $$"  # Sumatoria general en LaTeX

        # Integral definida exacta
        resultado_exacto = integrate(f, (x, datos.a, datos.b))
        resultado_exacto_val = float(N(resultado_exacto))

        # Métodos numéricos
        puntos = np.linspace(datos.a, datos.b, 1000)
        y_vals = np.array([f_lambda(xi) for xi in puntos])
        dx = (datos.b - datos.a) / 1000

        integral_simpson = simpson(y_vals, dx=dx)
        integral_romberg, _ = quad(f_lambda, datos.a, datos.b)
        integral_gauss, _ = quad(f_lambda, datos.a, datos.b)

        return {
            "primitiva_real": F_exacta_tex,
            "serie_taylor_general": sumatoria_general,  # Sumatoria infinita
            "serie_taylor_finita": f"$$ {suma_terminos} $$",  # Suma de los primeros términos
            "integral_definida_exacta": f"$$ \\int_{{{datos.a}}}^{{{datos.b}}} {str(f)} dx $$",
            "valor_numerico_exacto": resultado_exacto_val,
            "metodos_numericos": {
                "simpson": integral_simpson,
                "romberg": integral_romberg,
                "cuadratura_gaussiana": integral_gauss
            },
            "funciones_especiales": obtener_definiciones_especiales(F_exacta),
            "advertencias": advertencias
        }
    except Exception as e:
        return {"error": str(e)}

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
    return definiciones
