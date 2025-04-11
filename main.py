from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sympy import symbols, sympify, integrate, series, lambdify, N, latex
from scipy.integrate import simpson, quad

app = FastAPI()
x = symbols('x')

class InputDatos(BaseModel):
    funcion: str
    a: float
    b: float
    n_terminos: int

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

@app.post("/resolver-integral")
def resolver_integral(datos: InputDatos):
    try:
        f = sympify(datos.funcion)

        if str(f) == 'sin(x)/x':
            def f_lambda(x_val):
                return np.where(x_val == 0, 1.0, np.sin(x_val)/x_val)
        else:
            f_lambda = lambdify(x, f, modules=['numpy'])

        F_exacta = integrate(f, x)
        F_aproximada = integrate(series(f, x, 0, datos.n_terminos + 1).removeO(), x)
        resultado_exacto = integrate(f, (x, datos.a, datos.b))
        resultado_exacto_val = float(N(resultado_exacto))

        puntos = np.linspace(datos.a, datos.b, 1000)
        y_vals = np.array([f_lambda(xi) for xi in puntos])
        dx = (datos.b - datos.a) / 1000

        integral_simpson = simpson(y_vals, dx=dx)
        integral_romberg, _ = quad(f_lambda, datos.a, datos.b)
        integral_gauss, _ = quad(f_lambda, datos.a, datos.b)

        # 👇 Detectar funciones especiales
        explicaciones_funciones = obtener_definiciones_especiales(F_exacta)

        return {
            "primitiva_real": f"$$ {latex(F_exacta)} $$",
            "serie_taylor": f"$$ {latex(F_aproximada)} $$",
            "integral_definida_exacta": f"$$ {latex(resultado_exacto)} $$",
            "valor_numerico_exacto": resultado_exacto_val,
            "metodos_numericos": {
                "simpson": integral_simpson,
                "romberg": integral_romberg,
                "cuadratura_gaussiana": integral_gauss
            },
            "funciones_especiales": explicaciones_funciones
        }

    except Exception as e:
        return {"error": str(e)}
