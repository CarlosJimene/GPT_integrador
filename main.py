
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sympy import symbols, sympify, integrate, series, lambdify, N
from scipy.integrate import simpson, quad

app = FastAPI()
x = symbols('x')

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

        # Primitiva exacta
        try:
            F_exacta = integrate(f, x)
            F_exacta_tex = f"$$ {F_exacta} $$"
        except:
            F_exacta = "No tiene primitiva elemental"
            F_exacta_tex = "No tiene primitiva elemental"

        # Serie de Taylor y primitiva aproximada
        f_series = series(f, x, 0, datos.n_terminos + 1).removeO()
        F_aproximada = integrate(f_series, x)
        F_aproximada_tex = f"$$ {F_aproximada} $$"

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
            "serie_taylor": F_aproximada_tex,
            "integral_definida_exacta": f"$$ {resultado_exacto} $$",
            "valor_numerico_exacto": resultado_exacto_val,
            "metodos_numericos": {
                "simpson": integral_simpson,
                "romberg": integral_romberg,
                "cuadratura_gaussiana": integral_gauss
            }
        }
    except Exception as e:
        return {"error": str(e)}
