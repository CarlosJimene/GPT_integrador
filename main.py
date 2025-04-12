from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from sympy import (
    symbols, sympify, integrate, series, lambdify, N,
    solveset, Interval, oo, log, sin, cos, exp, diff, factorial,
    limit, Sum, S, latex, simplify
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
            F_exacta_tex = f"$$ {latex(F_exacta)} $$"
        except:
            F_exacta = "No tiene primitiva elemental"
            F_exacta_tex = "No tiene primitiva elemental"

        if str(datos.a) == "inf" or str(datos.b) == "inf":
            resultado_exacto = limit(integrate(f, (x, datos.a, datos.b)), x, oo)
            if resultado_exacto in [oo, -oo]:
                raise HTTPException(status_code=400, detail="La integral tiene un valor infinito.")
            resultado_exacto_val = float(N(resultado_exacto))
            resultado_exacto_tex = f"$$ {latex(resultado_exacto)} $$"
        else:
            resultado_exacto = integrate(f, (x, datos.a, datos.b))
            resultado_exacto_val = float(N(resultado_exacto))
            resultado_exacto_tex = f"$$ {latex(resultado_exacto)} $$"

        # Serie de Taylor general (expresión simbólica infinita)
        serie_general = Sum(
            diff(f, x, n).subs(x, datos.a) / factorial(n) * (x - datos.a)**n,
            (n, 0, oo)
        )
        sumatoria_general_tex = f"$$ {latex(serie_general)} $$"

        # ✨ Explicación textual automática después de la sumatoria
        explicacion_taylor = (
            f"**Para la función** \\( {latex(f)} \\), "
            f"**el desarrollo en serie de Taylor alrededor de** \\( x = {datos.a} \\) **es:**"
        )

        # Serie truncada hasta n términos (forma simbólica exacta)
        terminos_taylor = []
        for i in range(datos.n_terminos):
            deriv_i = diff(f, x, i)
            coef_i = simplify(deriv_i.subs(x, datos.a) / factorial(i))
            term_i = coef_i * (x - datos.a)**i
            terminos_taylor.append(term_i)

        f_series = sum(terminos_taylor)
        f_series_tex = f"$$ {latex(f_series)} $$"
        F_aproximada = integrate(f_series, x)
        F_aproximada_tex = f"$$ {latex(F_aproximada)} $$"

        integral_definida_tex = f"$$ \\int_{{{datos.a}}}^{{{datos.b}}} {latex(f)} \\, dx $$"

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
            "serie_taylor_general": sumatoria_general_tex,
            "explicacion_taylor_general": explicacion_taylor,
            "serie_taylor_finita": f_series_tex,
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
