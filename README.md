# 📐 GPT_Integrador

**GPT_Integrador** es una API desarrollada con FastAPI que permite calcular integrales definidas de funciones simbólicas, detectar primitivas (elementales o especiales), y realizar aproximaciones mediante series de Taylor, además de aproximar el valor de la integral definida mediante diversos métodos numéricos. También genera gráficas comparativas y expresiones listas para usar en GeoGebra.

---

## 🚀 Funcionalidades principales

- ✅ Cálculo de primitivas simbólicas (cuando sea posible)
- 🔍 Detección de funciones especiales como `erf(x)`, `Si(x)`, `Li(x)`
- ∫ Cálculo de la integral definida entre dos límites
- 🔁 Aproximación con series de Taylor de orden configurable
- 📊 Comparación con métodos numéricos: Simpson, Romberg, Cuadratura Gaussiana y Monte Carlo
- 📎 Expresiones para copiar directamente en [GeoGebra](https://www.geogebra.org/graphing)

---

## 🛠 Tecnologías utilizadas

- Python 3.11+
- FastAPI
- SymPy
- NumPy
- SciPy

---

## 📦 Instalación

Clona el repositorio y luego instala las dependencias:

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
pip install -r requirements.txt
