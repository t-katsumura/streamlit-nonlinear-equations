import streamlit as st

import lorentz_attractor
import chua_circuit
import stuart_landau

PAGES = {
    "Lorentz Attractor": lorentz_attractor,
    "Chua's Circuit": chua_circuit,
    "Stuart Landau": stuart_landau,
}

st.sidebar.title('Non Linear Equations')
selection = st.sidebar.radio("Select equation", list(PAGES.keys()))
page = PAGES[selection]
page.app()