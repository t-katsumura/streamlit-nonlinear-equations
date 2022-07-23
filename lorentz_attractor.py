import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import plotly.graph_objects as go

@st.cache
def calc(p=10, r=30, b=1, tmin=0.0, tmax=100.0, dt=0.01):
    """
    Lorentz attractor

    dx/dt = -p*x + p*y
    dy/dt = -x*z + r*x - y
    dz/dt =  x*y - b*z
    """

    def func(v, t, p, r, b):
        x = v[0]
        y = v[1]
        z = v[2]
        return [-p*x+p*y, -x*z+r*x-y, x*y-b*z]

    v0 = [0.1, 0.1, 0.1]
    t = np.arange(tmin, tmax+0.5*dt, dt)

    v = odeint(func, v0, t, args=(p, r, b))
    v = np.array(v).astype('float32').T

    H = 3
    W = 1
    HW = H*W
    N = len(t)
    frequency = 1/dt
    params = {'p':p, 'r':r, 'b':b, 'v0':v0}

    return v


def app():

    st.title('Lorentz attractor')
    st.latex(r"""
        \begin{equation*}
        \left \{
        \begin{array}{l}
        \frac{dx}{dt} = - p \cdot x + p \cdot y \\
        \frac{dy}{dt} = - x \cdot z + r \cdot x \\
        \frac{dz}{dt} =   x \cdot y - b \cdot z
        \end{array}
        \right.
        \end{equation*}
    """)

    with st.form("parameters"):
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("Equation Parameters")
            p = st.slider("p", 0.0, 100.0, 10.0, step=0.01, format="%.2f")
            r = st.slider("r", 0.0, 100.0, 30.0, step=0.01, format="%.2f")
            b = st.slider("b", 0.0, 10.0, 1.0, step=0.001, format="%.3f")
        with col_right:
            st.write("Calculation Settings")
            tmin, tmax = st.slider('time range',0.0, 200.0, (0.0, 100.0), step=1.0, format="%f")
            dt = st.slider("time step", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
        st.form_submit_button("Update")

    v = calc(p, r, b, tmin, tmax, dt)
    
    # --------------------------------
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(v[0], v[1], v[2], linewidth=0.5, color='blue')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # plt.show()
    st.pyplot(fig, clear_figure=True)
    # --------------------------------
    
    
    # --------------------------------
    gofig = go.Figure()
    gofig.add_trace(
        go.Scatter3d(
            x = v[0],
            y = v[1],
            z = v[2],
            mode="lines",
            line_color="blue",
            line_width=2,
        ) 
    )
    gofig.update_layout(
        width=700,
        height=700,
    ) 
    st.plotly_chart(gofig, use_container_width=True)
    # --------------------------------
