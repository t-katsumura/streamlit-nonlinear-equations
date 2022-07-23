import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import plotly.graph_objects as go

@st.cache
def calc(mu=1.0, gamma=1.0, beta=0.0, tmin=0.0, tmax=100.0, dt=0.01):
    """
    Stuart Landau

    dr/dt = mu*r - r^3
    dc/dt = gamma - beta*r^2
    """

    def func(v, t, mu, gamma, beta):
        r = v[0]
        c = v[1]
        return [mu*r - r*r*r, gamma - beta*r*r]

    v0 = [0.1 , 0]
    t = np.arange(tmin, tmax+0.5*dt, dt).astype('float64')

    v = odeint(func, v0, t, args=(mu, gamma, beta))
    v = np.array(v).astype('float64').T

    return [v[0]*np.cos(v[1]), v[0]*np.sin(v[1])]

def app():

    st.title('Stuart Landau equation')
    st.latex(r"""
        \begin{equation*}
        \left \{
        \begin{array}{l}
        \frac{dr}{dt} = \mu \cdot r - r^3 \\
        \frac{dc}{dt} = \gamma - \beta \cdot r^2
        \end{array}
        \right.
        \end{equation*}
    """)

    with st.form("parameters"):
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("Equation Parameters")
            mu = st.slider("mu", -25.0, 25.0, 1.0, step=0.01, format="%.2f")
            gm = st.slider("gamma", -25.0, 25.0, 1.0, step=0.01, format="%.2f")
            bt = st.slider("beta", -25.0, 25.0, 0.0, step=0.01, format="%.2f")
        with col_right:
            st.write("Calculation Settings")
            tmin, tmax = st.slider('time range',0.0, 200.0, (0.0, 100.0), step=1.0, format="%f")
            dt = st.slider("time step", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
        st.form_submit_button("Update")

    v = calc(mu, gm, bt, tmin, tmax, dt)
    
    # --------------------------------
    fig = plt.figure()
    plt.plot(v[0], v[1], linewidth=0.5, color='blue')
    # plt.xlim(-1.1, 1.1)
    # plt.ylim(-1.1, 1.1)
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.show()
    st.pyplot(fig, clear_figure=True)
    # --------------------------------
    
    
    # --------------------------------
    gofig = go.Figure()
    gofig.add_trace(
        go.Scatter(
            x = v[0],
            y = v[1],
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
