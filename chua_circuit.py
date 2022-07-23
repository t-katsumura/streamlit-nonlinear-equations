import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import plotly.graph_objects as go

@st.cache
def calc(alpha=15.6, beta=28, m0=-1.143, m1=-0.714, tmin=0.0, tmax=100.0, dt=0.01):
    """
    Chua's circuit
    http://www.chuacircuits.com/matlabsim.php

    dx/dt = alpha( y - x - m1*x - 0.5*(m0-m1)*(|x+1|-|x-1|) )
    dy/dt = x - y + z
    dz/dt = -beta * y
    """

    def func(v, t, alpha, beta, m0, m1):
        x = v[0]
        y = v[1]
        z = v[2]
        return [alpha*(y-x-m1*x-0.5*(m0-m1)*(np.abs(x+1)-np.abs(x-1))), x-y+z, -beta*y]

    v0 = [0.7, 0, 0]
    t = np.arange(tmin, tmax+0.5*dt, dt).astype('float64')

    v = odeint(func, v0, t, args=(alpha, beta, m0, m1))
    v = np.array(v).astype('float64').T

    return v

def app():

    st.title('Chua\'s Circuit')
    st.latex(r"""
        \begin{equation*}
        \left \{
        \begin{array}{l}
        \frac{dx}{dt} = \alpha \cdot ( y-x-m_{1} \cdot x-0.5 \cdot (m_{0}-m_{1}) \cdot (|x+1|-|x-1|) ) \\
        \frac{dy}{dt} = x - y + z \\
        \frac{dz}{dt} = - \beta \cdot y
        \end{array}
        \right.
        \end{equation*}
    """)

    with st.form("parameters"):
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("Equation Parameters")
            ap = st.slider("alpha", 0.0, 50.0, 15.6, step=0.01, format="%.2f")
            bt = st.slider("beta", 0.0, 50.0, 28.0, step=0.01, format="%.2f")
            m0 = st.slider("m0", -5.0, 5.0, -1.143, step=0.001, format="%.3f")
            m1 = st.slider("m1", -1.0, 1.0, -0.714, step=0.001, format="%.3f")
        with col_right:
            st.write("Calculation Settings")
            tmin, tmax = st.slider('time range',0.0, 200.0, (0.0, 100.0), step=1.0, format="%f")
            dt = st.slider("time step", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
        st.form_submit_button("Update")

    v = calc(ap, bt, m0, m1, tmin, tmax, dt)
    
    # --------------------------------
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(v[0], v[1], v[2], linewidth=0.5, color='blue')
    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(-3, 3)
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
