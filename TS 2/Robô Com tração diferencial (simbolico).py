import sympy as sp

sp.init_printing(use_unicode=True)


r, d, phi, u_L, u_R = sp.symbols('r d phi u_L u_R')
omega, x_dot, y_dot = sp.symbols('omega x_dot y_dot')

# Montando a matriz de orietação do robô (q_dot), omega o ângulo de orientação, x_dot e y_dot as velocidades lineares nos eixos x e y
q_dot_sym = sp.Matrix([omega, x_dot, y_dot])

# Vetor de controle u com as velocidades das rodas esquerda (u_L) e direita (u_R).
u = sp.Matrix([u_L, u_R])

# Matriz de transformação G(q) com base nas caracteristicas fisicas do robô
G = sp.Matrix([
    [-r/(2*d),          r/(2*d)], 
    [(r/2)*sp.cos(phi), (r/2)*sp.cos(phi)], 
    [(r/2)*sp.sin(phi), (r/2)*sp.sin(phi)]
])

def posicao_robo(u, G): 
    # Monta a expressão visual "q_dot = G * u" sem resolvê-la
    equacao_visual = sp.Eq(q_dot_sym, sp.MatMul(G, u, evaluate=False))
    
    print("Equação matricial cinemática:\n")

    sp.pprint(equacao_visual)
    
    print("\n" + "-"*50 + "\n")

    # Realiza a multiplicação matricial analítica para extrair as linhas
    q_dot = G * u
    
    print("Orientação do robô (omega):\n")
    sp.pprint(q_dot[0])
    
    print("\nVelocidade linear no eixo x (x_dot):\n")
    sp.pprint(q_dot[1])
    
    print("\nVelocidade linear no eixo y (y_dot):\n")
    sp.pprint(q_dot[2])
    print("\n")

    return q_dot

q_dot = posicao_robo(u, G)