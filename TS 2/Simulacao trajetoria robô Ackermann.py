import numpy as np
import pygame as pg

"""Parâmetros para o pygames."""

pg.init() #Inicializa o módulo de vídeo do Pygame para criar a janela e renderizar gráficos.
Largura_Tela, Altura_Tela = 800, 800 # Altura e largura da Tela
Tela = pg.display.set_mode((Largura_Tela, Altura_Tela)) # Altura e largura da Tela
pg.display.set_caption("Simulação de Trajetória do Robô") # Nome da Janela do Pygame

Relogio = pg.time.Clock() # Relógio para controlar o tempo real (dt)

Branco = (255, 255, 255) # Define a cor branca em RGB (para o fundo da Tela).
Preto = (0, 0, 0) # Define a cor preta em RGB.
Cinza = (200, 200, 200) # Define a cor cinza em RGB.
Azul = (0, 150, 255) # Define a cor azul para o robô em RGB.
Vermelho = (255, 50, 50) # Define a cor vermelha para o frente do robô em RGB.

Escala = 20   # Multiplicador para converter "metros" em "pixels" na Tela

x_robo_TD = (Largura_Tela / (2 * Escala) ) + 50 # Coordenada inicial em x do robô (Convertendo pixels para metros)
y_robo_TD = (Altura_Tela - 200) / (2 * Escala) # Coordenada inicial em y do robô (Convertendo pixels para metros)

x_robo_Ackermann = (Largura_Tela / (2 * Escala) ) - 50 # Coordenada inicial em x do robô Ackermann (Convertendo pixels para metros)
y_robo_Ackermann = (Altura_Tela - 200) / (2 * Escala) # Coordenada inicial em y do robô Ackermann (Convertendo pixels para metros)

'''Parâmetros para a HUD e botões de controle'''

Fonte_HUD = pg.font.SysFont("Consolas", 18) # Fonte para a HUD

# Dimensões e posições dos botões de controle
Posicao_Botoes_Y = Altura_Tela - 180 # Altura fixa para os botões (20 pixels de margem inferior + 5 pixels de borda + 15 pixels de margem interna)
Largura_Botao = 100  # Largura dos botões
Altura_Botao = 40  # Altura dos botçoes
Espaco_Botao = 15 # Espaço entre os botões


# Eixos no canto inferior esquerdo (Setas para os eixos x e y globais)
Setas_X = 30
Setas_Y = Altura_Tela - 40
Comprimento_Setas = 25

"""Parâmetros para o robô (tração diferencial) e simulação."""
r = 0.5 # raio das rodas do robô.
d = 1 # distância entre as rodas e o centro do robô.
phi = 0 # Angulo de orientação do robô em relação ao eixo x global.
u_L = 0 # Velocidade da roda esquerda.
u_R = 0 # Velocidade da roda direita.
vel_motores_lr = 3 # Variavel auxiliar para a velocidade dos motores. Usei ela para alterar as velocidades dos dois motores simultâneamente quando clicar nos botões de controle.

"""Parâmetros robô Ackermann"""""
L_ackermann = 2.5 # Distância entre o eixo dianteiro e o eixo traseiro do robô Ackermann.
velocidade_ackermann = 0 # Velocidade linear do robô Ackermann.
psi = 0 # Angulo de direção das rodas dianteiras do robô Ackermann.
psi_dot = 0 # Velocidade angular do chassi do robô Ackermann.
theta = 0 # Angulo de orientação do robô Ackermann em relação ao eixo x global.
PSI_MAX = np.pi / 4 # Limite de esterçamento para evitar tan(psi) instável.

posicao_robo_espacoTD = np.array([phi, x_robo_TD, y_robo_TD]) # Posição inicial do robô no espaço (phi, x, y).
posicao_robo_Ackermann = np.array([theta, x_robo_Ackermann, y_robo_Ackermann, psi]) # Posição inicial do robô Ackermann no espaço (theta, x, y, psi).

def calculo_velocidade_TDiferencial():
    """
    Calcula as taxas de variação da postura do robô diferencial no referencial global.

    Esta função resolve a cinemática direta do robô utilizando a matriz Jacobiana G(q). 
    Ela mapeia as velocidades angulares individuais das rodas motrizes para as 
    velocidades linear e angular do chassi no espaço cartesiano 2D.

    Parâmetros consumidos (Variáveis Globais):
    * posicao_robo_espacoTD : numpy.ndarray
        Vetor de estado do robô. A função lê o índice [0] para extrair a orientação 
        atual phi (em radianos) em relação ao eixo X global.
    * u_L : float
        Velocidade angular de controle aplicada à roda motriz esquerda.
    * u_R : float
        Velocidade angular de controle aplicada à roda motriz direita.
    * r : float
        Raio nominal das rodas do robô.
    * d : float
        Distância do centro do eixo longitudinal até cada roda (meia-bitola).

    Retorna:
    * q_dot : numpy.ndarray
        Um vetor 1D contendo as três componentes derivadas da postura atual:
        - q_dot[0]: Velocidade angular do chassi em torno do eixo Z (omega).
        - q_dot[1]: Velocidade linear do robô projetada no eixo X (x_dot).
        - q_dot[2]: Velocidade linear do robô projetada no eixo Y (y_dot).
    """
    phi = posicao_robo_espacoTD[0] # Obtém a orientação atual do robô (phi) para calcular as velocidades lineares corretamente.

    # Vetor de controle u com as velocidades das rodas esquerda (u_L) e direita (u_R).
    u = np.array([u_L, u_R])

    # Matriz de transformação G(q) com base nas caracteristicas fisicas do robô
    G = np.array([
        [-r/(2*d),          r/(2*d)], 
        [(r/2)*np.cos(phi), (r/2)*np.cos(phi)], 
        [(r/2)*np.sin(phi), (r/2)*np.sin(phi)]
    ])

    q_dot = G @ u
    return q_dot

def calculo_velocidade_Ackermann():

    theta = posicao_robo_Ackermann[0] # Obtém a orientação atual do robô (phi) para calcular as velocidades lineares corretamente.
    psi = posicao_robo_Ackermann[3] # Obtém o ângulo de direção das rodas dianteiras do robô Ackermann.

    u_ackermann = np.array([velocidade_ackermann, psi_dot]) # Vetor de controle u com a velocidade linear do robô e a velocidade angular das rodas dianteiras.

    # Matriz de transformação G(q) com base nas caracteristicas fisicas do robô Ackermann
    G_ackermann = np.array([
        [np.tan(psi) / L_ackermann, 0],
        [np.cos(theta), 0],
        [np.sin(theta), 0],
        [0, 1]
    ])

    q_dot_ackermann = G_ackermann @ u_ackermann
    return q_dot_ackermann
    


def integracao_numerica(q_dot,q_dot_ack, dt):
    """
    Atualiza a postura do robô no espaço através de integração numérica e aplica topologia de tela infinita.

    A função utiliza o Método de Integração de Euler de primeira ordem para acumular as 
    velocidades no vetor de estado atual do robô, avançando-o no tempo. Em seguida, 
    converte os limites da janela gráfica (pixels) para o espaço físico (metros) e aplica 
    uma operação de módulo para confinar o robô, fazendo com que ele reapareça do lado 
    oposto ao cruzar uma borda.

    Parâmetros:
    * q_dot : numpy.ndarray ou list
        Vetor 1D contendo as taxas de variação da postura no formato [omega, x_dot, y_dot].
    * dt : float
        Passo de tempo (delta t) da simulação em segundos, decorrido desde o último frame.

    Variáveis Globais Consumidas e Modificadas:
    * posicao_robo_espacoTD : numpy.ndarray
        Vetor de estado [phi, x, y]. Os valores são modificados e retidos in-place (diretamente 
        na memória) a cada iteração.
    * Largura_Tela : int
        Largura total da janela do Pygame em pixels.
    * Altura_Tela : int
        Altura total da janela do Pygame em pixels (descontada a área da HUD na lógica de limite).
    * Escala : float ou int
        Fator de conversão geométrica definindo quantos pixels representam 1 metro.

    Retorna:
    * None
        A função não retorna valores de forma explícita, atuando diretamente na mutação da 
        variável global `posicao_robo_espacoTD`.
    """
    # Atualiza as posições lineares (x, y) para o robô de tração diferencial
    posicao_robo_espacoTD[1] += q_dot[1] * dt  # Atualiza x usando x_dot
    posicao_robo_espacoTD[2] += q_dot[2] * dt  # Atualiza y usando y_dot
    posicao_robo_espacoTD[0] += q_dot[0] * dt  # Atualiza phi usando omega

    posicao_robo_Ackermann[1] += q_dot_ack[1] * dt  # Atualiza x usando x_dot
    posicao_robo_Ackermann[2] += q_dot_ack[2] * dt  # Atualiza y usando y_dot
    posicao_robo_Ackermann[0] += q_dot_ack[0] * dt  # Atualiza theta usando theta_dot
    posicao_robo_Ackermann[3] += q_dot_ack[3] * dt  # Atualiza psi usando psi_dot

    # 2. Limites máximos do espaço jogável (Convertendo pixels para metros)
    Max_X_Metros = Largura_Tela / Escala
    Max_Y_Metros = (Altura_Tela - 200) / Escala

    # 3. Lógica de "Tela Infinita" usando o operador de Módulo (%)
    posicao_robo_espacoTD[1] = posicao_robo_espacoTD[1] % Max_X_Metros
    posicao_robo_espacoTD[2] = posicao_robo_espacoTD[2] % Max_Y_Metros
    posicao_robo_Ackermann[1] = posicao_robo_Ackermann[1] % Max_X_Metros
    posicao_robo_Ackermann[2] = posicao_robo_Ackermann[2] % Max_Y_Metros
    posicao_robo_Ackermann[3] = np.clip(posicao_robo_Ackermann[3], -PSI_MAX, PSI_MAX)


def mostra_robo_diferencial():

    """Desenha o robô na Tela do Pygame com base na posição e orientação atuais, utilizando proporções físicas reais convertidas para pixels.

    A função calcula as coordenadas de desenho do robô, suas rodas e a seta indicadora de orientação (omega) com base na posição atual do robô no espaço. Além de desenhar a posição do robô no plano cartesiano (X , Y) global ela também representa a rotação do robô em relação ao angulo phi, rotacionando as rodas e a seta do robo de acordo com a variação desse angulo no tempo. Ela utiliza a biblioteca Pygame para renderizar formas geométricas que representam o chassi, as rodas e a seta de orientação.
    
    Não retorna valores, apenas desenha o robô na Tela."""

    x_pos = int(posicao_robo_espacoTD[1] * Escala) # Converte a posição x do robô de metros para pixels
    y_pos = int(Altura_Tela - 200 - (posicao_robo_espacoTD[2] * Escala)) # Converte a posição y do robô de metros para pixels
    phi = posicao_robo_espacoTD[0] # Converte a orientação do robô de radianos para pixels (não é necessário, mas mantive a variável phi para clareza)

    Raio_Chassi = int(d * Escala)    # Proporções Físicas em Pixels
    Comprimento_Roda = int(2 * r * Escala)    # Proporções Físicas em Pixels
    Largura_Roda = int(0.3 * Escala) # Largura visual arbitrária

    pg.draw.circle(Tela, Azul, (x_pos, y_pos), Raio_Chassi) # Desenha o chassi baseado na distância 'd'
    pg.draw.circle(Tela, Preto, (x_pos, y_pos), Raio_Chassi, 2) # Borda do chassi

    # Subfunção para desenhar as rodas rotacionadas
    def desenha_roda(x_centro, y_centro, angulo):

        """Desenha uma roda retangular rotacionada em torno do centro (x_centro, y_centro) com base no ângulo de orientação do robô.
        Essa subfunção é utilizada apenas para aspectos visuais da demosntração do bobô focado no comportamento das rodas. Ela não contribui para o controle ou cinemática do robô."""

        # Vetores diretores (Frente e Lado) baseados no ângulo phi
        dx_f = (Comprimento_Roda / 2) * np.cos(angulo) # Calcula o deslocamento horizontal da roda (metade do comprimento da roda) baseado no ângulo de orientação do robô.
        dy_f = -(Comprimento_Roda / 2) * np.sin(angulo) # Calcula o deslocamento vertical da roda (metade do comprimento da roda) baseado no ângulo de orientação do robô.
        
        dx_l = (Largura_Roda / 2) * np.cos(angulo + np.pi/2) # Calcula o deslocamento horizontal da roda (metade da largura da roda) baseado no ângulo de orientação do robô.
        dy_l = -(Largura_Roda / 2) * np.sin(angulo + np.pi/2) # Calcula o deslocamento vertical da roda (metade da largura da roda) baseado no ângulo de orientação do robô.
        
        # Calcula os 4 cantos do polígono (retângulo da roda)
        pontos = [
            (x_centro + dx_f + dx_l, y_centro + dy_f + dy_l),
            (x_centro + dx_f - dx_l, y_centro + dy_f - dy_l),
            (x_centro - dx_f - dx_l, y_centro - dy_f - dy_l),
            (x_centro - dx_f + dx_l, y_centro - dy_f + dy_l)
        ]
        pg.draw.polygon(Tela, Preto, pontos)

    # Calcula a posição dos centros das rodas (Deslocamento perpendicular 'd')
    # Roda Direita (phi - 90°)
    x_dir = x_pos + (d * Escala) * np.cos(phi - np.pi/2) 
    y_dir = y_pos - (d * Escala) * np.sin(phi - np.pi/2)
    
    # Roda Esquerda (phi + 90°)
    x_esq = x_pos + (d * Escala) * np.cos(phi + np.pi/2)
    y_esq = y_pos - (d * Escala) * np.sin(phi + np.pi/2)

    # Desenha as rodas usando a subfunção desenha_roda
    desenha_roda(x_dir, y_dir, phi)
    desenha_roda(x_esq, y_esq, phi)

    # Imprime uma seta indicando a orientação do robô (omega) saindo do centro do chassi, apontando para a direção "frontal" do robô (phi)
    Comprimento_Seta_Omega = Raio_Chassi + 10
    Final_Seta_Omega_X = int(x_pos + np.cos(phi) * Comprimento_Seta_Omega) 
    Final_Seta_Omega_Y = int(y_pos - np.sin(phi) * Comprimento_Seta_Omega) 
    pg.draw.line(Tela, Preto, (x_pos, y_pos), (Final_Seta_Omega_X, Final_Seta_Omega_Y), 3)
    
    # Imprime a ponta da seta (triângulo) para indicar a direção do "nariz" do robô
    pg.draw.polygon(Tela, Preto, [
        (Final_Seta_Omega_X, Final_Seta_Omega_Y),
        (int(Final_Seta_Omega_X - 6 * np.cos(phi - np.pi / 6)), int(Final_Seta_Omega_Y + 6 * np.sin(phi - np.pi / 6))),
        (int(Final_Seta_Omega_X - 6 * np.cos(phi + np.pi / 6)), int(Final_Seta_Omega_Y + 6 * np.sin(phi + np.pi / 6)))
    ])
    
    # Label do Omega
    Texto_Omega = Fonte_HUD.render("ω", True, Preto)
    Tela.blit(Texto_Omega, (Final_Seta_Omega_X + 5, Final_Seta_Omega_Y - 12))

def mostra_robo_ackermann():
    """
    Desenha o robô com cinemática de Ackermann (Modelo Bicicleta) na Tela do Pygame.

    A função extrai o vetor de estado do robô Ackermann, onde X e Y representam
    a posição do eixo traseiro. O chassi é desenhado como uma linha conectando o 
    eixo traseiro ao eixo dianteiro (calculado via trigonometria com a distância 'l').
    A roda traseira é desenhada alinhada ao chassi, e a roda dianteira virtual é 
    desenhada considerando o ângulo de esterçamento (psi).
    """
    # 1. Extração do Estado (Assumindo que você chamou a variável de posicao_robo_Ackermann)
    theta = posicao_robo_Ackermann[0] # Orientação do chassi em radianos
    x_metros = posicao_robo_Ackermann[1] # Posição X do eixo traseiro
    y_metros = posicao_robo_Ackermann[2] # Posição Y do eixo traseiro
    psi = posicao_robo_Ackermann[3] # Ângulo de esterçamento do volante em radianos

    # 2. Conversão da base (Eixo Traseiro) para pixels
    x_tras = int(x_metros * Escala)
    y_tras = int(Altura_Tela - 200 - (y_metros * Escala))

    # 3. Proporções Físicas em Pixels
    Comprimento_Roda = int(2 * r * Escala)
    Largura_Roda = int(0.3 * Escala)
    Comprimento_Chassi = int(L_ackermann * Escala) # 'l' é a distância entre-eixos definida no início do código

    # 4. Cálculo da posição do Eixo Dianteiro (Ponta do chassi)
    x_frente = int(x_tras + Comprimento_Chassi * np.cos(theta))
    y_frente = int(y_tras - Comprimento_Chassi * np.sin(theta))

    # 5. Desenha o "Chassi" (Linha vermelha conectando os dois eixos)
    pg.draw.line(Tela, Vermelho, (x_tras, y_tras), (x_frente, y_frente), 4)
    
    # Desenha pequenos pontos de articulação nos eixos
    pg.draw.circle(Tela, Preto, (x_tras, y_tras), 4)
    pg.draw.circle(Tela, Preto, (x_frente, y_frente), 4)

    # 6. Subfunção para desenhar as rodas (mesma lógica matemática que você já criou)
    def desenha_roda(x_centro, y_centro, angulo):
        dx_f = (Comprimento_Roda / 2) * np.cos(angulo)
        dy_f = -(Comprimento_Roda / 2) * np.sin(angulo)
        
        dx_l = (Largura_Roda / 2) * np.cos(angulo + np.pi/2)
        dy_l = -(Largura_Roda / 2) * np.sin(angulo + np.pi/2)
        
        pontos = [
            (x_centro + dx_f + dx_l, y_centro + dy_f + dy_l),
            (x_centro + dx_f - dx_l, y_centro + dy_f - dy_l),
            (x_centro - dx_f - dx_l, y_centro - dy_f - dy_l),
            (x_centro - dx_f + dx_l, y_centro - dy_f + dy_l)
        ]
        pg.draw.polygon(Tela, Preto, pontos)

    # 7. Desenha as Duas Rodas
    # Roda Traseira: Fica no eixo traseiro e NÃO esterça (ângulo é apenas theta)
    desenha_roda(x_tras, y_tras, theta)

    # Roda Dianteira: Fica no eixo dianteiro e esterça (ângulo é theta + psi)
    desenha_roda(x_frente, y_frente, theta + psi)


def obter_controle_usuario():
    """
    Função para detectar os inputs do usuário através de cliques do mouse na interface gráfica.
    Atualiza as variáveis de controle u_L e u_R (Diferencial) ou velocidade_ackermann e psi_dot (Ackermann) 
    dependendo da fileira e do botão clicado.
    """
    global u_L, u_R, velocidade_ackermann, psi_dot

    if pg.mouse.get_pressed()[0]:  # Botão esquerdo do mouse
        mouse_x, mouse_y = pg.mouse.get_pos()
        
        # 1. CONTROLES DO ROBÔ DIFERENCIAL (Fileira Superior)
        # Altura vai de (Altura_Tela - 180) até (Altura_Tela - 140)
        if (Altura_Tela - 180) <= mouse_y <= (Altura_Tela - 140):
            if 150 <= mouse_x <= 250:      # Botão "PARADO"
                u_L, u_R = 0, 0
            elif 265 <= mouse_x <= 365:    # Botão "AVANÇAR"
                u_L, u_R = vel_motores_lr, vel_motores_lr
            elif 380 <= mouse_x <= 480:    # Botão "RÉ"
                u_L, u_R = -vel_motores_lr, -vel_motores_lr
            elif 495 <= mouse_x <= 595:    # Botão "DIR" (Gira no próprio eixo)
                u_L, u_R = -vel_motores_lr, vel_motores_lr
            elif 610 <= mouse_x <= 710:    # Botão "ESQ" (Gira no próprio eixo)
                u_L, u_R = vel_motores_lr, -vel_motores_lr

        # 2. CONTROLES DO ROBÔ ACKERMANN (Fileira Inferior)
        # Altura vai de (Altura_Tela - 120) até (Altura_Tela - 80)
        elif (Altura_Tela - 120) <= mouse_y <= (Altura_Tela - 80):
            taxa_esterco = 1.5 # Variável local: Velocidade com que o volante gira (rad/s)
            
            if 150 <= mouse_x <= 250:      # Botão "PARADO"
                velocidade_ackermann, psi_dot = 0, 0
                
            elif 265 <= mouse_x <= 365:    # Botão "AVANÇAR"
                velocidade_ackermann, psi_dot = vel_motores_lr, 0
                
            elif 380 <= mouse_x <= 480:    # Botão "RÉ"
                velocidade_ackermann, psi_dot = -vel_motores_lr, 0
                
            elif 495 <= mouse_x <= 595:    # Botão "DIR"
                psi_dot = -taxa_esterco
                
            elif 610 <= mouse_x <= 710:    # Botão "ESQ"
                psi_dot = taxa_esterco

    else:
        # Sem clique, as velocidades são zeradas imediatamente para parar o robô.
        u_L, u_R = 0, 0
        velocidade_ackermann = 0
        psi_dot = 0

def desenhar_botao(texto, cor, x, y):

    """Função para desenhar um botão na tela com um texto e uma cor especificadas em coordenadas X e Y.
    Recebe o texto a ser exibido, a cor, e as posições X e Y. Renderiza e centraliza o texto."""

    # Cria o retângulo do botão usando o novo parâmetro 'y'
    Botao_Retangular = pg.Rect(x, y, Largura_Botao, Altura_Botao)
    pg.draw.rect(Tela, cor, Botao_Retangular)
        
    # Renderiza o texto e centraliza ele EXATAMENTE no meio do botão
    Texto_renderizado = Fonte_HUD.render(texto, True, Branco)
    Texto_rentangular = Texto_renderizado.get_rect(center=Botao_Retangular.center)
    Tela.blit(Texto_renderizado, Texto_rentangular)

def desenhar_interface():

    """Desenha a interface gráfica da HUD, incluindo a faixa inferior, os botões de controle para os robôs Diferencial e Ackermann, e as setas dos eixos globais."""

    Tela.fill(Branco) # Limpa o frame anterior
    pg.draw.rect(Tela, Cinza, (0, Altura_Tela - 200, Largura_Tela, 200)) # Desenha a faixa inferior na Tela
    pg.draw.rect(Tela, Preto, (0, Altura_Tela - 200, Largura_Tela, 200), 5) # Borda preta de 5 pixels

    # Definição de Alturas e Espaçamentos
    Y_Linha_Dif = Altura_Tela - 180   # Altura da fileira do Diferencial
    Y_Linha_Ack = Altura_Tela - 120   # Altura da fileira do Ackermann (60 pixels abaixo)
    Offset_X_Botoes = 150             # Empurra os botões 150 pixels para a direita

    # --- Textos Indicativos (Labels) ---
    Texto_Dif = Fonte_HUD.render("DIFERENCIAL:", True, Preto)
    Tela.blit(Texto_Dif, (20, Y_Linha_Dif + 10)) # Centralizado visualmente com a altura do botão

    Texto_Ack = Fonte_HUD.render("ACKERMANN:", True, Preto)
    Tela.blit(Texto_Ack, (20, Y_Linha_Ack + 10))


    # --- Renderização dos Botões ---
    Botoes_Config = [
        ("PARADO", Azul),
        ("AVANÇAR", Azul),
        ("RÉ", Azul),
        ("DIR", Vermelho),
        ("ESQ", Vermelho)
    ]

    # Desenha a fileira 1 (Diferencial)
    Posicao_X = Offset_X_Botoes
    for texto, cor in Botoes_Config:
        desenhar_botao(texto, cor, Posicao_X, Y_Linha_Dif)
        Posicao_X += Largura_Botao + Espaco_Botao

    # Desenha a fileira 2 (Ackermann)
    Posicao_X = Offset_X_Botoes
    for texto, cor in Botoes_Config:
        desenhar_botao(texto, cor, Posicao_X, Y_Linha_Ack)
        Posicao_X += Largura_Botao + Espaco_Botao


    # --- Desenho dos Eixos Globais (Mantido igual) ---
    pg.draw.line(Tela, Vermelho, (Setas_X, Setas_Y), (Setas_X + Comprimento_Setas, Setas_Y), 2)
    pg.draw.polygon(Tela, Vermelho, [(Setas_X + Comprimento_Setas, Setas_Y), (Setas_X + Comprimento_Setas - 5, Setas_Y - 3), (Setas_X + Comprimento_Setas - 5, Setas_Y + 3)])
    texto_x = Fonte_HUD.render("X", True, Vermelho)
    Tela.blit(texto_x, (Setas_X + Comprimento_Setas + 5, Setas_Y - 8))

    pg.draw.line(Tela, Azul, (Setas_X, Setas_Y), (Setas_X, Setas_Y - Comprimento_Setas), 2)
    pg.draw.polygon(Tela, Azul, [(Setas_X, Setas_Y - Comprimento_Setas), (Setas_X - 3, Setas_Y - Comprimento_Setas + 5), (Setas_X + 3, Setas_Y - Comprimento_Setas + 5)])
    texto_y = Fonte_HUD.render("Y", True, Azul)
    Tela.blit(texto_y, (Setas_X - 8, Setas_Y - Comprimento_Setas - 8))

Rodando = True

while Rodando: # Laço de repetição principal do jogo, que verifica que o jogo está rodando e atualiza a simulação a cada frame.

    dt = Relogio.tick(60) / 1000.0 # Trava o jogo em 60 FPS e pega o tempo que passou desde o último frame (em segundos)

    for Evento in pg.event.get():  #Interrupções para encerrar o jogo.
        if Evento.type == pg.QUIT: # Se o usuário clicar no "X" da janela, encerra o jogo
            Rodando = False
        elif Evento.type == pg.KEYDOWN and Evento.key == pg.K_ESCAPE: # Se o usuário pressionar a tecla "ESC", encerra o jogo
            Rodando = False
    

    desenhar_interface() # Desenha a interface da HUD (caixa cinza, bordas, etc)

    obter_controle_usuario() # Verifica se o usuário clicou em algum botão e atualiza u_L e u_R

    integracao_numerica(calculo_velocidade_TDiferencial(),calculo_velocidade_Ackermann(), dt) # Atualiza a posição do robô no espaço usando integração numérica

    mostra_robo_diferencial() # Desenha o robô na Tela

    mostra_robo_ackermann() # Desenha o robô Ackermann na Tela
    
    pg.display.flip() # Atualiza o conteúdo da Tela para mostrar as mudanças feitas (desenho do robô, interface, etc)