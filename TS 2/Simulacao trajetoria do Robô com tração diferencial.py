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

x_robo = Largura_Tela / (2 * Escala) # Coordenada inicial em x do robô (Convertendo pixels para metros)
y_robo = (Altura_Tela - 200) / (2 * Escala) # Coordenada inicial em y do robô (Convertendo pixels para metros)

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

posicao_robo_espaco = np.array([phi, x_robo, y_robo]) # Posição inicial do robô no espaço (phi, x, y).

def calculo_velocidade_orientacao():
    """
    Calcula as taxas de variação da postura do robô diferencial no referencial global.

    Esta função resolve a cinemática direta do robô utilizando a matriz Jacobiana G(q). 
    Ela mapeia as velocidades angulares individuais das rodas motrizes para as 
    velocidades linear e angular do chassi no espaço cartesiano 2D.

    Parâmetros consumidos (Variáveis Globais):
    * posicao_robo_espaco : numpy.ndarray
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
    phi = posicao_robo_espaco[0] # Obtém a orientação atual do robô (phi) para calcular as velocidades lineares corretamente.

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

def integracao_numerica(q_dot, dt):
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
    * posicao_robo_espaco : numpy.ndarray
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
        variável global `posicao_robo_espaco`.
    """
    # Atualiza as posições lineares (x, y)
    posicao_robo_espaco[1] += q_dot[1] * dt  # Atualiza x usando x_dot
    posicao_robo_espaco[2] += q_dot[2] * dt  # Atualiza y usando y_dot
    posicao_robo_espaco[0] += q_dot[0] * dt  # Atualiza phi usando omega

    # 2. Limites máximos do espaço jogável (Convertendo pixels para metros)
    Max_X_Metros = Largura_Tela / Escala
    Max_Y_Metros = (Altura_Tela - 200) / Escala

    # 3. Lógica de "Tela Infinita" usando o operador de Módulo (%)
    posicao_robo_espaco[1] = posicao_robo_espaco[1] % Max_X_Metros
    posicao_robo_espaco[2] = posicao_robo_espaco[2] % Max_Y_Metros


def mostra_robo():

    """Desenha o robô na Tela do Pygame com base na posição e orientação atuais, utilizando proporções físicas reais convertidas para pixels.

    A função calcula as coordenadas de desenho do robô, suas rodas e a seta indicadora de orientação (omega) com base na posição atual do robô no espaço. Além de desenhar a posição do robô no plano cartesiano (X , Y) global ela também representa a rotação do robô em relação ao angulo phi, rotacionando as rodas e a seta do robo de acordo com a variação desse angulo no tempo. Ela utiliza a biblioteca Pygame para renderizar formas geométricas que representam o chassi, as rodas e a seta de orientação.
    
    Não retorna valores, apenas desenha o robô na Tela."""

    x_pos = int(posicao_robo_espaco[1] * Escala) # Converte a posição x do robô de metros para pixels
    y_pos = int(Altura_Tela - 200 - (posicao_robo_espaco[2] * Escala)) # Converte a posição y do robô de metros para pixels
    phi = posicao_robo_espaco[0] # Converte a orientação do robô de radianos para pixels (não é necessário, mas mantive a variável phi para clareza)

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


def obter_controle_usuario():

    """Função para detectar os inputs do usuário através de cliques do mouse na interface gráfica, atualizando as variáveis de controle u_L e u_R com base no botão clicado. Ela verifica se o botão esquerdo do mouse foi pressionado e, em seguida, determina a posição do clique para identificar qual botão foi acionado (PARADO, AVANÇAR, RÉ, DIR ou ESQ). Dependendo do botão clicado, as velocidades das rodas esquerda (u_L) e direita (u_R) são atualizadas para controlar o movimento do robô."""

    if pg.mouse.get_pressed()[0]:  # Botão esquerdo do mouse
        global u_L, u_R
        mouse_x, mouse_y = pg.mouse.get_pos()
        
        # Como todos os botões estão na mesma altura, checamos o Y (Altura) apenas uma vez
        # A altura vai de (Altura_Tela - 180) até (Altura_Tela - 180 + 40 = Altura_Tela - 140)
        if (Altura_Tela - 180) <= mouse_y <= (Altura_Tela - 140):
            
            # Checamos o X (Largura) de cada botão individualmente
            if 20 <= mouse_x <= 120:  # Botão "PARADO"
                u_L, u_R = 0, 0
            elif 135 <= mouse_x <= 235:  # Botão "AVANÇAR"
                u_L, u_R = vel_motores_lr, vel_motores_lr
            elif 250 <= mouse_x <= 350:  # Botão "RÉ"
                u_L, u_R = -vel_motores_lr, -vel_motores_lr
            elif 365 <= mouse_x <= 465:  # Botão "DIR"
                u_L, u_R = -vel_motores_lr, vel_motores_lr
            elif 480 <= mouse_x <= 580:  # Botão "ESQ"
                u_L, u_R = vel_motores_lr, -vel_motores_lr

def desenhar_botao(texto, cor, x):

    """Função para desenhar um botão na tela com um texto e uma cor especificadas.
    Recebe o texto a ser exibido no botão, a cor do botão e a posição horizontal (x) onde o botão deve ser desenhado. A função cria um retângulo para o botão, renderiza o texto e centraliza-o dentro do retângulo, garantindo que os botões de controle sejam visualmente claros e interativos para o usuário."""

    # Cria o retângulo do botão
    Botao_Retangular = pg.Rect(x, Posicao_Botoes_Y, Largura_Botao, Altura_Botao)
    pg.draw.rect(Tela, cor, Botao_Retangular)
        
    # Renderiza o texto e centraliza ele EXATAMENTE no meio do botão
    Texto_renderizado = Fonte_HUD.render(texto, True, Branco)
    Texto_rentangular = Texto_renderizado.get_rect(center=Botao_Retangular.center)
    Tela.blit(Texto_renderizado, Texto_rentangular)

def desenhar_interface():

    """Desenha a interface gráfica da HUD, incluindo a faixa inferior, os botões de controle e as setas dos eixos globais. Esta função é responsável por criar a base visual da simulação, garantindo que os elementos de controle e orientação estejam claramente visíveis para o usuário. Ela limpa o frame anterior, desenha a faixa inferior onde os botões estão localizados, renderiza cada botão com seu respectivo texto e cor, e finalmente desenha as setas indicadoras dos eixos X e Y globais no canto inferior esquerdo da tela."""

    Tela.fill(Branco) # Limpa o frame anterior
    pg.draw.rect(Tela, Cinza, (0, Altura_Tela - 200, Largura_Tela, 200)) # Desenha a faixa inferior na Tela
    pg.draw.rect(Tela, Preto, (0, Altura_Tela - 200, Largura_Tela, 200), 5) # Borda preta de 5 pixels

    # Desenhando os 5 botões de forma sequencial (usando a função desenhar_botao para evitar repetição de código)
    Posicao_X_Botoes = 20
    desenhar_botao("PARADO", Azul, Posicao_X_Botoes)
    
    Posicao_X_Botoes += Largura_Botao + Espaco_Botao
    desenhar_botao("AVANÇAR", Azul, Posicao_X_Botoes)
    
    Posicao_X_Botoes += Largura_Botao + Espaco_Botao
    desenhar_botao("RÉ", Azul, Posicao_X_Botoes)
    
    Posicao_X_Botoes += Largura_Botao + Espaco_Botao
    desenhar_botao("DIR", Vermelho, Posicao_X_Botoes)
    
    Posicao_X_Botoes += Largura_Botao + Espaco_Botao
    desenhar_botao("ESQ", Vermelho, Posicao_X_Botoes)

    # Desenha a seta do Eixo X global (vermelho)
    pg.draw.line(Tela, Vermelho, (Setas_X, Setas_Y), (Setas_X + Comprimento_Setas, Setas_Y), 2)
    pg.draw.polygon(Tela, Vermelho, [(Setas_X + Comprimento_Setas, Setas_Y), (Setas_X + Comprimento_Setas - 5, Setas_Y - 3), (Setas_X + Comprimento_Setas - 5, Setas_Y + 3)])
    texto_x = Fonte_HUD.render("X", True, Vermelho)
    Tela.blit(texto_x, (Setas_X + Comprimento_Setas + 5, Setas_Y - 8))

    # Desenha a seta do Eixo Y global (azul)
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

    integracao_numerica(calculo_velocidade_orientacao(), dt) # Atualiza a posição do robô no espaço usando integração numérica

    mostra_robo() # Desenha o robô na Tela
    
    pg.display.flip() # Atualiza o conteúdo da Tela para mostrar as mudanças feitas (desenho do robô, interface, etc)