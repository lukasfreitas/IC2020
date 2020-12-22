#------------------------------------------------------------------------------------------------------------------------Teste final
#-----------------------------------------------------------------------------------------------------------------------------------
import  pymunkoptions
pymunkoptions.options["debug"] = False
import  pymunk
import  pickle
import math
import matplotlib.pyplot as plt
import numpy as np

def registrador(filename,var):
    outfile = open(filename,'wb')
    pickle.dump(var,outfile)
    outfile.close()  

class PIDControl:
    def __init__(self, Ts, KP, KI, KD=0):
        self.ek = 0.0       # erro atual
        self.uk = 0.0       # controle atual
        self.ek1 = 0.0      # erro anterior e[k-1]
        self.uk1 = 0.0      # controle anterior u[k-1]
        self.ek2 = 0.0      # erro 2x anterior e[k-2]
        self.KP = KP        # proporcional
        self.KI = KI        # integral
        self.KD = KD        # derivativo
        self.Ts = Ts        # sample time


    def control(self, r, y):
        # registra erro e controle passados
        self.ek2 = self.ek1
        self.ek1 = self.ek
        self.uk1 = self.uk

        # atualiza erro
        self.ek = r - y
        
        # controle PI discretizado por backwards differences
        du = (self.KP + self.KI*self.Ts + self.KD/self.Ts)*self.ek - (self.KP + 2*self.KD/self.Ts)*self.ek1 + self.KD/self.Ts*self.ek2
        self.uk = self.uk1 + du
        return self.uk

class Simulacao:
    def __init__(self, FPS=60.0, KP=200, KI=50, KD=25):
        #-----------------------------------------------------------------------------------------------------------------------------------
        self.FPS = 60.0 # 60 quadros por segundo
        #-----------------------------------------------------------------------------------------------------------------------------------
        ambiente               =              pymunk.Space() 
        ambiente.gravity       =              0,-98.1 ##x,y
        #-------------------------------------------------------------------------------------------------------------------------------piso
        piso_ponto_A           =              0,0
        piso_ponto_B           =              640,0
        piso_shape             =              pymunk.Segment(None,(piso_ponto_A),(piso_ponto_B),2)
        piso_momento           =              pymunk.moment_for_segment(1000,(piso_ponto_A),(piso_ponto_B),2)
        piso_fisica            =              pymunk.Body(1,piso_momento,pymunk.Body.KINEMATIC)
        piso_shape.body        =              piso_fisica
        piso_shape.friction    =              0.62

        ambiente.add(piso_fisica,piso_shape)
        #-----------------------------------------------------------------------------------------------------------------------------chassi
        chassi_posicao         =              100,35
        chassi_tamanho         =              80,20
        chassi_massa           =              30 #gramas
        chassi_shape           =              pymunk.Poly.create_box(None,size=chassi_tamanho)
        chassi_momento         =              pymunk.moment_for_box(chassi_massa,chassi_tamanho)
        chassi_fisica          =              pymunk.Body(chassi_massa,chassi_momento,pymunk.Body.DYNAMIC)
        chassi_shape.body      =              chassi_fisica   
        chassi_fisica.position =              chassi_posicao 
        chassi_shape.friction  =              0.4 

        ambiente.add(chassi_shape,chassi_fisica)
        #-------------------------------------------------------------------------------------------------------------------------R_traseira
        R_traseira_posicao     =              (chassi_posicao[0]-(chassi_tamanho[0]/2))-20,(chassi_posicao[1]-20)
        R_traseira_massa       =              10
        R_traseira_raio_in     =              10
        R_traseira_raio_out    =              10
        R_traseira_momento     =              pymunk.moment_for_circle(R_traseira_massa,R_traseira_raio_in,R_traseira_raio_out)
        R_traseira_fisica      =              pymunk.Body(R_traseira_massa,R_traseira_momento,pymunk.Body.DYNAMIC)
        R_traseira_shape       =              pymunk.Circle(R_traseira_fisica,R_traseira_raio_out)
        R_traseira_fisica.position  =         R_traseira_posicao
        R_traseira_shape.friction   =         0.6

        ambiente.add(R_traseira_shape,R_traseira_fisica)
        #------------------------------------------------------------------------------------------------------------------------R_dianteira
        R_dianteira_posicao     =             (chassi_posicao[0]+(chassi_tamanho[0]/2))+20,(chassi_posicao[1]-20)
        R_dianteira_massa       =             10
        R_dianteira_raio_in     =             10
        R_dianteira_raio_out    =             10
        R_dianteira_momento     =             pymunk.moment_for_circle(R_dianteira_massa,R_dianteira_raio_in,R_dianteira_raio_out)
        R_dianteira_fisica      =             pymunk.Body(R_dianteira_massa,R_dianteira_momento,pymunk.Body.DYNAMIC)
        R_dianteira_shape       =             pymunk.Circle(R_dianteira_fisica,R_dianteira_raio_out)
        R_dianteira_fisica.position  =        R_dianteira_posicao
        R_dianteira_shape.friction   =        0.6

        ambiente.add(R_dianteira_fisica,R_dianteira_shape)
        #----------------------------------------------------------------------------------------------------------------------------Pendulo
        pendulo_posicao     =              chassi_posicao[0],chassi_posicao[1]+70
        pendulo_massa       =              10
        pendulo_raio_in     =              5
        pendulo_raio_out    =              5
        pendulo_momento     =              pymunk.moment_for_circle(pendulo_massa,pendulo_raio_in,pendulo_raio_out)
        pendulo_fisica      =              pymunk.Body(pendulo_massa,pendulo_momento,pymunk.Body.DYNAMIC)
        pendulo_shape       =              pymunk.Circle(pendulo_fisica,pendulo_raio_out)
        pendulo_fisica.position  =         pendulo_posicao
        pendulo_shape.friction   =         0.6
        pendulo_shape.color      =         (255, 0, 0, 255) ##RGBA Vermelho

        junta_pendulo = pymunk.PinJoint(pendulo_fisica,chassi_fisica,(0,0),(0,10))

        # filtro de colisão - permite contato apenas com o chão
        chassi_shape.filter = pymunk.ShapeFilter(group=1)
        pendulo_shape.filter = chassi_shape.filter
        R_traseira_shape.filter = chassi_shape.filter
        R_dianteira_shape.filter = chassi_shape.filter


        ambiente.add(pendulo_fisica,pendulo_shape,junta_pendulo)

        #-----------------------------------------------------------------------------------------------------------------------------juntas
        ambiente.add(pymunk.PinJoint(R_traseira_fisica , chassi_fisica,      (0,0), (-40, 10)),
                    pymunk.PinJoint(R_dianteira_fisica, chassi_fisica,      (0,0), ( 40, 10)),
                    pymunk.PinJoint(R_dianteira_fisica, chassi_fisica,      (0,0), ( 40,-10)),
                    pymunk.PinJoint(R_traseira_fisica , chassi_fisica,      (0,0), (-40,-10)))
        #------------------------------------------------------------------------------------------------------------------------------Motor
        velocidade = 0
        M_traseiro = pymunk.SimpleMotor(R_traseira_fisica,chassi_fisica,velocidade)
        M_dianteiro = pymunk.SimpleMotor(R_dianteira_fisica,chassi_fisica,velocidade)
        ambiente.add(M_traseiro,M_dianteiro)

        self.ambiente = ambiente
        self.pendulo_fisica = pendulo_fisica
        self.chassi_fisica = chassi_fisica
        self.M_dianteiro = M_dianteiro
        self.M_traseiro = M_traseiro

        self.rodas_controle = PIDControl(1.0/FPS, KP, KI, KD) # PID
        self.r = math.pi/2 # referência a ser seguida

    def pendulo_angulo(self):
        angulo_junta_pendulo = math.atan2(self.pendulo_fisica.position[1] - self.chassi_fisica.position[1], self.pendulo_fisica.position[0] - self.chassi_fisica.position[0])

        # regulariza o angulo na faixa [0, 2*pi]
        while angulo_junta_pendulo < 0:
            angulo_junta_pendulo += 2*math.pi

        while angulo_junta_pendulo > 2*math.pi:
            angulo_junta_pendulo -= 2*math.pi

        return float(angulo_junta_pendulo)

    def calcula_metricas(self,r,y):
            IAE = np.sum(np.abs(r - y))
            ISE = np.sum((r - y)**2)
            return IAE,ISE    

    def update_fisica(self, dt=1.0/60.0):
        self.ambiente.step(dt)

        # referencia - 90 graus (pi/2)
        r = self.r
        # posicao atual do pendulo
        y = self.pendulo_angulo()
        # calcula ação de controle
        uk = self.rodas_controle.control(r, y)

        criterios = self.calcula_metricas(r,y)

        # ação de controle na forma de torque angular nos motores
        self.M_dianteiro.rate = -uk
        self.M_traseiro.rate = -uk

        # remove objetos fora do espaço de simulação
        for shape in self.ambiente.shapes:
            if shape.body.position.y < -30:
                self.ambiente.remove(shape, shape.body)   
        return criterios              

if __name__ == "__main__":       
        N = 1000                   # quantidade de passos na simulação
        KP,KI,KD = 200, 150, 0     # parâmetros do controlador PID

        # Cria a simulação
        sim = Simulacao(KP=KP, KI=KI, KD=KD)

        # Séries históricas do ângulo (y) e do controle (u)
        y = []
        u = []
        # Séries históricas dos critérios ISE e IAE
        ISEr = []
        IAEr = []

        # Realiza a simulação em N passos
        for k in range(N):
            # armazena o ângulo atual
            y.append(sim.pendulo_angulo())
            #print([k, math.pi/2, sim.pendulo_angulo()])

            # aplica uma perturbação ao sistema
            if k == 10:
                sim.pendulo_fisica.apply_force_at_local_point((1000,0), (0,0))

            # atualiza a simulação
            criterios = sim.update_fisica()
            # atualiza os criterios de erro
            IAEr.append(criterios[0])
            ISEr.append(criterios[1])
            
            # o cálculo de controle é feito em update_fisica, armazena o resultado
            u.append(sim.M_dianteiro.rate)

        # Plot dos resultados
        t = np.arange(N)*(1.0/sim.FPS)
        r = np.ones_like(t)*sim.r
        y = np.array(y)
        u = np.array(u)

        fig, ax = plt.subplots(2, 1)
        plt.subplot(2, 1, 1)
        plt.plot(t, r, 'k--')
        plt.plot(t, y)
        plt.ylabel('Ângulo (rad)')

        plt.subplot(2, 1, 2)
        plt.plot(t, u)

        plt.xlabel('Tempo (s)')
        plt.ylabel('Controle (rad/s)')
        #fig.savefig("sim1.png")
        plt.show()

        # Calcula métricas
        IAE = 'IAE'
        ISE = 'ISE'
        registrador(IAE,IAEr)
        registrador(ISE,ISEr)
        


        #print('IAE: {}'.format(IAE))
        #print('ISE: {}'.format(ISE))